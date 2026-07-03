"""First-run self-provisioning: prep deps and (optionally) FreeCAD itself.

The end user should never have to run ``pip`` or hand-install anything. Two
things the conversion worker needs are provisioned automatically:

1. **Prep deps** (``pymeshlab`` + ``manifold3d``) — the mesh decimation and
   overlapping-body union that the fully-closed path relies on. They must be
   importable *by FreeCAD's bundled Python* (that's where the worker runs), but
   we must NOT pip-install into ``FreeCAD.app`` (that would break its code
   signature). Instead we install them with FreeCAD's own ``pip`` using
   ``--target`` into a per-user, per-(python-version+arch) directory under
   ``~/Library/Application Support/mesh2step/pydeps`` and inject that directory
   onto the worker's ``sys.path`` via ``PYTHONPATH``.

2. **FreeCAD** — if no system FreeCAD is found, we can download the official
   macOS arm64 release and unpack it into ``~/Applications`` (no admin needed).

Everything here is best-effort and reports progress through an optional
``log`` callable so the GUI can stream it into its console pane.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

# Prep-dep pins. manifold3d is REQUIRED (it resolves self-intersecting /
# overlapping-body meshes into one watertight solid — the load-bearing prep dep).
# pymeshlab is OPTIONAL: it only does planar decimation (run out-of-process in
# meshprep_runner), and the pipeline degrades gracefully without it. Both publish
# macOS arm64 (cp311) wheels.
#
# pymeshlab is capped to the 2023.12 line: newer wheels (2025.7) bundle a Qt
# whose plugin loader hard-crashes (SIGTRAP) at import on hardened macOS, which
# a try/except cannot catch. If even 2023.12 crashes on a given machine, the
# worker's crash-safe probe (see meshprep._pymeshlab_importable) simply skips
# decimation — so a broken pymeshlab never blocks a conversion.
#
# CRITICAL: these are installed with ``--no-deps``. The *only* dependency of
# either wheel is numpy, which FreeCAD already bundles (1.26.4). A plain
# ``pip install --target`` would pull numpy 2.x into the pydeps dir, and because
# that dir is prepended to PYTHONPATH it would SHADOW FreeCAD's numpy 1.26.4 —
# a 1.x/2.x C-ABI mismatch that crashes pymeshlab (and manifold3d) even when
# imported alone. So we never place numpy in pydeps; the runner subprocess uses
# FreeCAD's bundled numpy 1.26.4. ``_purge_shadowing_numpy`` self-heals installs
# that already have the bad numpy (e.g. from an earlier provisioning bug).
REQUIRED_PACKAGES = ["manifold3d>=3.0"]
OPTIONAL_PACKAGES = ["pymeshlab>=2023.12,<2025"]
PREP_PACKAGES = REQUIRED_PACKAGES + OPTIONAL_PACKAGES

# The module whose presence gates "prep deps ready" (manifold3d — the required
# one). pymeshlab importability is probed separately, crash-safely.
REQUIRED_MODULE = "manifold3d"

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _ssl_context():
    """A TLS context that verifies certs, using certifi's CA bundle if present.

    Some python.org macOS builds ship without a usable system CA bundle (the
    infamous ``CERTIFICATE_VERIFY_FAILED``); certifi — pulled in transitively by
    pyvista/requests — provides one. Falls back to the default context.
    """
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _log(log, msg: str) -> None:
    if log is not None:
        try:
            log(msg)
        except Exception:  # noqa: BLE001 - logging must never break provisioning
            pass


def support_dir() -> Path:
    """Per-user, writable base directory for mesh2step's own state."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "mesh2step"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home())) / "mesh2step"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "mesh2step"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_settings() -> dict:
    """Persisted app settings (small JSON in the support dir). Best-effort.

    Sanitised on load: empty/blank string values are dropped so a stale or
    half-written settings file can never override code defaults or runtime
    detection (an empty stored path must mean "not set", not "set to nothing").
    """
    p = support_dir() / "settings.json"
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items()
                        if not (v is None or (isinstance(v, str) and not v.strip()))}
    except Exception:  # noqa: BLE001 - a corrupt settings file must not block startup
        pass
    return {}


def save_settings(settings: dict) -> None:
    """Persist app settings. Best-effort: failures are silently ignored."""
    try:
        (support_dir() / "settings.json").write_text(
            json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _py_tag(freecad_python: str) -> str:
    """A ``pyMAJOR.MINOR-ARCH`` tag identifying a specific interpreter ABI.

    Prep deps ship compiled wheels, so the ``--target`` cache must be keyed by
    the FreeCAD Python's version *and* CPU arch — a cache built for cp311/arm64
    is useless to a cp310/x86_64 FreeCAD.
    """
    try:
        out = subprocess.check_output(
            [freecad_python, "-c",
             "import sys,platform;"
             "print(f'py{sys.version_info[0]}.{sys.version_info[1]}-{platform.machine()}')"],
            text=True, creationflags=_NO_WINDOW,
        ).strip()
        if out:
            return out
    except Exception:  # noqa: BLE001 - fall back to *our* interpreter's tag
        pass
    return f"py{sys.version_info[0]}.{sys.version_info[1]}-{platform.machine()}"


def pydeps_dir(freecad_python: str) -> Path:
    """The ``--target`` directory prep deps are installed into for this FreeCAD."""
    return support_dir() / "pydeps" / _py_tag(freecad_python)


def prep_deps_present(freecad_python: str) -> bool:
    """True if the REQUIRED prep module (manifold3d) imports under FreeCAD's
    Python with the provisioned dir injected.

    Only manifold3d gates readiness: pymeshlab is optional and, on some macOS
    builds, crashes on import — so importing it here would spuriously report the
    deps as missing (and could crash this probe too).
    """
    target = pydeps_dir(freecad_python)
    check = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import %s\n"
        "print('ok')\n" % (str(target), REQUIRED_MODULE)
    )
    try:
        out = subprocess.run(
            [freecad_python, "-c", check],
            capture_output=True, text=True, timeout=120, creationflags=_NO_WINDOW,
        )
        return out.returncode == 0 and "ok" in out.stdout
    except Exception:  # noqa: BLE001
        return False


def pymeshlab_usable(freecad_python: str) -> bool:
    """True if pymeshlab imports (without crashing) under FreeCAD's Python.

    Probed in a subprocess because a pymeshlab import can hard-crash (SIGTRAP)
    on some macOS builds — uncatchable in-process.
    """
    target = pydeps_dir(freecad_python)
    check = "import sys; sys.path.insert(0, r'%s'); import pymeshlab" % str(target)
    try:
        out = subprocess.run(
            [freecad_python, "-c", check],
            capture_output=True, text=True, timeout=120, creationflags=_NO_WINDOW,
        )
        return out.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _purge_shadowing_numpy(target: Path, log=None) -> bool:
    """Remove any ``numpy`` sitting in the pydeps dir so it can't shadow FreeCAD's.

    A ``numpy`` inside ``target`` is prepended to the worker's PYTHONPATH and
    shadows FreeCAD's bundled numpy 1.26.4; if it is a 2.x it crashes pymeshlab
    and manifold3d (1.x/2.x C-ABI mismatch). The prep wheels' only dependency is
    numpy, and FreeCAD already provides a compatible one, so numpy never belongs
    here. Returns True if anything was removed. Self-heals installs left in the
    bad state by an earlier ``pip install --target`` (without ``--no-deps``).
    """
    import shutil

    removed = False
    try:
        entries = list(target.glob("numpy")) + list(target.glob("numpy-*.dist-info")) \
            + list(target.glob("numpy.libs")) + list(target.glob("numpy-*.data"))
    except OSError:
        return False
    for p in entries:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed = True
        except OSError:
            pass
    if removed:
        _log(log, "  ✔ removed a shadowing numpy from pydeps (uses FreeCAD's "
                  "bundled numpy instead)")
    return removed


def ensure_prep_deps(freecad_python: str, log=None, force: bool = False) -> Path | None:
    """Ensure pymeshlab + manifold3d are importable by FreeCAD's Python.

    Installs them (once) into the per-user pydeps dir via FreeCAD's pip and
    returns that directory, or ``None`` if provisioning failed. Idempotent: a
    no-op when the deps already import.
    """
    target = pydeps_dir(freecad_python)
    if not force and prep_deps_present(freecad_python):
        # Self-heal an existing install that may carry a shadowing numpy from an
        # older provisioning run (the bug that crashed pymeshlab standalone).
        _purge_shadowing_numpy(target, log)
        return target

    target.mkdir(parents=True, exist_ok=True)
    _log(log, f"Provisioning prep deps (pymeshlab, manifold3d) → {target}")
    _log(log, "  (one-time download; uses FreeCAD's pip, does not touch FreeCAD.app)")
    # --no-deps: the only dependency is numpy, which FreeCAD bundles (1.26.4).
    # Pulling numpy into pydeps would shadow FreeCAD's and crash the prep deps.
    cmd = [
        freecad_python, "-m", "pip", "install",
        "--no-input", "--disable-pip-version-check", "--no-deps",
        "--target", str(target), *PREP_PACKAGES,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=_NO_WINDOW,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _log(log, "  pip: " + line)
        proc.wait(timeout=1800)
        if proc.returncode != 0:
            _log(log, f"  ⚠ pip exited {proc.returncode}; prep deps may be unavailable")
            return None
    except Exception as exc:  # noqa: BLE001
        _log(log, f"  ⚠ prep-dep install failed: {exc}")
        return None

    # Belt-and-braces: strip any numpy pip still deposited (older pip ignores
    # --no-deps for already-cached wheels in some edge cases).
    _purge_shadowing_numpy(target, log)

    if prep_deps_present(freecad_python):
        _log(log, "  ✔ manifold3d ready (watertight/self-intersection resolver)")
        if pymeshlab_usable(freecad_python):
            _log(log, "  ✔ pymeshlab ready (planar decimation)")
        else:
            _log(log, "  ⚠ pymeshlab unavailable on this macOS (its bundled Qt "
                      "crashes on import); decimation will be skipped — "
                      "conversions still work.")
        return target
    _log(log, "  ⚠ prep deps installed but manifold3d did not import cleanly")
    return None


def prep_env(freecad_python: str, base_env: dict | None = None) -> dict:
    """A process env for driving FreeCAD's bundled Python as a worker.

    Prepends, on ``PYTHONPATH``:
      * FreeCAD's own library dir — so ``import FreeCAD``/``Mesh`` works when we
        launch ``<freecad_python> -m mesh2step.worker`` ourselves (FreeCAD only
        wires that up automatically under freecadcmd / its GUI, not a plain
        ``python -m``), and
      * the provisioned pydeps dir (pymeshlab/manifold3d).

    Safe to call even when the deps aren't installed — it just points at the
    (possibly empty) target dirs, so the worker degrades gracefully.
    """
    from .freecad_env import freecad_lib_dir

    env = dict(base_env if base_env is not None else os.environ)
    parts = [str(pydeps_dir(freecad_python))]
    lib = freecad_lib_dir(freecad_python)
    if lib:
        parts.append(lib)
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


# --------------------------------------------------------------------------- #
# FreeCAD auto-download (macOS)                                                #
# --------------------------------------------------------------------------- #

_FREECAD_RELEASES_API = (
    "https://api.github.com/repos/FreeCAD/FreeCAD/releases/latest"
)


def _mac_asset_matches(name: str) -> bool:
    """True if a release asset name looks like a macOS arm64 FreeCAD bundle."""
    n = name.lower()
    if not (n.endswith(".dmg") or n.endswith(".zip")):
        return False
    if "macos" not in n and "osx" not in n and "mac" not in n:
        return False
    # arm64 / Apple-Silicon builds are named ...arm64... or ...aarch64...
    return "arm64" in n or "aarch64" in n or "apple" in n


def freecad_download_url(arch: str | None = None) -> str | None:
    """Resolve the official FreeCAD macOS arm64 download URL, or None.

    Queries the GitHub *latest release* API and picks the arm64 macOS asset.
    Network access is required; returns None on any failure so callers can show
    a manual-install message.
    """
    arch = arch or platform.machine()
    try:
        req = urllib.request.Request(
            _FREECAD_RELEASES_API,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "mesh2step-provisioner"},
        )
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None

    assets = data.get("assets", [])
    # Prefer an arch-specific arm64 asset; the sniffing above already narrows it.
    for asset in assets:
        name = asset.get("name", "")
        if _mac_asset_matches(name):
            return asset.get("browser_download_url")
    return None


def install_freecad(log=None, dry_run: bool = False) -> str | None:
    """Download and install FreeCAD into ~/Applications (no admin). macOS only.

    Returns the path to the installed ``FreeCAD.app`` on success, or None on
    failure. With ``dry_run=True`` it only resolves + reports the URL (used by
    tests and for a bandwidth-free smoke check).
    """
    if sys.platform != "darwin":
        _log(log, "Auto-install of FreeCAD is only implemented for macOS.")
        return None

    url = freecad_download_url()
    if not url:
        _log(log, "Could not resolve a FreeCAD download URL (no network, or no "
                  "matching macOS arm64 asset). Install FreeCAD 0.20+ manually "
                  "from https://www.freecad.org/downloads.php")
        return None
    _log(log, f"FreeCAD download URL: {url}")
    if dry_run:
        return None

    dest_apps = Path.home() / "Applications"
    dest_apps.mkdir(parents=True, exist_ok=True)
    import tempfile

    suffix = ".dmg" if url.lower().endswith(".dmg") else ".zip"
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / ("FreeCAD" + suffix)
        _log(log, "Downloading FreeCAD… (this can take a few minutes)")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mesh2step-provisioner"})
            with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp, \
                    open(pkg, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            _log(log, f"⚠ Download failed: {exc}")
            return None

        try:
            if suffix == ".dmg":
                app = _install_from_dmg(pkg, dest_apps, log)
            else:
                app = _install_from_zip(pkg, dest_apps, log)
        except Exception as exc:  # noqa: BLE001
            _log(log, f"⚠ Install failed: {exc}")
            return None

    if app and Path(app).is_dir():
        _log(log, f"✔ FreeCAD installed → {app}")
        return app
    _log(log, "⚠ FreeCAD install did not produce a FreeCAD.app")
    return None


def _install_from_dmg(dmg: Path, dest_apps: Path, log) -> str | None:
    """Mount a .dmg, copy FreeCAD.app out, detach. Returns the installed path."""
    _log(log, "Mounting disk image…")
    out = subprocess.check_output(
        ["hdiutil", "attach", str(dmg), "-nobrowse", "-mountrandom", "/tmp"],
        text=True,
    )
    mount_point = None
    for line in out.splitlines():
        parts = line.split("\t")
        if parts and parts[-1].startswith("/"):
            mount_point = parts[-1].strip()
    if not mount_point:
        _log(log, "⚠ Could not determine the mount point.")
        return None
    try:
        srcs = list(Path(mount_point).glob("FreeCAD*.app"))
        if not srcs:
            _log(log, "⚠ No FreeCAD.app inside the disk image.")
            return None
        src = srcs[0]
        dest = dest_apps / "FreeCAD.app"
        _log(log, f"Copying {src.name} → {dest}")
        if dest.exists():
            import shutil
            shutil.rmtree(dest, ignore_errors=True)
        subprocess.check_call(["ditto", str(src), str(dest)])
        return str(dest)
    finally:
        subprocess.run(["hdiutil", "detach", mount_point, "-quiet"], check=False)


def _install_from_zip(zpath: Path, dest_apps: Path, log) -> str | None:
    """Unzip an archive containing FreeCAD.app into ~/Applications."""
    import shutil
    import tempfile
    import zipfile

    _log(log, "Unpacking archive…")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(tmp)
        srcs = list(Path(tmp).rglob("FreeCAD*.app"))
        if not srcs:
            _log(log, "⚠ No FreeCAD.app inside the archive.")
            return None
        dest = dest_apps / "FreeCAD.app"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(srcs[0]), str(dest))
        return str(dest)
