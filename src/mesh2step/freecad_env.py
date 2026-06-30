"""Locate a system FreeCAD install and make ``import FreeCAD`` work.

FreeCAD ships its own Python. Either run under that interpreter (``freecadcmd``)
or call :func:`ensure_freecad` to add FreeCAD's ``bin/`` to ``sys.path`` so a
normal venv can ``import FreeCAD``. We never pip-install FreeCAD.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Common install locations per platform. Globs are expanded in order.
_CANDIDATE_GLOBS = {
    "win32": [
        r"C:\Program Files\FreeCAD*\bin",
        r"C:\Program Files (x86)\FreeCAD*\bin",
    ],
    "darwin": [
        "/Applications/FreeCAD.app/Contents/Resources/lib",
        "/Applications/FreeCAD*.app/Contents/Resources/lib",
    ],
    "linux": [
        "/usr/lib/freecad/lib",
        "/usr/lib/freecad-python3/lib",
        "/usr/local/lib/freecad/lib",
        "/opt/freecad*/lib",
        str(Path.home() / "squashfs-root/usr/lib"),  # extracted AppImage
    ],
}


def _candidate_dirs(explicit: str | None) -> list[Path]:
    dirs: list[Path] = []
    if explicit:
        dirs.append(Path(explicit))
    env = os.environ.get("FREECAD_BIN") or os.environ.get("FREECAD_LIB")
    if env:
        dirs.append(Path(env))
    for pattern in _CANDIDATE_GLOBS.get(sys.platform, _CANDIDATE_GLOBS["linux"]):
        base = Path(pattern).anchor or "/"
        rel = Path(pattern).relative_to(base) if Path(pattern).is_absolute() else Path(pattern)
        try:
            dirs.extend(sorted(Path(base).glob(str(rel)), reverse=True))
        except (OSError, ValueError):
            continue
    return dirs


def ensure_freecad(explicit: str | None = None):
    """Import and return the ``FreeCAD`` module, injecting it onto ``sys.path``.

    Raises a helpful ``ImportError`` if no install can be found.
    """
    try:  # already importable (e.g. running under freecadcmd)
        import FreeCAD  # type: ignore

        return FreeCAD
    except ImportError:
        pass

    tried: list[str] = []
    for d in _candidate_dirs(explicit):
        tried.append(str(d))
        if not d.is_dir():
            continue
        sys.path.insert(0, str(d))
        try:
            import FreeCAD  # type: ignore

            return FreeCAD
        except ImportError:
            sys.path.pop(0)

    raise ImportError(
        "Could not import FreeCAD. Install FreeCAD 0.20+ and either run under "
        "its interpreter (freecadcmd) or pass --freecad-bin / set $FREECAD_BIN "
        "to its bin/ (Windows) or lib/ (macOS/Linux) directory.\n"
        f"Searched: {tried or '[no candidates]'}"
    )


# Glob patterns for FreeCAD's bundled Python *executable* (used by the GUI to
# launch the conversion worker out-of-process).
_PYTHON_GLOBS = {
    "win32": [
        r"C:\Program Files\FreeCAD*\bin\python.exe",
        r"C:\Program Files (x86)\FreeCAD*\bin\python.exe",
    ],
    "darwin": [
        "/Applications/FreeCAD.app/Contents/Resources/bin/python*",
        "/Applications/FreeCAD*.app/Contents/Resources/bin/python*",
    ],
    "linux": [
        "/usr/bin/freecadcmd",
        "/usr/bin/freecad-python3",
        "/opt/freecad*/bin/python*",
        str(Path.home() / "squashfs-root/usr/bin/python*"),
    ],
}


def find_freecad_python(explicit: str | None = None) -> str | None:
    """Locate FreeCAD's bundled Python executable, or ``None`` if not found.

    ``explicit`` may point at FreeCAD's bin/ dir or directly at the executable.
    """
    candidates: list[Path] = []
    if explicit:
        p = Path(explicit)
        candidates += [p, p / "python.exe", p / "python", p / "bin" / "python.exe"]
    env = os.environ.get("FREECAD_PYTHON")
    if env:
        candidates.append(Path(env))
    for pattern in _PYTHON_GLOBS.get(sys.platform, _PYTHON_GLOBS["linux"]):
        pat = Path(pattern)
        base = pat.anchor or "/"
        rel = pat.relative_to(base) if pat.is_absolute() else pat
        try:
            candidates += sorted(Path(base).glob(str(rel)), reverse=True)
        except (OSError, ValueError):
            continue
    for c in candidates:
        if c.is_file():
            return str(c)
    return None
