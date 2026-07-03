"""Failure corpus: save inputs that fail to convert, for regression testing.

When enabled (GUI checkbox / ``--save-failures``), any conversion that does not
end in a single watertight solid — including worker crashes — COPIES the
original input STL into a corpus directory, sorted by failure category, and
records it in a manifest. Future versions of the program can then be swept
against exactly the meshes that broke earlier ones. Files are never moved and
never auto-deleted: once a mesh is in the corpus it stays there, and later
*passing* results are appended to its manifest history (that is the regression
value — "this used to fail, now it converts").

Layout::

    <dest>/
      manifest.json           one entry per unique file (keyed by sha256)
      not_watertight/foo.stl  copies, one subdirectory per failure category
      crash/bar.stl
      ...

Destination default: ``<repo>/tests/data/community/failures/`` when running
from a source checkout (detected by walking up from this file), else the
per-user support dir (``failed_corpus/``) — the frozen app can't write into a
repo it doesn't have. See :func:`resolve_dest`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Failure taxonomy                                                             #
# --------------------------------------------------------------------------- #

# The single place mapping a conversion outcome to a category (= subdirectory).
# Keep categories few and stable — they are directory names in the corpus. To
# extend, add a check in classify_result() and list the new name here.
CATEGORIES = (
    "crash",                  # worker died / exception / timeout / no result
    "boolean_failed",         # boolean clean-up ran but couldn't close the solid
    "reconstruction_failed",  # surface reconstruction errored (faceted fallback), still open
    "not_watertight",         # finished, but the result has open shells (catch-all)
    "faceted_improvable",     # user-flagged: watertight, but faceted surfaces remain
)


def classify_result(result: dict | None) -> str | None:
    """Failure category for a worker/pipeline result, or ``None`` for a pass.

    A *pass* is a single watertight solid (``stats.is_solid``). Everything else
    is a failure worth keeping. ``result`` is the worker's JSON dict
    (``{"ok": bool, "error": ..., "stats": {...}}``); a missing/falsy result
    counts as a crash.
    """
    if not result or not result.get("ok"):
        return "crash"
    stats = result.get("stats") or {}
    if stats.get("is_solid"):
        return None
    blob = " ".join(
        list(stats.get("warnings") or []) + list(stats.get("warnings_extra") or [])
    ).lower()
    if "boolean clean-up" in blob:
        return "boolean_failed"
    if stats.get("reconstruction_error"):
        return "reconstruction_failed"
    return "not_watertight"


def _quality_stats(result: dict | None) -> dict:
    """Structured quality scores to store on a manifest entry/history event.

    Pulls the few numbers worth querying later (RTAF, skipped_facets, quality
    verdict) out of the conversion stats. Kept small and stable; absent keys are
    simply omitted so old and new manifests stay compatible."""
    stats = (result or {}).get("stats") or {}
    out: dict = {}
    for k in ("rtaf", "skipped_facets", "quality"):
        if stats.get(k) is not None:
            out[k] = stats[k]
    return out


def _error_summary(result: dict | None) -> str:
    """Short human-readable reason string for the manifest."""
    if not result:
        return "no result produced"
    if not result.get("ok"):
        return str(result.get("error") or "unknown error")[:500]
    stats = result.get("stats") or {}
    parts = list(stats.get("warnings") or []) + list(stats.get("warnings_extra") or [])
    if stats.get("reconstruction_error"):
        parts.append(f"reconstruction_error: {stats['reconstruction_error']}")
    return ("; ".join(parts))[:500] or "not a single watertight solid"


# --------------------------------------------------------------------------- #
# Destination + manifest                                                       #
# --------------------------------------------------------------------------- #


def repo_community_dir() -> Path | None:
    """``tests/data/community`` of the source checkout we're running from, or None.

    Walks up from this file — a frozen app (package copied into the bundle)
    finds nothing and falls back to the per-user dir.
    """
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tests" / "data" / "community"
        if cand.is_dir():
            return cand
    return None


def resolve_dest(explicit: str | None = None) -> Path:
    """The failure-corpus directory: explicit > repo checkout > per-user dir."""
    if explicit:
        return Path(explicit).expanduser()
    community = repo_community_dir()
    if community is not None:
        return community / "failures"
    from .provision import support_dir

    return support_dir() / "failed_corpus"


def _manifest_path(dest: Path) -> Path:
    return dest / "manifest.json"


def _load_manifest(dest: Path) -> dict:
    p = _manifest_path(dest)
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt manifest must not block saving
        pass
    return {"files": {}}


def _save_manifest(dest: Path, manifest: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tmp = _manifest_path(dest).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(_manifest_path(dest))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_existing_copy(dest: Path, size: int, sha: str) -> Path | None:
    """An identical file already anywhere under ``dest`` (hash dedupe).

    The manifest is the primary index; this filesystem scan additionally covers
    a lost/edited manifest. Only same-size files are hashed, so it stays cheap.
    """
    if not dest.is_dir():
        return None
    for p in dest.rglob("*.stl"):
        try:
            if p.stat().st_size == size and _sha256(p) == sha:
                return p
        except OSError:
            continue
    return None


def _unique_target(cat_dir: Path, name: str) -> Path:
    """A free filename in ``cat_dir`` (same name + different content -> suffix)."""
    target = cat_dir / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    for i in range(2, 1000):
        target = cat_dir / f"{stem}-{i}{suffix}"
        if not target.exists():
            return target
    raise RuntimeError(f"could not find a free name for {name} in {cat_dir}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version() -> str:
    from . import DISPLAY_VERSION

    return DISPLAY_VERSION


# --------------------------------------------------------------------------- #
# Recording                                                                    #
# --------------------------------------------------------------------------- #


def record_result(input_path: str | Path, result: dict | None,
                  dest: str | Path | None = None, log=None) -> dict | None:
    """Record a conversion outcome against the failure corpus.

    * Failure (see :func:`classify_result`): copy ``input_path`` into
      ``<dest>/<category>/`` (unless an identical file is already in the corpus)
      and add/extend its manifest entry.
    * Pass: if the file is already in the corpus, append a "pass" history entry
      (the file is kept — that's the regression record); otherwise do nothing.

    Returns a small summary dict (``action`` in {"saved", "known_failure",
    "pass_recorded"}) or ``None`` when there was nothing to record. Best-effort:
    any internal error is reported via ``log`` and swallowed — corpus book-
    keeping must never break a conversion.
    """
    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:  # noqa: BLE001
                pass

    try:
        return _record(Path(input_path), result, resolve_dest(
            str(dest) if dest is not None else None), _log)
    except Exception as exc:  # noqa: BLE001
        _log(f"⚠ failure-corpus bookkeeping failed: {exc}")
        return None


def record_flag(input_path: str | Path, result: dict | None,
                dest: str | Path | None = None, log=None) -> dict | None:
    """User-initiated flag: keep a *watertight* result whose remaining faceted
    surfaces should be reconstructed better in a future version.

    Same corpus, dedupe and manifest rules as :func:`record_result`, under the
    ``faceted_improvable`` category, with the quality-report surface stats in
    the manifest entry so the improvement is measurable later.
    """
    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:  # noqa: BLE001
                pass

    stats = (result or {}).get("stats") or {}
    parts = ["user-flagged: faceted surfaces to improve"]
    if (result or {}).get("method"):
        parts.append(f"method={result['method']}")
    for k in ("faces_in", "faces_out", "planar_faces", "cylinder_faces",
              "gap_faces", "gap_patches", "skipped_facets", "rtaf", "quality"):
        if k in stats:
            parts.append(f"{k}={stats[k]}")
    for w in stats.get("warnings") or []:
        parts.append(w)
    summary = "; ".join(parts)[:500]
    try:
        return _record(Path(input_path), result, resolve_dest(
            str(dest) if dest is not None else None), _log,
            category="faceted_improvable", outcome="flagged", summary=summary)
    except Exception as exc:  # noqa: BLE001
        _log(f"⚠ failure-corpus bookkeeping failed: {exc}")
        return None


def _record(src: Path, result: dict | None, dest: Path, log,
            category: str | None = "auto", outcome: str = "fail",
            summary: str | None = None) -> dict | None:
    if not src.is_file():
        return None
    if category == "auto":
        category = classify_result(result)
    sha = _sha256(src)
    manifest = _load_manifest(dest)
    files = manifest.setdefault("files", {})
    entry = files.get(sha)
    stamp = {"date": _now(), "version": _version()}

    if category is None:
        if entry is None:
            return None  # pass on a file we're not tracking — nothing to do
        entry.setdefault("history", []).append({**stamp, "outcome": "pass"})
        _save_manifest(dest, manifest)
        log(f"Failure corpus: {src.name} now converts watertight — pass recorded "
            f"(file kept for regression).")
        return {"action": "pass_recorded", "sha256": sha, "file": entry.get("file")}

    if summary is None:
        summary = _error_summary(result)
    # Structured quality scores on the entry, so a flagged file carries its RTAF
    # (and skipped_facets / quality verdict) as queryable fields — regression
    # tooling can compare a later pass's score without parsing the summary text.
    quality_stats = _quality_stats(result)
    fail_event = {**stamp, "outcome": outcome, "category": category, "error": summary,
                  **quality_stats}

    if entry is not None:
        entry.setdefault("history", []).append(fail_event)
        _save_manifest(dest, manifest)
        log(f"Failure corpus: {src.name} already saved ({entry['file']}) — "
            f"recorded repeat {category}.")
        return {"action": "known_failure", "sha256": sha, "file": entry.get("file")}

    existing = _find_existing_copy(dest, src.stat().st_size, sha)
    if existing is not None:
        rel = str(existing.relative_to(dest))
    else:
        cat_dir = dest / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        target = _unique_target(cat_dir, src.name)
        shutil.copy2(src, target)  # COPY — the user's file is never moved
        rel = str(target.relative_to(dest))
        log(f"Failure corpus: saved {src.name} → {dest / rel}  [{category}]")

    files[sha] = {
        "file": rel,
        "original_path": str(src.resolve()),
        "original_name": src.name,
        "sha256": sha,
        "first_seen": stamp["date"],
        "category": category,
        "error": summary,
        "version": stamp["version"],
        **quality_stats,
        "history": [fail_event],
    }
    _save_manifest(dest, manifest)
    return {"action": "saved", "sha256": sha, "file": rel, "category": category}


# --------------------------------------------------------------------------- #
# Corpus enumeration (for sweeps / regression tooling)                         #
# --------------------------------------------------------------------------- #


def iter_corpus(community_dir: str | Path | None = None) -> list[Path]:
    """Every STL in the community corpus: top level + ``failures/**``.

    The canonical enumeration for regression sweeps — new tooling should call
    this instead of globbing, so saved failures are always included.
    """
    root = Path(community_dir) if community_dir else repo_community_dir()
    if root is None or not Path(root).is_dir():
        return []
    root = Path(root)
    tops = sorted(root.glob("*.stl"))
    fails = sorted((root / "failures").rglob("*.stl")) if (root / "failures").is_dir() else []
    return tops + fails
