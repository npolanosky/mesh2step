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
