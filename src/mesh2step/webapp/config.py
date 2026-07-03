"""Runtime configuration for the web app.

All knobs are plain attributes with environment-variable fallbacks so the app
can be configured from a systemd unit without a config file. See docs/WEBAPP.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    """Per-user working directory for jobs, uploads and results.

    Overridable with ``$MESH2STEP_WEB_DIR``. Kept separate from the desktop
    app's support dir so the two never fight over the same files.
    """
    env = os.environ.get("MESH2STEP_WEB_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".mesh2step-web"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class WebConfig:
    """Everything the server needs to run.

    ``freecad_python`` is resolved lazily (via ``freecad_env.find_freecad_python``)
    if left as ``None`` — the server logs a clear warning when it can't be found
    rather than refusing to start, so the UI still loads and can report it.
    """

    host: str = field(default_factory=lambda: os.environ.get("MESH2STEP_WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("MESH2STEP_WEB_PORT", 8765))
    data_dir: Path = field(default_factory=_default_data_dir)
    # How many conversions may run at once. 1 (default) keeps FreeCAD/OCC off
    # each other's toes; a beefy server can raise it.
    concurrency: int = field(default_factory=lambda: _env_int("MESH2STEP_WEB_CONCURRENCY", 1))
    # Explicit FreeCAD python; None -> auto-detect at startup.
    freecad_python: str | None = field(
        default_factory=lambda: os.environ.get("MESH2STEP_FREECAD_PYTHON") or None
    )
    # Deviation-heatmap tessellation deflection (mm). Matches viewer.py's default.
    deflection: float = field(
        default_factory=lambda: float(os.environ.get("MESH2STEP_WEB_DEFLECTION", "0.1"))
    )
    # Per-conversion wall-clock ceiling (seconds).
    convert_timeout: float = field(
        default_factory=lambda: float(os.environ.get("MESH2STEP_WEB_TIMEOUT", "1800"))
    )
    # Max upload size (bytes). Rejected with 413 above this. Default 200 MB.
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("MESH2STEP_WEB_MAX_UPLOAD", 200 * 1024 * 1024)
    )
    # Failure-corpus destination (None -> failstore.resolve_dest default).
    failures_dir: str | None = field(
        default_factory=lambda: os.environ.get("MESH2STEP_WEB_FAILURES_DIR") or None
    )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def ensure_dirs(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
