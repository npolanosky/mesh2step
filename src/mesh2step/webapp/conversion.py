"""Drive the FreeCAD conversion worker out-of-process, streaming progress.

Mirrors ``gui.run_worker`` (same subprocess + PYTHONPATH wiring via
``provision.prep_env``) but exposes a line callback so the web layer can push
progress/log lines to the browser over SSE. The web process never imports
FreeCAD; all CAD work happens in the worker subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _package_src() -> str:
    """Directory to put on PYTHONPATH so FreeCAD's Python can import mesh2step."""
    # webapp/ -> mesh2step/ -> src/
    return str(Path(__file__).resolve().parents[2])


class WorkerError(RuntimeError):
    pass


def _worker_env(freecad_python: str) -> dict:
    from .. import provision

    env = provision.prep_env(freecad_python)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _package_src() + (os.pathsep + existing if existing else "")
    return env


def run_worker(job: dict, freecad_python: str, *, on_line=None,
               timeout: float = 1800) -> dict:
    """Run one worker job, streaming stdout lines to ``on_line`` (if given).

    Returns the worker's parsed result dict. Raises :class:`WorkerError` if the
    worker produced no result file (crash before writing).
    """
    with tempfile.TemporaryDirectory() as tmp:
        job_file = Path(tmp) / "job.json"
        res_file = Path(tmp) / "result.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")

        proc = subprocess.Popen(
            [freecad_python, "-m", "mesh2step.worker",
             "--job", str(job_file), "--result", str(res_file)],
            env=_worker_env(freecad_python),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=_NO_WINDOW,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line and on_line is not None:
                on_line(line)
        proc.wait(timeout=timeout)
        if not res_file.exists():
            raise WorkerError(f"worker produced no result (exit {proc.returncode})")
        return json.loads(res_file.read_text(encoding="utf-8-sig"))


def tessellate_step(step_path: str | Path, out_mesh: str | Path,
                    freecad_python: str, *, deflection: float = 0.1,
                    timeout: float = 600) -> dict:
    """Tessellate a STEP to an STL mesh (worker ``tessellate`` mode)."""
    job = {"mode": "tessellate", "input": str(step_path),
           "output": str(out_mesh), "deflection": float(deflection)}
    result = run_worker(job, freecad_python, timeout=timeout)
    if not result.get("ok"):
        raise WorkerError(result.get("error", "tessellation failed"))
    return result
