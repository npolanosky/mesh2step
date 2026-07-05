"""Drive the FreeCAD conversion worker out-of-process, streaming progress.

Mirrors ``gui.run_worker`` (same subprocess + PYTHONPATH wiring via
``provision.prep_env``) but exposes a line callback so the web layer can push
progress/log lines to the browser over SSE. The web process never imports
FreeCAD; all CAD work happens in the worker subprocess.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_IS_WIN = sys.platform == "win32"


class CancelledError(RuntimeError):
    """Raised when a worker was killed by an explicit cancel request."""


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort hard kill of ``proc`` **and its children**.

    ``run_worker`` starts the worker in its own process group / session (see
    ``_popen_kwargs``) so the FreeCAD child spawned by ``mesh2step.worker`` dies
    too. On POSIX we ``killpg`` the group; on Windows we use ``taskkill /T``.
    Independent of any SIGTERM cleanliness in the worker itself — this is the
    guaranteed teardown path for cancel.
    """
    if proc.poll() is not None:  # already dead
        return
    try:
        if _IS_WIN:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            # Kill the whole group. The process was started with setsid, so its
            # pgid == its pid; killpg reaches the meshprep/FreeCAD grandchild too.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:  # noqa: BLE001 - teardown must never raise
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _popen_kwargs() -> dict:
    """Platform kwargs that isolate the worker into its own killable group."""
    if _IS_WIN:
        # CREATE_NEW_PROCESS_GROUP so taskkill /T reaches the whole tree.
        return {"creationflags": _NO_WINDOW | 0x00000200}  # CREATE_NEW_PROCESS_GROUP
    # New session => new process group whose pgid == child pid.
    return {"start_new_session": True}


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
               on_start=None, timeout: float = 1800) -> dict:
    """Run one worker job, streaming stdout lines to ``on_line`` (if given).

    ``on_start(proc)``, if given, is called with the live :class:`subprocess.Popen`
    right after spawn — the web layer registers it so a cancel request can kill
    the whole process tree (see :func:`kill_process_tree`). The worker runs in its
    own process group / session so the FreeCAD/meshprep grandchild dies with it.

    Returns the worker's parsed result dict. Raises :class:`CancelledError` if the
    worker was killed by a cancel, or :class:`WorkerError` if it produced no
    result file (crash before writing).
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
            **_popen_kwargs(),
        )
        if on_start is not None:
            on_start(proc)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line and on_line is not None:
                on_line(line)
        proc.wait(timeout=timeout)
        # A negative returncode is death-by-signal (SIGKILL/SIGTERM from a
        # cancel). No result file after a signal => treat as cancellation, not a
        # generic worker crash, so the UI shows "cancelled" rather than "failed".
        if not res_file.exists():
            if proc.returncode is not None and proc.returncode < 0:
                raise CancelledError(
                    f"worker cancelled (signal {-proc.returncode})")
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


def tessellate_typed(step_path: str | Path, out_blob: str | Path,
                     out_meta: str | Path, freecad_python: str, *,
                     deflection: float = 0.1, timeout: float = 600) -> None:
    """Per-face typed tessellation for the surface-provenance viewer overlay.

    Runs :mod:`mesh2step.webapp.stepmesh` under FreeCAD's Python (same env
    wiring as the conversion worker) to write an M2SM blob whose vertex
    colours encode each face's surface category, plus a JSON legend sidecar.
    Raises :class:`WorkerError` with the script's stderr on failure.
    """
    proc = subprocess.run(
        [freecad_python, "-m", "mesh2step.webapp.stepmesh",
         "--step", str(step_path), "--out-blob", str(out_blob),
         "--out-meta", str(out_meta), "--deflection", str(float(deflection))],
        env=_worker_env(freecad_python),
        capture_output=True, text=True, timeout=timeout,
        creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0 or not Path(out_blob).is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        raise WorkerError(f"typed tessellation failed: {detail}")
