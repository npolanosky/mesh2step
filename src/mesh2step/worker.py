"""Out-of-process conversion worker, run under FreeCAD's Python.

The GUI (running under an ordinary Python with tkinter) shells out to this
module using FreeCAD's bundled interpreter, passing a JSON job file and reading
a JSON result file. This keeps the heavy FreeCAD/OCC dependency out of the GUI
process and lets the GUI be packaged without bundling FreeCAD.

Job JSON:
    {"mode": "inspect", "input": "part.stl", "config": {...}}
    {"mode": "convert", "input": "part.stl", "output": "part.step", "config": {...}}

Result JSON:
    {"ok": true, ...}  |  {"ok": false, "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path

from .config import UNIT_SCALE_MM, ConversionConfig

# Marker line the worker prints on stdout when it is cancelled (SIGTERM). The
# launcher (GUI/webapp) that killed the worker can match this to distinguish a
# clean user cancellation from a crash. Prefixed like a progress line so the
# existing line pumps carry it through unchanged.
CANCELLED_MARKER = "CANCELLED: worker terminated by signal"


def _install_cancellation_handler() -> None:
    """Make this worker cleanly killable, reaping every child it spawns.

    The conversion can spawn child subprocesses (the pymeshlab decimation runner
    in ``meshprep``), and a naive kill of just the worker PID would orphan them —
    leaving a stray FreeCAD/pymeshlab python grinding after the user cancels. Two
    measures guarantee a single kill reaps the whole tree:

    1. The worker becomes a **process-group leader** (``os.setpgrp``). A launcher
       that does ``os.killpg(os.getpgid(worker_pid), SIGTERM)`` then signals the
       worker *and* every child it started in one call — no PID bookkeeping.
    2. A **SIGTERM/SIGINT handler** forwards the signal to the rest of the process
       group (its children), prints the cancellation marker, and exits non-zero.
       This covers a launcher that kills only the worker PID: the handler fans the
       signal out to the children before exiting, so nothing is left orphaned.

    POSIX-only (no ``setpgrp``/``killpg`` on Windows); a no-op there, where the
    launcher uses a Job object / ``CREATE_NEW_PROCESS_GROUP`` to the same end.
    """
    if os.name != "posix":
        return
    try:
        os.setpgrp()  # new process group; this worker is its leader
    except OSError:
        pass

    def _handle(signum, _frame):
        # Fan the terminating signal out to the rest of our process group (the
        # children we spawned) so a decimation runner can't outlive us. We ARE the
        # group leader, so ``killpg`` hits us too — ignore the signal in ourselves
        # first so we survive to exit cleanly with a distinct code, then signal the
        # group, then exit. (Without the ignore, the default disposition would kill
        # us mid-handler and the exit code would be the signal, not our marker.)
        try:
            print(CANCELLED_MARKER, flush=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
        # Non-zero, distinct from a normal failure (1), so the launcher can tell
        # "cancelled" from "conversion failed".
        os._exit(143)  # 128 + SIGTERM(15), the conventional signalled-exit code

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):
            pass


def terminate_worker(proc, timeout: float = 5.0) -> bool:
    """Cleanly cancel a running worker subprocess, reaping its whole child tree.

    Launcher-side helper for the GUI/webapp: given the ``subprocess.Popen`` object
    returned when they spawned ``python -m mesh2step.worker``, this signals the
    worker's entire process group (the worker made itself the group leader via
    ``_install_cancellation_handler``), so the worker AND any child it spawned
    (the pymeshlab decimation runner) all receive SIGTERM in one call — no
    orphans. Escalates to SIGKILL on the group if the tree doesn't exit within
    ``timeout``.

    Returns True if the process exited within the timeout. On Windows (no process
    groups here) it falls back to ``proc.terminate()``/``proc.kill()``.
    """
    if proc is None or proc.poll() is not None:
        return True
    if os.name != "posix":
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            proc.kill()
        return proc.poll() is not None

    def _signal_group(sig) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            # Group already gone, or the worker never became a leader — fall back
            # to signalling just the worker PID.
            try:
                os.kill(proc.pid, sig)
            except OSError:
                pass

    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 - Popen.wait raises TimeoutExpired
        _signal_group(signal.SIGKILL)
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            pass
        return proc.poll() is not None


def _config_from(d: dict) -> ConversionConfig:
    """Build a ConversionConfig from a plain dict, ignoring unknown keys."""
    fields = ConversionConfig.__dataclass_fields__  # type: ignore[attr-defined]
    return ConversionConfig(**{k: v for k, v in (d or {}).items() if k in fields})


def run_inspect(job: dict) -> dict:
    """Measure the mesh without converting: bounding boxes + unit-scale table."""
    from .analysis import measure
    from .mesh_io import load_stl

    cfg = _config_from(job.get("config", {}))
    vertices, faces = load_stl(job["input"], weld_tol=cfg.weld_tol)
    info = measure(vertices)
    info["triangle_count"] = int(len(faces))

    # Input mesh health (non-manifold / self-intersections) via FreeCAD.
    try:
        from .meshprep import mesh_health

        info["health"] = mesh_health(job["input"])
    except Exception as exc:  # noqa: BLE001 - health check is best-effort
        info["health"] = {"error": str(exc)}

    # 3D locations of defects, so the preview can highlight problem regions.
    # Only bother when health already flags a problem (keeps clean meshes fast).
    info["problem_points"] = []
    try:
        if info.get("health", {}).get("self_intersections"):
            from .meshprep import problem_points

            info["problem_points"] = problem_points(job["input"])
    except Exception:  # noqa: BLE001 - best-effort
        pass

    # What the longest dimension becomes under each unit preset, so the GUI can
    # help the user pick the source units.
    longest = float(info["aabb"]["dimensions"][0]) if info["aabb"]["dimensions"] else 0.0
    info["unit_preview_mm"] = {u: longest * f for u, f in UNIT_SCALE_MM.items() if u != "inch"}
    return {"ok": True, "mode": "inspect", **info}


def _emit(msg: str) -> None:
    """Stream a progress line the GUI can pick up from stdout."""
    print(f"PROGRESS: {msg}", flush=True)


def run_convert(job: dict) -> dict:
    """Full STL -> STEP conversion."""
    from .pipeline import convert

    cfg = _config_from(job.get("config", {}))
    out = job.get("output") or str(Path(job["input"]).with_suffix(".step"))
    result = convert(job["input"], out, cfg, on_progress=_emit)
    return {
        "ok": True,
        "mode": "convert",
        "output": str(result.output_path),
        "outputs": [str(p) for p in (result.outputs or [result.output_path])],
        "method": result.method,
        "stats": result.stats,
    }


def run_tessellate(job: dict) -> dict:
    """Tessellate a STEP/BREP shape to a mesh file (for the deviation viewer)."""
    import FreeCAD  # type: ignore  # noqa: F401
    import Mesh  # type: ignore
    import Part  # type: ignore

    shape = Part.Shape()
    shape.read(job["input"])
    pts, tris = shape.tessellate(float(job.get("deflection", 0.1)))
    mesh = Mesh.Mesh()
    mesh.addFacets([(pts[a], pts[b], pts[c]) for a, b, c in tris])
    mesh.write(job["output"])
    return {"ok": True, "mode": "tessellate", "output": job["output"],
            "facets": int(mesh.CountFacets)}


_HANDLERS = {"inspect": run_inspect, "convert": run_convert, "tessellate": run_tessellate}


def run_job(job: dict) -> dict:
    handler = _HANDLERS.get(job.get("mode"))
    if handler is None:
        return {"ok": False, "error": f"unknown mode {job.get('mode')!r}"}
    try:
        return handler(job)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def main(argv: list[str] | None = None) -> int:
    # Make the worker a killable process-group leader that reaps its children on
    # SIGTERM (cancellation) — see the handler for why. Installed before any heavy
    # work so a cancel at any point leaves no orphaned decimation subprocess.
    _install_cancellation_handler()

    parser = argparse.ArgumentParser(prog="mesh2step.worker")
    parser.add_argument("--job", type=Path, help="path to job JSON (else stdin)")
    parser.add_argument("--result", type=Path, help="path to write result JSON (else stdout)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # utf-8-sig tolerates a BOM (e.g. job files written by PowerShell).
    raw = args.job.read_text(encoding="utf-8-sig") if args.job else sys.stdin.read()
    job = json.loads(raw)
    result = run_job(job)

    payload = json.dumps(result, indent=2)
    if args.result:
        args.result.write_text(payload)
    else:
        print(payload)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
