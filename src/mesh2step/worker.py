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
import sys
import traceback
from pathlib import Path

from .config import UNIT_SCALE_MM, ConversionConfig


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

    # What the longest dimension becomes under each unit preset, so the GUI can
    # help the user pick the source units.
    longest = float(info["aabb"]["dimensions"][0]) if info["aabb"]["dimensions"] else 0.0
    info["unit_preview_mm"] = {u: longest * f for u, f in UNIT_SCALE_MM.items() if u != "inch"}
    return {"ok": True, "mode": "inspect", **info}


def run_convert(job: dict) -> dict:
    """Full STL -> STEP conversion."""
    from .pipeline import convert

    cfg = _config_from(job.get("config", {}))
    out = job.get("output") or str(Path(job["input"]).with_suffix(".step"))
    result = convert(job["input"], out, cfg)
    return {
        "ok": True,
        "mode": "convert",
        "output": str(result.output_path),
        "method": result.method,
        "stats": result.stats,
    }


_HANDLERS = {"inspect": run_inspect, "convert": run_convert}


def run_job(job: dict) -> dict:
    handler = _HANDLERS.get(job.get("mode"))
    if handler is None:
        return {"ok": False, "error": f"unknown mode {job.get('mode')!r}"}
    try:
        return handler(job)
    except Exception as exc:  # noqa: BLE001 - report any failure as JSON
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def main(argv: list[str] | None = None) -> int:
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
