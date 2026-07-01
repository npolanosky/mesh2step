"""STL/STEP overlay viewer with a deviation heatmap.

Renders the input STL and the output STEP together, colouring the STEP by its
geometric deviation from the mesh (blue = on-surface, red = far). Also a
development/QA tool: it shows exactly which reconstructed faces drifted.

Runs under an ordinary Python with pyvista. The STEP is first tessellated to a
mesh by FreeCAD's Python (worker ``tessellate`` mode); pyvista then computes
point-to-surface distance and renders. Requires ``pip install ".[viewer]"``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .freecad_env import find_freecad_python


def _package_src() -> str:
    if getattr(sys, "frozen", False):
        return str(Path(getattr(sys, "_MEIPASS", ".")) / "mesh2step_src")
    return str(Path(__file__).resolve().parent.parent)


def _tessellate_step(step_path: str, out_mesh: str, deflection: float,
                     freecad_python: str) -> None:
    """Ask FreeCAD's Python to tessellate the STEP into a mesh file."""
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "job.json"
        res = Path(tmp) / "res.json"
        job.write_text(json.dumps({
            "mode": "tessellate", "input": step_path,
            "output": out_mesh, "deflection": deflection}), encoding="utf-8")
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _package_src() + (os.pathsep + existing if existing else "")
        no_window = 0x08000000 if sys.platform == "win32" else 0
        proc = subprocess.run(
            [freecad_python, "-m", "mesh2step.worker",
             "--job", str(job), "--result", str(res)],
            env=env, capture_output=True, text=True, creationflags=no_window,
        )
        if not res.exists():
            raise RuntimeError(
                f"tessellation worker failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        result = json.loads(res.read_text(encoding="utf-8-sig"))
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "tessellation failed"))


def deviation_stats(step_poly) -> dict:
    """max / rms / p95 / mean absolute deviation (mm)."""
    import numpy as np

    dev = np.abs(step_poly["deviation"])
    return {
        "max": float(dev.max()),
        "rms": float(np.sqrt(np.mean(dev**2))),
        "p95": float(np.percentile(dev, 95)),
        "mean": float(dev.mean()),
    }


def build_scene(stl_path: str, step_path: str, deflection: float = 0.1,
                freecad_python: str | None = None):
    """Return (stl_poly, step_poly_with_deviation, stats). No rendering."""
    import numpy as np  # noqa: F401
    import pyvista as pv

    fc = freecad_python or find_freecad_python()
    if not fc:
        raise RuntimeError("FreeCAD Python not found (needed to tessellate the STEP)")

    tmp = tempfile.mkdtemp(prefix="mesh2step_view_")
    step_mesh = os.path.join(tmp, "step.stl")
    _tessellate_step(step_path, step_mesh, deflection, fc)

    stl = pv.read(stl_path)
    step = pv.read(step_mesh)
    # Signed distance from each STEP point to the STL surface; |·| is deviation.
    step = step.compute_implicit_distance(stl)
    step["deviation"] = abs(step["implicit_distance"])
    return stl, step, deviation_stats(step)


def view(stl_path: str, step_path: str, deflection: float = 0.1,
         clamp: float | None = None, freecad_python: str | None = None,
         screenshot: str | None = None) -> dict:
    """Open an interactive overlay window (or save a screenshot); return stats."""
    import pyvista as pv

    stl, step, stats = build_scene(stl_path, step_path, deflection, freecad_python)
    hi = clamp if clamp is not None else max(stats["p95"], stats["max"] * 0.5, 1e-6)

    off = screenshot is not None
    plotter = pv.Plotter(off_screen=off, window_size=[1100, 800])
    plotter.add_text("mesh2step — STL (grey) vs STEP deviation heatmap", font_size=10)
    plotter.add_mesh(stl, color="lightgray", opacity=0.25, label="input STL")
    plotter.add_mesh(step, scalars="deviation", cmap="jet", clim=[0.0, hi],
                     scalar_bar_args={"title": "deviation (mm)"}, label="STEP")
    plotter.add_legend()
    plotter.camera_position = "iso"
    print(f"deviation (mm): max={stats['max']:.4f}  rms={stats['rms']:.4f}  "
          f"p95={stats['p95']:.4f}  mean={stats['mean']:.4f}")
    if off:
        plotter.screenshot(screenshot)
        plotter.close()
    else:
        plotter.show()
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mesh2step-view",
                                description="Overlay STL + STEP with a deviation heatmap.")
    p.add_argument("stl", help="input STL mesh")
    p.add_argument("step", help="output STEP file")
    p.add_argument("--deflection", type=float, default=0.1, help="STEP tessellation (mm)")
    p.add_argument("--clamp", type=float, default=None, help="max deviation for the colour scale (mm)")
    p.add_argument("--screenshot", default=None, help="render off-screen to this PNG instead of a window")
    p.add_argument("--freecad-python", default=None)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    view(args.stl, args.step, args.deflection, args.clamp, args.freecad_python, args.screenshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
