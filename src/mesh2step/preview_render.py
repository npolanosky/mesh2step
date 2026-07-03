"""Off-screen preview renderer, run as a subprocess of the GUI.

Renders JSON scene specs to image files with pyvista. The embedded viewer uses
this instead of rendering in-process because creating a VTK off-screen render
context from a background thread deadlocks on macOS (Cocoa wants the main
thread) — a subprocess renders on *its* main thread, and a hung render is
killable via a timeout instead of freezing the GUI.

Two modes:

* one-shot: ``preview_render <spec.json>`` — render one spec and exit.
* server: ``preview_render --serve`` — read one JSON request per stdin line,
  answer one JSON line on stdout, repeat until EOF. The GUI keeps ONE server
  alive for its whole life, so the heavy pyvista/VTK import (plus a frozen
  app's bootstrap — several seconds) is paid once. Loaded meshes are cached
  (LRU, keyed by path+mtime+size) so drag re-renders don't re-parse the file.

Protocol (one JSON object per line):

* render request — the spec below, plus optional ``"id"`` (echoed back).
* ``{"ping": true, "id": n}`` — liveness probe; answered without rendering.
* response — ``{"ok": true, "png": path, "id": n}`` or
  ``{"ok": false, "error": msg, "id": n}``.

Spec::

    {
      "id": 7,
      "width": 800, "height": 600,
      "background": "#f0f2f5",
      "mode": "shaded" | "edges" | "wire",
      "out": "preview.png",              # .jpg/.jpeg -> JPEG (fast drag frames)
      "camera": {"azimuth": 0, "elevation": 0, "zoom": 1, "pan": [0, 0]},
      "meshes": [
        {"path": "scene.vtp",          # any pyvista-readable mesh file
         "color": "#8b95a1",           # ignored when "scalars" is set
         "opacity": 1.0,
         "points": false,              # render as sphere markers
         "scalars": "deviation",       # active point-data array to colour by
         "clim": [0.0, 1.0]},
        ...
      ]
    }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Light neutral background — model colours and the heatmap read much better on
# it than on the app's dark panel, and scalar-bar text is dark to match.
DEFAULT_BG = "#f0f2f5"
SCALAR_TEXT = "#1f2937"

# Server-side mesh cache: at most this many loaded polydata objects, LRU.
# Bounds the server's memory while making drag re-renders parse-free.
_CACHE_MAX = 6


def _load_mesh(path: str, cache: dict | None):
    """pyvista-read ``path`` through the LRU cache (keyed path+mtime+size)."""
    import pyvista as pv

    if cache is None:
        return pv.read(path)
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return pv.read(path)
    if key in cache:
        cache[key] = cache.pop(key)      # bump to most-recent (dict is ordered)
        return cache[key]
    poly = pv.read(path)
    for stale in [k for k in cache if k[0] == path]:
        cache.pop(stale)                 # same file changed on disk
    cache[key] = poly
    while len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))     # evict least-recent
    return poly


def _apply_camera(plotter, cam: dict) -> None:
    """Apply a relative camera pose on top of the auto-fitted iso view.

    ``cam`` holds deltas accumulated by the GUI's mouse interaction:
    ``azimuth``/``elevation`` in degrees, ``zoom`` as a multiplicative dolly
    factor (>1 = closer), ``pan`` as a view-plane offset in viewport fractions.
    Applying them fresh from the canonical fitted pose each render keeps the
    mapping stateless and drift-free.
    """
    from math import radians, tan

    import numpy as np

    camera = plotter.camera
    az = float(cam.get("azimuth", 0.0))
    el = float(cam.get("elevation", 0.0))
    zoom = float(cam.get("zoom", 1.0)) or 1.0
    pan = cam.get("pan") or [0.0, 0.0]
    if az:
        camera.Azimuth(az)
    if el:
        camera.Elevation(el)
    camera.OrthogonalizeViewUp()
    if pan[0] or pan[1]:
        pos = np.array(camera.GetPosition())
        foc = np.array(camera.GetFocalPoint())
        up = np.array(camera.GetViewUp(), dtype=float)
        up /= max(np.linalg.norm(up), 1e-12)
        fwd = foc - pos
        dist = max(np.linalg.norm(fwd), 1e-12)
        fwd /= dist
        right = np.cross(fwd, up)
        right /= max(np.linalg.norm(right), 1e-12)
        # World-space size of the viewport at the focal plane.
        h = 2.0 * dist * tan(radians(camera.GetViewAngle()) / 2.0)
        w = h * (plotter.window_size[0] / max(plotter.window_size[1], 1))
        off = right * (-pan[0] * w) + up * (pan[1] * h)
        camera.SetPosition(*(pos + off))
        camera.SetFocalPoint(*(foc + off))
    if zoom != 1.0:
        camera.Dolly(zoom)
    plotter.renderer.ResetCameraClippingRange()


def render(spec: dict, cache: dict | None = None) -> None:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True,
                         window_size=[int(spec.get("width", 800)),
                                      int(spec.get("height", 600))])
    plotter.set_background(spec.get("background", DEFAULT_BG))
    mode = spec.get("mode", "shaded")
    for m in spec.get("meshes", []):
        poly = _load_mesh(m["path"], cache)
        if m.get("points"):
            plotter.add_mesh(poly, color=m.get("color", "#ef4444"), style="points",
                             render_points_as_spheres=True, point_size=9)
            continue
        kw: dict = {"opacity": m.get("opacity", 1.0)}
        scalars = m.get("scalars")
        if scalars is not None:
            poly.set_active_scalars(scalars)
            kw.update(scalars=scalars, cmap="jet",
                      clim=list(m.get("clim", (0.0, 1.0))),
                      scalar_bar_args={"title": "dev (mm)", "color": SCALAR_TEXT})
        elif m.get("color"):
            kw["color"] = m["color"]
        if mode == "wire":
            kw["style"] = "wireframe"
        elif mode == "edges":
            kw["show_edges"] = True
        plotter.add_mesh(poly, **kw)
    plotter.camera_position = "iso"
    cam = spec.get("camera")
    if cam:
        _apply_camera(plotter, cam)
    out = spec["out"]
    if out.lower().endswith((".jpg", ".jpeg")):
        # JPEG drag frames: smaller and faster to encode/decode than PNG.
        from PIL import Image

        img = plotter.screenshot(return_img=True)
        Image.fromarray(img).save(out, quality=80)
    else:
        plotter.screenshot(out)
    plotter.close()


def serve() -> int:
    """Line-protocol render server: JSON request in, JSON result out, until EOF.

    Imports pyvista up front so the first request already runs warm. Any
    exception is reported as a JSON error line — the server never dies from a
    bad scene, only from EOF (parent exited) or a hard native crash (the GUI
    restarts it in that case).
    """
    import pyvista  # noqa: F401 - warm the heavy import once

    cache: dict = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            spec = json.loads(line)
            req_id = spec.get("id")
            if spec.get("ping"):
                resp: dict = {"ok": True, "pong": True}
            else:
                render(spec, cache)
                resp = {"ok": True, "png": spec["out"]}
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            resp = {"ok": False, "error": str(exc)}
        if req_id is not None:
            resp["id"] = req_id
        print(json.dumps(resp), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--serve"]:
        return serve()
    if len(args) != 1:
        print("usage: preview_render <spec.json> | --serve", file=sys.stderr)
        return 2
    render(json.loads(Path(args[0]).read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
