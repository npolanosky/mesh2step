"""In-window 3D preview panel (STL / STEP / deviation heatmap).

Two rendering strategies, chosen by platform:

* **Windows** — embeds a real, GPU-accelerated VTK render window directly in the
  tkinter GUI by reparenting it into a Tk frame (``vtkRenderWindow.SetParentInfo``
  with the frame's native window id). True trackball interaction, docked.

* **macOS / Linux** — reparenting a VTK/NSView render window into Tk is not
  workable, so each view (STL / STEP / deviation heatmap) is rendered
  **off-screen to an image** and shown in a Tk label. The render runs in a
  short-lived ``preview_render`` subprocess: creating a VTK off-screen context
  from a background thread deadlocks on macOS (Cocoa wants the main thread),
  and a subprocess also gives us a hard timeout — a stuck render surfaces as an
  error message in the panel instead of hanging the app. Live camera
  interaction is one click away via **Pop out ↗** (own interactive VTK window).

States it shows through a conversion:
  * the input STL as soon as a file is selected (defect regions marked red)
  * the output STEP once the conversion finishes
  * the STEP coloured by its deviation from the STL (heatmap)
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# Light neutral render background: model colours and the deviation heatmap
# read far better on it than the old dark panel, and scalar-bar/placeholder
# text is dark to match. BG (Windows live renderer) and BG_HEX (static path +
# render subprocess) must stay in sync.
BG = (0.941, 0.949, 0.961)
BG_HEX = "#f0f2f5"
FG = "#1f2937"
MUTED = "#6b7280"

# Scene colours tuned for the light background.
STL_COLOR = "#8b95a1"       # input mesh: medium grey
STEP_COLOR = "#3f97cf"      # output solid: medium blue
GHOST_COLOR = "#64748b"     # translucent STL context under the heatmap

# Mouse interaction tuning for the static (image-based) preview.
_ORBIT_DEG_PER_PX = 0.4     # drag sensitivity: degrees of orbit per pixel
_ZOOM_PER_STEP = 1.05       # zoom factor per wheel step
_DRAG_DOWNSCALE = 2         # render at 1/2 resolution while dragging

# Whether this platform embeds a live VTK window (Windows) or renders static
# preview images (everywhere else — reparenting VTK into Tk isn't workable).
_LIVE_EMBED = sys.platform == "win32"

# Hard ceiling for one off-screen preview render, seconds. The first request
# also absorbs the render server's start-up (frozen bootstrap + pyvista import).
_RENDER_TIMEOUT = 180

# A panel resize smaller than this (px, either axis) does not trigger a
# re-render — the current image just letterboxes. Prevents render feedback
# loops and render spam while the user drags the sash.
_RESIZE_THRESHOLD = 24

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_log = logging.getLogger("mesh2step")


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _package_src() -> str:
    if getattr(sys, "frozen", False):
        return str(Path(getattr(sys, "_MEIPASS", ".")) / "mesh2step_src")
    return str(Path(__file__).resolve().parent.parent)


class EmbeddedViewer(ttk.Frame):
    def __init__(self, parent, freecad_python_getter=None, on_popout=None):
        super().__init__(parent, style="Bg.TFrame")
        self._get_fc = freecad_python_getter or (lambda: None)

        self._scenes: dict[str, list] = {}        # name -> list of vtk actors (win)
        self._specs: dict[str, list] = {}          # name -> file-based scene specs (mac)
        self._images: dict[str, object] = {}       # name -> PhotoImage (mac; kept alive)
        self._active: str | None = None
        self._mode = "shaded"                      # shaded | edges | wire
        self._q: queue.Queue = queue.Queue()
        self._vtk = None                           # (renwin, renderer, interactor)
        self._vtk_failed = False
        self._scene_hint: dict[str, str] = {}
        self._rendering = False                    # a static render is in flight
        self._pending_render: str | None = None    # view requested while busy
        self._resize_job = None
        self._tmpdir: str | None = None            # scene .vtp files live here
        self._displayed: str | None = None         # which view's image is on screen
        self._last_render_size: tuple | None = None  # (w, h) of last finished render
        self._render_proc = None                   # persistent render server
        self._server_lock = threading.Lock()       # one request in flight at a time
        self._pending_force = False                # pending render must run even if cached
        # Camera pose (relative to the auto-fitted iso view), shared across the
        # STL/STEP/heatmap views so switching keeps the pose. Driven by mouse
        # interaction on the preview image; zeros = default fitted view.
        self._camera = {"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0,
                        "pan": [0.0, 0.0]}
        self._dragging = False
        self._drag_last: tuple | None = None
        self._interactive_render = False           # next render may be low-res

        # --- toolbar ---
        bar = ttk.Frame(self, style="Bg.TFrame")
        bar.pack(fill="x", padx=6, pady=6)
        self._btns: dict[str, ttk.Button] = {}
        for name, label in (("stl", "STL"), ("step", "STEP"), ("heatmap", "Heatmap")):
            b = ttk.Button(bar, text=label, width=9, state="disabled",
                           command=lambda n=name: self.set_view(n))
            b.pack(side="left", padx=(0, 4))
            self._btns[name] = b
        ttk.Button(bar, text="Fit view", command=self.reset_view).pack(side="left", padx=(8, 0))
        if on_popout is not None:
            self._popout_btn = ttk.Button(bar, text="Pop out ↗", command=on_popout)
            self._popout_btn.pack(side="right")
        self._mode_var = tk.StringVar(value="Shaded")
        mode_cb = ttk.Combobox(bar, textvariable=self._mode_var, width=13, state="readonly",
                               values=["Shaded", "Shaded + edges", "Wireframe"])
        mode_cb.pack(side="right", padx=(0, 8))
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode)
        ttk.Label(bar, text="Display", style="Muted.TLabel").pack(side="right", padx=(0, 4))

        # --- render surface ---
        # Windows: a bare frame VTK reparents into. Elsewhere: a label that shows
        # the off-screen-rendered preview image.
        self._surface = tk.Frame(self, bg=BG_HEX, width=640, height=480)
        self._surface.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._surface.bind("<Configure>", self._on_surface_configure)
        self._image_label = None
        if not _LIVE_EMBED:
            # CRITICAL: keep pack propagation OFF. With it on, showing a rendered
            # image resizes the label -> the frame -> fires <Configure> -> queues
            # another render whose placeholder shrinks the frame again — an
            # endless render/flash loop. The frame must own its size; images and
            # placeholder text just fit inside it.
            self._surface.pack_propagate(False)
            self._image_label = tk.Label(self._surface, bg=BG_HEX, bd=0,
                                         text="Select an STL to preview", fg=MUTED,
                                         font=("Consolas", 10))
            self._image_label.pack(fill="both", expand=True)
            # Image-based camera control: left-drag orbits, right/middle- or
            # shift-drag pans, wheel zooms. Each gesture updates the shared
            # camera pose and asks the render server for a fresh frame
            # (in-flight-one + latest-wins keeps it smooth at ~0.05-0.2s/frame).
            lbl = self._image_label
            lbl.bind("<ButtonPress-1>", self._drag_start)
            lbl.bind("<B1-Motion>", lambda e: self._drag_move(e, "orbit"))
            lbl.bind("<Shift-B1-Motion>", lambda e: self._drag_move(e, "pan"))
            lbl.bind("<ButtonRelease-1>", self._drag_end)
            for btn in ("2", "3"):   # right button is 2 on aqua, 3 elsewhere
                lbl.bind(f"<ButtonPress-{btn}>", self._drag_start)
                lbl.bind(f"<B{btn}-Motion>", lambda e: self._drag_move(e, "pan"))
                lbl.bind(f"<ButtonRelease-{btn}>", self._drag_end)
            lbl.bind("<MouseWheel>", self._on_zoom)
            # Pre-warm the render server at startup so the first preview only
            # costs the render, not the multi-second pyvista/frozen bootstrap.
            self.after(300, self._prewarm_server)

        self._hint = tk.Label(self, text="Select an STL to preview", bg=BG_HEX, fg=MUTED,
                              anchor="w", font=("Consolas", 9))
        self._hint.pack(fill="x", padx=8, pady=(0, 6))

        self.after(60, self._poll)

    # ---- worker-thread handoff -------------------------------------------
    def _poll(self):
        try:
            while True:
                fn = self._q.get_nowait()
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
        except queue.Empty:
            pass
        self.after(60, self._poll)

    # ---- VTK setup (Windows live-embed only) ------------------------------
    def _ensure_vtk(self):
        """Create the embedded GPU render window on first use. Returns it or None."""
        if self._vtk is not None or self._vtk_failed:
            return self._vtk
        if not _LIVE_EMBED:
            self._vtk_failed = True
            return None
        w = self._surface.winfo_width()
        h = self._surface.winfo_height()
        if w <= 1 or h <= 1:
            return None  # not realised yet; caller retries after <Configure>
        try:
            import vtkmodules.all as vtk

            renwin = vtk.vtkRenderWindow()
            renwin.SetParentInfo(str(self._surface.winfo_id()))
            renwin.SetSize(w, h)
            ren = vtk.vtkRenderer()
            ren.SetBackground(*BG)
            renwin.AddRenderer(ren)
            vtk.vtkLightKit().AddLightsToRenderer(ren)   # pleasant 3-point-ish lighting
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renwin)
            iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            iren.Initialize()
            self._vtk = (renwin, ren, iren)
            return self._vtk
        except Exception as exc:  # noqa: BLE001
            self._vtk_failed = True
            self._hint.config(text=f"3D preview unavailable ({exc}); use Pop out ↗.")
            return None

    def _on_surface_configure(self, _e=None):
        if _LIVE_EMBED:
            v = self._ensure_vtk()
            if v is not None:
                renwin, _ren, _iren = v
                renwin.SetSize(max(self._surface.winfo_width(), 1),
                               max(self._surface.winfo_height(), 1))
                renwin.Render()
            return
        # Static path: re-render the active scene only when the panel size REALLY
        # changed since the last completed render (debounced; small jitter is
        # absorbed by the threshold). Never re-render just because an image or
        # placeholder was swapped in — that's how feedback loops start.
        if self._active and self._active in self._specs and self._size_changed():
            if self._resize_job is not None:
                self.after_cancel(self._resize_job)
            self._resize_job = self.after(250, self._rerender_active)

    def _size_changed(self) -> bool:
        if self._last_render_size is None:
            return False  # nothing rendered yet; the first render sizes itself
        lw, lh = self._last_render_size
        return (abs(self._surface.winfo_width() - lw) > _RESIZE_THRESHOLD
                or abs(self._surface.winfo_height() - lh) > _RESIZE_THRESHOLD)

    def _rerender_active(self):
        self._resize_job = None
        if self._active and self._size_changed():
            # All cached images are the old size now; drop them so view switches
            # re-render, but KEEP showing the current image until its replacement
            # arrives (never flash the placeholder over a live image).
            self._images.clear()
            self._render_static(self._active)

    # ---- public API -------------------------------------------------------
    def clear(self):
        self._scenes.clear()
        self._specs.clear()
        self._images.clear()
        self._active = None
        for b in self._btns.values():
            b.config(state="disabled")
        if self._vtk:
            self._vtk[1].RemoveAllViewProps()
            self._vtk[0].Render()
        if self._image_label is not None:
            self._image_label.config(image="", text="Select an STL to preview")
        self._displayed = None
        self._last_render_size = None
        self._camera = {"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0,
                        "pan": [0.0, 0.0]}
        self._hint.config(text="Select an STL to preview")

    def show_stl(self, stl_path: str, problem_points=None):
        self._hint.config(text="Loading mesh…")
        if not _LIVE_EMBED:
            # Start the render server now so its cold start (frozen bootstrap +
            # pyvista import) overlaps with reading/writing the scene files.
            self._ensure_render_server()

        def work():
            try:
                import pyvista as pv
                mesh = pv.read(stl_path)
                specs = [dict(poly=mesh, color=STL_COLOR)]
                npts = 0
                if problem_points:
                    import numpy as np
                    pts = np.asarray(problem_points, dtype=float)
                    if pts.size:
                        npts = len(pts)
                        specs.append(dict(poly=pv.PolyData(pts), color="#ef4444", points=True))
                hint = (f"input STL — {mesh.n_points:,} pts, {mesh.n_cells:,} tris"
                        + (f"   ⚠ {npts} defect markers" if npts else ""))
                specs = self._materialise(specs, "stl")
                self._q.put(lambda: self._install("stl", specs, hint, reset=True))
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: self._hint.config(text=f"Could not load mesh: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def show_result(self, stl_path: str, step_path: str):
        fc = self._get_fc()
        self._hint.config(text="Building STEP preview…")
        if not _LIVE_EMBED:
            self._ensure_render_server()  # warm it during tessellation

        def work():
            try:
                from .viewer import build_scene
                stl_poly, step_poly, stats = build_scene(stl_path, step_path, 0.1, fc)
                hi = max(stats["p95"], stats["max"] * 0.5, 1e-6)
                step_specs = [dict(poly=step_poly, color=STEP_COLOR)]
                heat_specs = [dict(poly=stl_poly, color=GHOST_COLOR, opacity=0.15),
                              dict(poly=step_poly, scalars="deviation", clim=(0.0, hi))]
                step_hint = f"output STEP — {step_poly.n_cells:,} tris"
                heat_hint = (f"deviation (mm)  max={stats['max']:.3f}  rms={stats['rms']:.3f}  "
                             f"p95={stats['p95']:.3f}  mean={stats['mean']:.3f}")
                step_specs = self._materialise(step_specs, "step")
                heat_specs = self._materialise(heat_specs, "heatmap")

                def apply():
                    self._install("step", step_specs, step_hint, reset=False, activate=False)
                    self._install("heatmap", heat_specs, heat_hint, reset=False, activate=True)
                self._q.put(apply)
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: self._hint.config(text=f"STEP preview failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    # ---- scene assembly ---------------------------------------------------
    def _materialise(self, specs, name):
        """On the static path, write each poly to a .vtp so the render
        subprocess can load it; pass through untouched on the live path.
        Runs on the producing worker thread (file IO off the UI thread)."""
        if _LIVE_EMBED:
            return specs
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="mesh2step_preview_")
        out = []
        for i, spec in enumerate(specs):
            fspec = {k: v for k, v in spec.items() if k != "poly"}
            path = os.path.join(self._tmpdir, f"{name}_{i}.vtp")
            spec["poly"].save(path)
            fspec["path"] = path
            out.append(fspec)
        return out

    def _install(self, name, specs, hint, reset, activate=True):
        self._scene_hint[name] = hint
        self._btns[name].config(state="normal")
        if not _LIVE_EMBED:
            self._specs[name] = specs      # file-based specs (see _materialise)
            self._images.pop(name, None)   # scene changed; drop any stale image
            if activate:
                self.set_view(name)
            return
        self._install_vtk(name, specs, hint, reset, activate)

    # ---- static image path (macOS / Linux) --------------------------------
    def _render_static(self, name: str):
        """Render scene ``name`` to an image via the preview_render subprocess."""
        specs = self._specs.get(name)
        if not specs:
            return
        if self._rendering:
            self._pending_render = name
            return
        self._rendering = True
        self._active = name
        for n, b in self._btns.items():
            b.state(["pressed"] if n == name else ["!pressed"])
        self._hint.config(text=(self._scene_hint.get(name, "") + "   (rendering…)").strip())
        # Placeholder ONLY when the view on screen isn't the one being rendered
        # (first render / view switch). A same-view re-render keeps showing the
        # current image until the replacement arrives — no flashing.
        if self._image_label is not None and self._displayed != name:
            self._image_label.config(image="", text="Rendering preview…")
            self._displayed = None

        w = max(self._surface.winfo_width(), 320)
        h = max(self._surface.winfo_height(), 240)
        # While a drag is in progress render at reduced resolution (the image is
        # upscaled to the panel, then replaced by a full-res frame on release).
        scale = _DRAG_DOWNSCALE if self._interactive_render else 1
        spec = {
            "width": int(w // scale), "height": int(h // scale), "background": BG_HEX,
            "mode": self._mode, "meshes": specs,
            # Snapshot the camera NOW — the dicts keep mutating during a drag.
            "camera": {"azimuth": self._camera["azimuth"],
                       "elevation": self._camera["elevation"],
                       "zoom": self._camera["zoom"],
                       "pan": list(self._camera["pan"])},
            "out": os.path.join(self._tmpdir or tempfile.gettempdir(),
                                f"render_{name}.png"),
        }

        def work():
            t0 = time.monotonic()
            try:
                png = self._render_via_server(spec)
                _log.info("preview render %s (%dx%d) took %.2fs",
                          name, spec["width"], spec["height"], time.monotonic() - t0)
                self._q.put(lambda p=png: self._show_image(name, p, (int(w), int(h))))
            except Exception as exc:  # noqa: BLE001
                _log.warning("preview render %s failed after %.2fs: %s",
                             name, time.monotonic() - t0, exc)
                self._q.put(lambda e=exc: self._on_render_error(e))

        threading.Thread(target=work, daemon=True).start()

    # ---- persistent render server ------------------------------------------
    def _server_cmd(self):
        if getattr(sys, "frozen", False):
            return [sys.executable, "--render-preview", "--serve"], None
        env = dict(os.environ)
        env["PYTHONPATH"] = _package_src() + os.pathsep + env.get("PYTHONPATH", "")
        return [sys.executable, "-m", "mesh2step.preview_render", "--serve"], env

    def _ensure_render_server(self):
        """The persistent renderer subprocess, (re)spawned if absent or dead.

        One long-lived server means the pyvista/VTK import (plus a frozen app's
        multi-second bootstrap) is paid once per app run, not once per render —
        the difference between 4–10 s and well under a second per preview.
        """
        p = self._render_proc
        if p is not None and p.poll() is None:
            return p
        cmd, env = self._server_cmd()
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, bufsize=1, creationflags=_NO_WINDOW)
        self._render_proc = p
        atexit.register(self._kill_render_server)  # never outlive the app
        return p

    def _kill_render_server(self):
        p, self._render_proc = self._render_proc, None
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass

    def _render_via_server(self, spec: dict) -> str:
        """Send one render request to the server; return the PNG path.

        Serialised by a lock (the server handles one request at a time). A dead
        server is restarted once; a request that exceeds the timeout kills the
        server (it will be respawned on the next request) and raises.
        """
        import select

        with self._server_lock:
            last_err = None
            for _attempt in (1, 2):
                proc = self._ensure_render_server()
                try:
                    proc.stdin.write(json.dumps(spec) + "\n")
                    proc.stdin.flush()
                except Exception as exc:  # noqa: BLE001 - server died; retry once
                    last_err = exc
                    self._kill_render_server()
                    continue
                deadline = time.monotonic() + _RENDER_TIMEOUT
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._kill_render_server()
                        raise RuntimeError(f"render timed out after {_RENDER_TIMEOUT}s")
                    ready, _, _ = select.select([proc.stdout], [], [],
                                                min(remaining, 1.0))
                    if not ready:
                        if proc.poll() is not None:
                            last_err = RuntimeError("render server died")
                            self._kill_render_server()
                            break
                        continue
                    line = proc.stdout.readline()
                    if not line:
                        last_err = RuntimeError("render server closed its pipe")
                        self._kill_render_server()
                        break
                    try:
                        resp = json.loads(line)
                    except ValueError:
                        continue  # stray non-protocol output; keep reading
                    if not isinstance(resp, dict) or "ok" not in resp:
                        continue
                    if not resp["ok"]:
                        raise RuntimeError(resp.get("error", "render failed"))
                    return resp["png"]
            raise RuntimeError(f"render server unavailable ({last_err})")

    def _show_image(self, name: str, png_path: str, size: tuple):
        self._rendering = False
        try:
            from PIL import Image, ImageTk

            img = Image.open(png_path)
            if img.size != tuple(size):
                # Low-res drag frame: stretch to the panel so the view doesn't
                # jump between drag frames and the full-res release frame.
                img = img.resize((int(size[0]), int(size[1])))
            photo = ImageTk.PhotoImage(img)
        except Exception as exc:  # noqa: BLE001
            self._on_render_error(exc)
            return
        self._images[name] = photo  # keep a ref so Tk doesn't GC it
        self._last_render_size = size
        if self._active == name and self._image_label is not None:
            self._image_label.config(image=photo, text="")
            self._displayed = name
            self._hint.config(text=self._scene_hint.get(name, ""))
        self._kick_pending()

    def _on_render_error(self, exc):
        self._rendering = False
        if self._image_label is not None and not self._image_label.cget("image"):
            self._image_label.config(text="Preview unavailable — use Pop out ↗")
        self._hint.config(text=f"Preview render failed: {exc} — use Pop out ↗.")
        self._kick_pending()

    def _kick_pending(self):
        nxt, self._pending_render = self._pending_render, None
        force, self._pending_force = self._pending_force, False
        if nxt and (force or nxt != self._active or nxt not in self._images):
            self._render_static(nxt)

    # ---- mouse camera control (static path) --------------------------------
    def _drag_start(self, e):
        if self._active in self._specs:
            self._dragging = True
            self._drag_last = (e.x, e.y)

    def _drag_move(self, e, kind: str):
        if not self._dragging or self._drag_last is None:
            return
        dx = e.x - self._drag_last[0]
        dy = e.y - self._drag_last[1]
        self._drag_last = (e.x, e.y)
        if not dx and not dy:
            return
        cam = self._camera
        if kind == "orbit":
            cam["azimuth"] -= dx * _ORBIT_DEG_PER_PX
            cam["elevation"] = max(-89.0, min(89.0,
                                              cam["elevation"] + dy * _ORBIT_DEG_PER_PX))
        else:  # pan (view-plane, in viewport fractions)
            w = max(self._surface.winfo_width(), 1)
            h = max(self._surface.winfo_height(), 1)
            cam["pan"][0] += dx / w
            cam["pan"][1] += dy / h
        self._camera_render(interactive=True)

    def _drag_end(self, _e):
        if not self._dragging:
            return
        self._dragging = False
        self._drag_last = None
        self._camera_render(interactive=False)  # full-res settle frame

    def _on_zoom(self, e):
        if self._active not in self._specs:
            return
        steps = e.delta if sys.platform == "darwin" else int(e.delta / 120)
        if not steps:
            return
        cam = self._camera
        cam["zoom"] = max(0.05, min(50.0, cam["zoom"] * (_ZOOM_PER_STEP ** steps)))
        self._camera_render(interactive=False)

    def _camera_render(self, interactive: bool):
        """Re-render the active view with the current camera pose.

        In-flight-one + latest-wins: if a render is running, just mark the view
        pending (forced) — when the frame lands, the newest camera state is
        rendered. Intermediate poses are dropped, so a fast drag never queues up.
        """
        name = self._active
        if not name or name not in self._specs:
            return
        self._images.clear()   # every cached pose is stale now
        self._interactive_render = interactive
        if self._rendering:
            self._pending_render = name
            self._pending_force = True
            return
        self._render_static(name)

    def _prewarm_server(self):
        if not _LIVE_EMBED:
            try:
                self._ensure_render_server()
            except Exception:  # noqa: BLE001 - warm-up is opportunistic
                pass

    # ---- VTK live path (Windows) ------------------------------------------
    def _make_actor(self, spec):
        import vtkmodules.all as vtk

        poly = spec["poly"]
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        scalars = spec.get("scalars")
        if scalars is not None:
            poly.GetPointData().SetActiveScalars(scalars)
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)   # blue (low) -> red (high)
            lut.SetNumberOfTableValues(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(*spec.get("clim", (0.0, 1.0)))
            mapper.SetScalarModeToUsePointData()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        pr = actor.GetProperty()
        if spec.get("color"):
            pr.SetColor(*_hex_rgb(spec["color"]))
        pr.SetOpacity(spec.get("opacity", 1.0))
        if spec.get("points"):
            pr.SetRepresentationToPoints()
            pr.SetPointSize(9)
            try:
                pr.RenderPointsAsSpheresOn()
            except Exception:  # noqa: BLE001
                pass
            actor._is_points = True  # tag so display-mode skips it
        return actor

    def _install_vtk(self, name, specs, hint, reset, activate=True):
        v = self._ensure_vtk()
        if v is None:
            # No GPU surface yet; stash specs to build on activation/configure.
            self._scenes[name] = specs
            if activate:
                self._active = name
                self._hint.config(text=hint + "   (preview initialising…)")
                self.after(120, lambda: self.set_view(name))
            return
        actors = [self._make_actor(s) for s in specs]
        self._scenes[name] = actors
        if activate:
            self._show_actors(name, reset=reset)

    def _show_actors(self, name, reset):
        v = self._ensure_vtk()
        if v is None:
            return
        renwin, ren, _iren = v
        actors = self._scenes.get(name)
        if actors and isinstance(actors[0], dict):   # stashed specs -> build now
            actors = [self._make_actor(s) for s in actors]
            self._scenes[name] = actors
        if not actors:
            return
        ren.RemoveAllViewProps()
        for a in actors:
            self._apply_mode_actor(a)
            ren.AddActor(a)
        if reset:
            ren.ResetCamera()
        self._active = name
        for n, b in self._btns.items():
            b.state(["pressed"] if n == name else ["!pressed"])
        self._hint.config(text=self._scene_hint.get(name, ""))
        renwin.Render()

    # ---- view controls ----------------------------------------------------
    def set_view(self, name: str):
        if _LIVE_EMBED:
            if name in self._scenes:
                self._show_actors(name, reset=False)
            return
        if name not in self._specs:
            return
        self._active = name
        for n, b in self._btns.items():
            b.state(["pressed"] if n == name else ["!pressed"])
        photo = self._images.get(name)
        if photo is not None and self._image_label is not None:
            self._image_label.config(image=photo, text="")
            self._displayed = name
            self._hint.config(text=self._scene_hint.get(name, ""))
        else:
            self._render_static(name)

    def reset_view(self):
        if _LIVE_EMBED:
            v = self._ensure_vtk()
            if v is not None:
                v[1].ResetCamera()
                v[0].Render()
            return
        # Fit view: back to the auto-fitted iso pose.
        self._camera = {"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0,
                        "pan": [0.0, 0.0]}
        self._interactive_render = False
        if self._active:
            self._images.clear()  # cached images hold the old pose
            self._render_static(self._active)

    def _on_mode(self, _e=None):
        self._mode = {"Shaded": "shaded", "Shaded + edges": "edges",
                      "Wireframe": "wire"}.get(self._mode_var.get(), "shaded")
        if _LIVE_EMBED:
            v = self._ensure_vtk()
            if v is None or self._active is None:
                return
            for a in self._scenes.get(self._active, []):
                if not isinstance(a, dict):
                    self._apply_mode_actor(a)
            v[0].Render()
            return
        # Static: display mode changed -> every cached image is stale.
        self._images.clear()
        if self._active:
            self._render_static(self._active)

    def _apply_mode_actor(self, actor):
        if getattr(actor, "_is_points", False):
            return
        pr = actor.GetProperty()
        if self._mode == "wire":
            pr.SetRepresentationToWireframe()
            pr.SetLineWidth(1)
        else:
            pr.SetRepresentationToSurface()
            pr.SetEdgeVisibility(1 if self._mode == "edges" else 0)
            pr.SetEdgeColor(0.20, 0.24, 0.30)
