"""In-window 3D preview panel (STL / STEP / deviation heatmap).

Embeds a 3D view directly in the tkinter GUI. Rendering is done *off-screen*
with pyvista (the same proven path as the standalone viewer) and the resulting
frame is shown in a tkinter label — this avoids the fragility of hosting a live
VTK/Tk render window inside a frozen app while still giving orbit/zoom and
instant view toggling.

States it shows through a conversion:
  * the input STL as soon as a file is selected (with defect regions marked red)
  * the output STEP once the conversion finishes
  * the STEP coloured by its deviation from the STL (heatmap)

All heavy data prep (STEP tessellation via FreeCAD, deviation) runs on a worker
thread; only the actual rendering touches VTK, always on the Tk main thread.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

BG = "#0f172a"
FG = "#e2e8f0"
MUTED = "#94a3b8"


class EmbeddedViewer(ttk.Frame):
    def __init__(self, parent, freecad_python_getter=None, on_popout=None):
        super().__init__(parent, style="Bg.TFrame")
        self._get_fc = freecad_python_getter or (lambda: None)
        self._on_popout = on_popout

        self._scenes: dict[str, dict] = {}      # name -> {"meshes": [(poly, kwargs)], "stats": {}}
        self._active: str | None = None
        self._plotter = None
        self._plotter_size: tuple[int, int] | None = None
        self._built_view: str | None = None
        self._azim = 0.0
        self._elev = 0.0
        self._zoom = 1.0
        self._drag = None
        self._resize_job = None
        self._photo = None                       # keep a ref so Tk doesn't GC it
        self._available = None                   # lazily probed pyvista availability
        self._q: queue.Queue = queue.Queue()     # worker-thread -> main-thread callables

        # --- toolbar ---
        bar = ttk.Frame(self, style="Bg.TFrame")
        bar.pack(fill="x", padx=6, pady=6)
        self._btns: dict[str, ttk.Button] = {}
        for name, label in (("stl", "STL"), ("step", "STEP"), ("heatmap", "Heatmap")):
            b = ttk.Button(bar, text=label, width=9, state="disabled",
                           command=lambda n=name: self.set_view(n))
            b.pack(side="left", padx=(0, 4))
            self._btns[name] = b
        ttk.Button(bar, text="Reset view", command=self.reset_view).pack(side="left", padx=(8, 0))
        if on_popout is not None:
            ttk.Button(bar, text="Pop out ↗", command=on_popout).pack(side="right")

        # --- render surface ---
        self.canvas = tk.Label(self, bg=BG, fg=MUTED, anchor="center",
                               text="3D preview\n\nselect an STL to begin",
                               font=("Segoe UI", 11), justify="center")
        self.canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # --- stats / hint line ---
        self.info = tk.Label(self, text="", bg=BG, fg=MUTED, anchor="w",
                             font=("Consolas", 9))
        self.info.pack(fill="x", padx=8, pady=(0, 6))

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Configure>", self._on_resize)

        self.after(60, self._poll)

    def _poll(self):
        """Run callables handed over from worker threads, on the main thread."""
        try:
            while True:
                fn = self._q.get_nowait()
                try:
                    fn()
                except Exception:  # noqa: BLE001 - one bad update mustn't stop the poller
                    pass
        except queue.Empty:
            pass
        self.after(60, self._poll)

    # ---- availability -----------------------------------------------------
    def _pv(self):
        """Import pyvista lazily; return the module or None if unavailable."""
        if self._available is None:
            try:
                import pyvista as pv  # noqa: F401
                self._available = True
            except Exception:  # noqa: BLE001
                self._available = False
        if not self._available:
            return None
        import pyvista as pv
        return pv

    # ---- public API -------------------------------------------------------
    def clear(self):
        self._scenes.clear()
        self._active = None
        self._built_view = None
        for b in self._btns.values():
            b.config(state="disabled")
        self._photo = None
        self.canvas.config(image="", text="3D preview\n\nselect an STL to begin")
        self.info.config(text="")

    def show_message(self, text: str):
        self.canvas.config(image="", text=text)
        self._photo = None

    def show_stl(self, stl_path: str, problem_points=None):
        """Show the input mesh immediately (data load is cheap, done inline)."""
        pv = self._pv()
        if pv is None:
            self.show_message("3D preview unavailable\n(pyvista not installed)")
            return
        self.show_message("Loading mesh…")
        self.update_idletasks()

        def work():
            try:
                mesh = pv.read(stl_path)
                meshes = [(mesh, dict(color="#cbd5e1", opacity=1.0,
                                      smooth_shading=True, show_edges=False))]
                npts = 0
                if problem_points:
                    import numpy as np
                    pts = np.asarray(problem_points, dtype=float)
                    if pts.size:
                        npts = len(pts)
                        meshes.append((pv.PolyData(pts),
                                       dict(color="#ef4444", render_points_as_spheres=True,
                                            point_size=10.0)))
                self._scenes["stl"] = {
                    "meshes": meshes,
                    "hint": (f"input STL — {mesh.n_points:,} pts, {mesh.n_cells:,} tris"
                             + (f"   ⚠ {npts} defect markers" if npts else "")),
                }
                self._q.put(lambda: self._on_scene_ready("stl"))
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: self.show_message(f"Could not load mesh:\n{e}"))

        threading.Thread(target=work, daemon=True).start()

    def show_result(self, stl_path: str, step_path: str):
        """Tessellate the STEP + compute deviation on a thread, then show it."""
        pv = self._pv()
        if pv is None:
            return
        fc = self._get_fc()
        self.info.config(text="Building STEP preview…")

        def work():
            try:
                from .viewer import build_scene  # reuse the proven tessellate+deviation path
                stl_poly, step_poly, stats = build_scene(stl_path, step_path, 0.1, fc)
                hi = max(stats["p95"], stats["max"] * 0.5, 1e-6)
                self._scenes["step"] = {
                    "meshes": [(step_poly, dict(color="#7dd3fc", smooth_shading=True))],
                    "hint": f"output STEP — {step_poly.n_cells:,} tris",
                }
                self._scenes["heatmap"] = {
                    "meshes": [
                        (stl_poly, dict(color="#334155", opacity=0.12)),
                        (step_poly, dict(scalars="deviation", cmap="jet", clim=[0.0, hi],
                                         scalar_bar_args={"title": "dev (mm)"})),
                    ],
                    "hint": (f"deviation (mm)  max={stats['max']:.3f}  rms={stats['rms']:.3f}  "
                             f"p95={stats['p95']:.3f}  mean={stats['mean']:.3f}"),
                }
                self._q.put(lambda: self._on_result_ready())
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: self.info.config(text=f"STEP preview failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    # ---- internal ---------------------------------------------------------
    def _on_scene_ready(self, name: str):
        self._btns[name].config(state="normal")
        # Reset orientation for a fresh part, then show it.
        self._azim = self._elev = 0.0
        self._zoom = 1.0
        self.set_view(name)

    def _on_result_ready(self):
        for name in ("step", "heatmap"):
            if name in self._scenes:
                self._btns[name].config(state="normal")
        self.set_view("heatmap")

    def set_view(self, name: str):
        if name not in self._scenes:
            return
        self._active = name
        for n, b in self._btns.items():
            b.state(["pressed"] if n == name else ["!pressed"])
        self._render()

    def reset_view(self):
        self._azim = self._elev = 0.0
        self._zoom = 1.0
        self._render()

    def _ensure_plotter(self, w: int, h: int):
        pv = self._pv()
        if pv is None:
            return None
        if self._plotter is None or self._plotter_size != (w, h):
            if self._plotter is not None:
                try:
                    self._plotter.close()
                except Exception:  # noqa: BLE001
                    pass
            self._plotter = pv.Plotter(off_screen=True, window_size=[w, h])
            self._plotter.set_background(BG)
            self._plotter_size = (w, h)
            self._built_view = None
        return self._plotter

    def _render(self):
        if self._active is None:
            return
        w = max(self.canvas.winfo_width(), 64)
        h = max(self.canvas.winfo_height(), 64)
        pl = self._ensure_plotter(w, h)
        if pl is None:
            return
        try:
            if self._built_view != self._active:
                pl.clear()
                for poly, kwargs in self._scenes[self._active]["meshes"]:
                    pl.add_mesh(poly, **kwargs)
                self._built_view = self._active
            # Absolute camera each frame: reset to iso, then apply orbit + zoom.
            pl.camera_position = "iso"
            cam = pl.camera
            cam.Azimuth(self._azim)
            cam.Elevation(self._elev)
            cam.OrthogonalizeViewUp()
            cam.Zoom(self._zoom)
            pl.renderer.ResetCameraClippingRange()
            img = pl.screenshot(return_img=True)
            self._show_array(img)
            self.info.config(text=self._scenes[self._active].get("hint", ""))
        except Exception as exc:  # noqa: BLE001
            self.show_message(f"render error:\n{exc}")

    def _show_array(self, img):
        try:
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(img))
        except Exception:  # noqa: BLE001 - PIL absent: hand Tk a PPM it can read natively
            photo = self._array_to_ppm_photo(img)
        self._photo = photo  # keep a ref so Tk doesn't garbage-collect the image
        self.canvas.config(image=photo, text="")

    @staticmethod
    def _array_to_ppm_photo(img):
        import os
        import tempfile
        h, w = img.shape[:2]
        path = os.path.join(tempfile.gettempdir(), "mesh2step_preview.ppm")
        with open(path, "wb") as fh:
            fh.write(b"P6\n%d %d\n255\n" % (w, h))
            fh.write(img[:, :, :3].astype("uint8").tobytes())
        return tk.PhotoImage(file=path)

    # ---- interaction ------------------------------------------------------
    def _on_press(self, e):
        self._drag = (e.x, e.y)

    def _on_drag(self, e):
        if self._drag is None or self._active is None:
            return
        dx = e.x - self._drag[0]
        dy = e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self._azim -= dx * 0.5
        self._elev = max(-89.0, min(89.0, self._elev + dy * 0.5))
        self._render()

    def _on_wheel(self, e):
        if self._active is None:
            return
        self._zoom *= 1.1 if e.delta > 0 else (1 / 1.1)
        self._zoom = max(0.2, min(8.0, self._zoom))
        self._render()

    def _on_resize(self, _e):
        # Debounce: re-render shortly after the user stops dragging the border.
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._render)
