"""In-window 3D preview panel (STL / STEP / deviation heatmap).

Embeds a **real, GPU-accelerated VTK render window** directly in the tkinter
GUI by reparenting it into a Tk frame (``vtkRenderWindow.SetParentInfo`` with the
frame's native window id). This gives true trackball interaction (rotate/zoom/
pan), correct lighting and edges — the same engine as the standalone viewer,
just docked — instead of streaming off-screen screenshots (which were slow and
unlit). The PyPI VTK build ships no working Tk render widget, so reparenting is
the route to native interaction.

States it shows through a conversion:
  * the input STL as soon as a file is selected (defect regions marked red)
  * the output STEP once the conversion finishes
  * the STEP coloured by its deviation from the STL (heatmap)

Reparenting relies on a native window id, so it is Windows-only; elsewhere the
panel shows a hint and the "Pop out" button opens the standalone window.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

BG = (0.06, 0.09, 0.16)      # matches the app's dark panel
FG = "#e2e8f0"
MUTED = "#94a3b8"


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class EmbeddedViewer(ttk.Frame):
    def __init__(self, parent, freecad_python_getter=None, on_popout=None):
        super().__init__(parent, style="Bg.TFrame")
        self._get_fc = freecad_python_getter or (lambda: None)

        self._scenes: dict[str, list] = {}       # name -> list of vtk actors
        self._active: str | None = None
        self._mode = "shaded"                     # shaded | edges | wire
        self._q: queue.Queue = queue.Queue()
        self._vtk = None                          # (renwin, renderer, interactor) once created
        self._vtk_failed = False

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
        self._mode_var = tk.StringVar(value="Shaded")
        mode_cb = ttk.Combobox(bar, textvariable=self._mode_var, width=13, state="readonly",
                               values=["Shaded", "Shaded + edges", "Wireframe"])
        mode_cb.pack(side="right", padx=(0, 8))
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode)
        ttk.Label(bar, text="Display", style="Muted.TLabel").pack(side="right", padx=(0, 4))

        # --- native render surface (VTK reparents into this frame) ---
        self._surface = tk.Frame(self, bg="#0f172a", width=640, height=480)
        self._surface.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._surface.bind("<Configure>", self._on_surface_configure)

        self._hint = tk.Label(self, text="Select an STL to preview", bg="#0f172a", fg=MUTED,
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

    # ---- VTK setup --------------------------------------------------------
    def _ensure_vtk(self):
        """Create the embedded GPU render window on first use. Returns it or None."""
        if self._vtk is not None or self._vtk_failed:
            return self._vtk
        if sys.platform != "win32":
            self._vtk_failed = True
            self._hint.config(text="Interactive preview is Windows-only — use Pop out ↗.")
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
        v = self._ensure_vtk()
        if v is not None:
            renwin, _ren, _iren = v
            renwin.SetSize(max(self._surface.winfo_width(), 1),
                           max(self._surface.winfo_height(), 1))
            renwin.Render()

    # ---- public API -------------------------------------------------------
    def clear(self):
        self._scenes.clear()
        self._active = None
        for b in self._btns.values():
            b.config(state="disabled")
        if self._vtk:
            self._vtk[1].RemoveAllViewProps()
            self._vtk[0].Render()
        self._hint.config(text="Select an STL to preview")

    def show_stl(self, stl_path: str, problem_points=None):
        self._hint.config(text="Loading mesh…")

        def work():
            try:
                import pyvista as pv
                mesh = pv.read(stl_path)
                specs = [dict(poly=mesh, color="#cbd5e1")]
                npts = 0
                if problem_points:
                    import numpy as np
                    pts = np.asarray(problem_points, dtype=float)
                    if pts.size:
                        npts = len(pts)
                        specs.append(dict(poly=pv.PolyData(pts), color="#ef4444", points=True))
                hint = (f"input STL — {mesh.n_points:,} pts, {mesh.n_cells:,} tris"
                        + (f"   ⚠ {npts} defect markers" if npts else ""))
                self._q.put(lambda: self._install("stl", specs, hint, reset=True))
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: self._hint.config(text=f"Could not load mesh: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def show_result(self, stl_path: str, step_path: str):
        fc = self._get_fc()
        self._hint.config(text="Building STEP preview…")

        def work():
            try:
                from .viewer import build_scene
                stl_poly, step_poly, stats = build_scene(stl_path, step_path, 0.1, fc)
                hi = max(stats["p95"], stats["max"] * 0.5, 1e-6)
                step_specs = [dict(poly=step_poly, color="#7dd3fc")]
                heat_specs = [dict(poly=stl_poly, color="#334155", opacity=0.12),
                              dict(poly=step_poly, scalars="deviation", clim=(0.0, hi))]
                step_hint = f"output STEP — {step_poly.n_cells:,} tris"
                heat_hint = (f"deviation (mm)  max={stats['max']:.3f}  rms={stats['rms']:.3f}  "
                             f"p95={stats['p95']:.3f}  mean={stats['mean']:.3f}")

                def apply():
                    self._install("step", step_specs, step_hint, reset=False, activate=False)
                    self._install("heatmap", heat_specs, heat_hint, reset=False, activate=True)
                self._q.put(apply)
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: self._hint.config(text=f"STEP preview failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    # ---- scene assembly (main thread) ------------------------------------
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

    def _install(self, name, specs, hint, reset, activate=True):
        v = self._ensure_vtk()
        if v is None:
            # No GPU surface yet; stash specs to build on activation/configure.
            self._scenes[name] = specs
            self._btns[name].config(state="normal")
            if activate:
                self._active = name
                self._hint.config(text=hint + "   (preview initialising…)")
                self.after(120, lambda: self.set_view(name))
            return
        actors = [self._make_actor(s) for s in specs]
        self._scenes[name] = actors
        self._scene_hint = getattr(self, "_scene_hint", {})
        self._scene_hint[name] = hint
        self._btns[name].config(state="normal")
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
        self._hint.config(text=getattr(self, "_scene_hint", {}).get(name, ""))
        renwin.Render()

    # ---- view controls ----------------------------------------------------
    def set_view(self, name: str):
        if name in self._scenes:
            self._show_actors(name, reset=False)

    def reset_view(self):
        v = self._ensure_vtk()
        if v is not None:
            v[1].ResetCamera()
            v[0].Render()

    def _on_mode(self, _e=None):
        self._mode = {"Shaded": "shaded", "Shaded + edges": "edges",
                      "Wireframe": "wire"}.get(self._mode_var.get(), "shaded")
        v = self._ensure_vtk()
        if v is None or self._active is None:
            return
        for a in self._scenes.get(self._active, []):
            if not isinstance(a, dict):
                self._apply_mode_actor(a)
        v[0].Render()

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
