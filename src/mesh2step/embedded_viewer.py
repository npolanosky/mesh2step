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

# Meshes above this triangle count get a decimated PREVIEW copy used for drag
# frames only; the settle frame and normal renders keep full quality.
_DECIMATE_ABOVE = 150_000
_DECIMATE_TARGET = 100_000

# Render-server hygiene: liveness ping while idle, and a memory ceiling above
# which the server is recycled between requests (VTK/allocator retention grows
# its RSS over many large scenes; a respawn is cheap and invisible).
_PING_INTERVAL_MS = 20_000
_PING_TIMEOUT = 5.0
_SERVER_RSS_CAP_MB = 900
_RSS_CHECK_EVERY = 10       # requests between RSS checks

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


class _ServerGone(RuntimeError):
    """The render server died or closed its pipe mid-request (retryable)."""


def _rss_mb(pid: int) -> float:
    """Resident set size of ``pid`` in MB (0 on any failure)."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      text=True, creationflags=_NO_WINDOW)
        return int(out.strip()) / 1024.0
    except Exception:  # noqa: BLE001
        return 0.0


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
        self._req_id = 0                           # render request counter
        self._scene_gen = 0                        # bumped when a scene is replaced
        self._load_seq = 0                         # bumped per file selection
        self._reqs_since_rss_check = 0
        self._retry_btn = None                     # recovery button (static path)

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
            # Visible recovery affordance: shown centred over the panel when a
            # render fails; never leaves the panel permanently dead.
            self._retry_btn = ttk.Button(self._surface, text="Retry preview",
                                         command=self._retry_render)
            # Pre-warm the render server at startup so the first preview only
            # costs the render, not the multi-second pyvista/frozen bootstrap.
            self.after(50, self._prewarm_server)
            # Watchdog: periodic liveness ping while idle; a dead/hung server is
            # recycled so the next render Just Works (covers native crashes and
            # processes killed across a sleep/wake).
            self.after(_PING_INTERVAL_MS, self._watchdog)

        self._hint = tk.Label(self, text="Select an STL to preview", bg=BG_HEX, fg=MUTED,
                              anchor="w", font=("Consolas", 9))
        self._hint.pack(fill="x", padx=8, pady=(0, 6))

        self.after(15, self._poll)  # snappy delivery of finished renders

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
        self.after(15, self._poll)  # snappy delivery of finished renders

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
        self._load_seq += 1   # orphan any scene builds still in flight
        self._camera = {"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0,
                        "pan": [0.0, 0.0]}
        self._hint.config(text="Select an STL to preview")

    def show_stl(self, stl_path: str, problem_points=None):
        self._hint.config(text="Loading mesh…")
        # Load-sequence token: scene building runs on worker threads and can
        # finish OUT OF ORDER (a big file selected first can land after a small
        # file selected second). Installs from a superseded show_stl are dropped.
        self._load_seq += 1
        tok = self._load_seq
        if not _LIVE_EMBED:
            # Start the render server now so its cold start (frozen bootstrap +
            # pyvista import) overlaps with reading/writing the scene files.
            self._ensure_render_server()

        def work():
            try:
                import pyvista as pv
                mesh = pv.read(stl_path)
                # "path" alongside "poly": the renderer loads (and caches) the
                # original STL directly — no .vtp copy of the full mesh needed.
                specs = [dict(path=stl_path, poly=mesh, color=STL_COLOR)]
                npts = 0
                if problem_points:
                    import numpy as np
                    pts = np.asarray(problem_points, dtype=float)
                    if pts.size:
                        npts = len(pts)
                        specs.append(dict(poly=pv.PolyData(pts), color="#ef4444", points=True))
                hint = (f"input STL — {mesh.n_points:,} pts, {mesh.n_cells:,} tris"
                        + (f"   ⚠ {npts} defect markers" if npts else ""))
                specs, decimated = self._materialise(specs, "stl")
                if decimated:
                    hint += "   · drag preview decimated"
                self._q.put(lambda: self._install_if_current(
                    tok, "stl", specs, hint, reset=True))
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: (tok == self._load_seq and
                                           self._hint.config(text=f"Could not load mesh: {e}")))

        threading.Thread(target=work, daemon=True).start()

    def show_result(self, stl_path: str, step_path: str):
        fc = self._get_fc()
        self._hint.config(text="Building STEP preview…")
        # Complements the CURRENT input scene — don't bump the sequence, but do
        # capture it so a result landing after the user switched files is dropped.
        tok = self._load_seq
        if not _LIVE_EMBED:
            self._ensure_render_server()  # warm it during tessellation

        def work():
            try:
                from .viewer import build_scene
                stl_poly, step_poly, stats = build_scene(stl_path, step_path, 0.1, fc)
                hi = max(stats["p95"], stats["max"] * 0.5, 1e-6)
                step_specs = [dict(poly=step_poly, color=STEP_COLOR)]
                heat_specs = [dict(path=stl_path, poly=stl_poly, color=GHOST_COLOR,
                                   opacity=0.15),
                              dict(poly=step_poly, scalars="deviation", clim=(0.0, hi))]
                step_hint = f"output STEP — {step_poly.n_cells:,} tris"
                heat_hint = (f"deviation (mm)  max={stats['max']:.3f}  rms={stats['rms']:.3f}  "
                             f"p95={stats['p95']:.3f}  mean={stats['mean']:.3f}")
                step_specs, dec1 = self._materialise(step_specs, "step")
                heat_specs, dec2 = self._materialise(heat_specs, "heatmap")
                if dec1:
                    step_hint += "   · drag preview decimated"
                if dec2:
                    heat_hint += "   · drag preview decimated"

                def apply():
                    self._install_if_current(tok, "step", step_specs, step_hint,
                                             reset=False, activate=False)
                    self._install_if_current(tok, "heatmap", heat_specs, heat_hint,
                                             reset=False, activate=True)
                self._q.put(apply)
            except Exception as exc:  # noqa: BLE001
                self._q.put(lambda e=exc: (tok == self._load_seq and
                                           self._hint.config(text=f"STEP preview failed: {e}")))

        threading.Thread(target=work, daemon=True).start()

    def _install_if_current(self, tok: int, name, specs, hint, reset, activate=True):
        """Install a built scene unless a newer file selection superseded it."""
        if tok != self._load_seq:
            _log.info("dropping stale scene install of %s (load %d != %d)",
                      name, tok, self._load_seq)
            return
        self._install(name, specs, hint, reset, activate)

    # ---- scene assembly ---------------------------------------------------
    def _materialise(self, specs, name):
        """Prepare file-backed scene specs for the render subprocess.

        Returns ``(specs, decimated)``. Polys that already have a source
        ``path`` (the original STL) are passed by reference — the server loads
        and caches the file itself, so no .vtp copy of the full mesh is
        written. Computed polys (tessellated STEP, heatmap, defect markers) are
        saved to .vtp. Very large meshes additionally get a decimated PREVIEW
        copy (``path_low``) used only for drag frames; full quality is kept for
        the settle frame. Runs on the producing worker thread (heavy work off
        the UI thread). Pass-through on the live (Windows) path.
        """
        if _LIVE_EMBED:
            return specs, False
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="mesh2step_preview_")
            atexit.register(self._cleanup_tmpdir)
        out, decimated = [], False
        for i, spec in enumerate(specs):
            fspec = {k: v for k, v in spec.items() if k != "poly"}
            poly = spec.get("poly")
            if (poly is not None and not spec.get("points")
                    and spec.get("scalars") is None
                    and poly.n_cells > _DECIMATE_ABOVE):
                try:
                    frac = 1.0 - (_DECIMATE_TARGET / float(poly.n_cells))
                    low = poly.decimate(frac)
                    low_path = os.path.join(self._tmpdir, f"{name}_{i}_low.vtp")
                    low.save(low_path)
                    fspec["path_low"] = low_path
                    decimated = True
                except Exception:  # noqa: BLE001 - decimation is best-effort
                    pass
            if "path" not in fspec:
                path = os.path.join(self._tmpdir, f"{name}_{i}.vtp")
                poly.save(path)
                fspec["path"] = path
            out.append(fspec)
        return out, decimated

    def _cleanup_tmpdir(self):
        import shutil

        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _install(self, name, specs, hint, reset, activate=True):
        self._scene_hint[name] = hint
        self._btns[name].config(state="normal")
        if not _LIVE_EMBED:
            self._specs[name] = specs      # file-based specs (see _materialise)
            self._images.pop(name, None)   # scene changed; drop any stale image
            self._scene_gen += 1           # in-flight renders of the old scene are stale
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
        interactive = self._interactive_render
        # Drag frames: reduced resolution + JPEG (faster encode/decode) + the
        # decimated preview copy of very large meshes. Settle frames and normal
        # renders use full resolution, PNG, and the full mesh.
        scale = _DRAG_DOWNSCALE if interactive else 1
        meshes = specs
        if interactive:
            meshes = [dict(m, path=m["path_low"]) if "path_low" in m else m
                      for m in specs]
        ext = "jpg" if interactive else "png"
        gen = self._scene_gen           # drop the result if the scene changes
        spec = {
            "width": int(w // scale), "height": int(h // scale), "background": BG_HEX,
            "mode": self._mode, "meshes": meshes,
            # Snapshot the camera NOW — the dicts keep mutating during a drag.
            "camera": {"azimuth": self._camera["azimuth"],
                       "elevation": self._camera["elevation"],
                       "zoom": self._camera["zoom"],
                       "pan": list(self._camera["pan"])},
            "out": os.path.join(self._tmpdir or tempfile.gettempdir(),
                                f"render_{name}.{ext}"),
        }

        def work():
            t0 = time.monotonic()
            try:
                png = self._render_via_server(spec)
                _log.info("preview render %s (%dx%d) took %.2fs",
                          name, spec["width"], spec["height"], time.monotonic() - t0)
                self._q.put(lambda p=png: self._show_image(name, p, (int(w), int(h)),
                                                           gen))
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
        """Send one render request to the server; return the image path.

        Serialised by a lock (the server handles one request at a time). Every
        request carries a fresh id and only the matching response is accepted —
        anything else on the pipe (stray output, a leftover answer from a
        recycled server) is dropped. A dead server is restarted once; a request
        that exceeds the timeout kills the server (respawned on the next
        request) and raises.
        """
        with self._server_lock:
            self._req_id += 1
            spec = dict(spec, id=self._req_id)
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
                try:
                    resp = self._read_response(proc, spec["id"], _RENDER_TIMEOUT)
                except _ServerGone as exc:
                    last_err = exc
                    self._kill_render_server()
                    continue
                if not resp["ok"]:
                    raise RuntimeError(resp.get("error", "render failed"))
                self._maybe_recycle_server(proc)
                return resp["png"]
            raise RuntimeError(f"render server unavailable ({last_err})")

    def _read_response(self, proc, req_id, timeout: float) -> dict:
        """Read the response matching ``req_id``; raise on death or timeout."""
        import select

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_render_server()
                raise RuntimeError(f"render timed out after {timeout:.0f}s")
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                if proc.poll() is not None:
                    raise _ServerGone("render server died")
                continue
            line = proc.stdout.readline()
            if not line:
                raise _ServerGone("render server closed its pipe")
            try:
                resp = json.loads(line)
            except ValueError:
                continue  # stray non-protocol output; keep reading
            if not isinstance(resp, dict) or "ok" not in resp:
                continue
            if resp.get("id") is not None and resp["id"] != req_id:
                continue  # stale response from an earlier (timed-out) request
            return resp

    def _maybe_recycle_server(self, proc) -> None:
        """Recycle the server if its RSS crossed the cap (checked periodically).

        VTK/allocator retention grows the server over many large scenes; a
        respawn between requests is invisible (next render re-pays only the
        import) and keeps memory bounded. Called with the lock held.
        """
        self._reqs_since_rss_check += 1
        if self._reqs_since_rss_check < _RSS_CHECK_EVERY:
            return
        self._reqs_since_rss_check = 0
        rss = _rss_mb(proc.pid)
        if rss > _SERVER_RSS_CAP_MB:
            _log.info("render server RSS %.0f MB > %d MB cap — recycling",
                      rss, _SERVER_RSS_CAP_MB)
            self._kill_render_server()

    # ---- watchdog / recovery -----------------------------------------------
    def _watchdog(self):
        """Periodic liveness check; replaces a dead or hung idle server."""
        try:
            if self._render_proc is not None and self._server_lock.acquire(False):
                try:
                    proc = self._render_proc
                    if proc is not None:
                        if proc.poll() is not None:
                            _log.warning("render server died while idle — respawning")
                            self._kill_render_server()
                            self._ensure_render_server()
                        elif not self._ping_server(proc):
                            _log.warning("render server unresponsive — recycling")
                            self._kill_render_server()
                            self._ensure_render_server()
                finally:
                    self._server_lock.release()
            # If the lock is busy a render is in flight — its own timeout covers
            # a hang, so the watchdog just skips this round.
        except Exception:  # noqa: BLE001 - watchdog must never take the app down
            pass
        self.after(_PING_INTERVAL_MS, self._watchdog)

    def _ping_server(self, proc) -> bool:
        """True if the server answers a ping within _PING_TIMEOUT seconds."""
        self._req_id += 1
        try:
            proc.stdin.write(json.dumps({"ping": True, "id": self._req_id}) + "\n")
            proc.stdin.flush()
            resp = self._read_response(proc, self._req_id, _PING_TIMEOUT)
            return bool(resp.get("ok"))
        except Exception:  # noqa: BLE001
            return False

    def _retry_render(self):
        """User-visible recovery: re-render the active view after a failure."""
        if self._retry_btn is not None:
            self._retry_btn.place_forget()
        self._hint.config(text="Retrying preview…")
        if self._active in self._specs:
            self._render_static(self._active)
        elif self._specs:
            self._render_static(next(iter(self._specs)))

    def _show_image(self, name: str, png_path: str, size: tuple, gen: int = -1):
        self._rendering = False
        if gen >= 0 and gen != self._scene_gen:
            # The scene changed while this frame rendered (new file selected) —
            # never show a stale image for the wrong content. If the view is
            # still wanted, re-render it against the fresh scene.
            _log.info("dropping stale render of %s (gen %d != %d)",
                      name, gen, self._scene_gen)
            if name == self._active and name in self._specs:
                self._render_static(name)
            else:
                self._kick_pending()
            return
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
        if self._retry_btn is not None:
            self._retry_btn.place_forget()
        if self._active == name and self._image_label is not None:
            self._image_label.config(image=photo, text="")
            self._displayed = name
            self._hint.config(text=self._scene_hint.get(name, ""))
        self._kick_pending()

    def _on_render_error(self, exc):
        self._rendering = False
        if self._image_label is not None and not self._image_label.cget("image"):
            self._image_label.config(text="Preview render failed")
        if self._retry_btn is not None and self._specs:
            self._retry_btn.place(relx=0.5, rely=0.5, anchor="center")
        self._hint.config(text=f"Preview render failed: {exc} — Retry, or Pop out ↗.")
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
