"""Modern drag-and-drop GUI for STL -> STEP conversion.

Runs under an ordinary Python with tkinter (no numpy/FreeCAD needed). It shells
out to :mod:`mesh2step.worker` using FreeCAD's bundled Python for the actual
mesh inspection and conversion, streaming the worker's progress lines into an
in-app log pane.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from .config import UNIT_SCALE_MM
from .freecad_env import find_freecad_python

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND = True
except Exception:  # noqa: BLE001
    _DND = False

UNIT_CHOICES = ["mm", "cm", "m", "in"]

# Run child processes without flashing a console window on Windows.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Map a progress message substring to a percent, for a determinate progress bar.
_MILESTONES = [
    ("Locating FreeCAD", 4), ("Preparing mesh", 8), ("Loading", 12),
    ("Scaling", 16), ("Detecting cylinders", 28), ("Found", 34),
    ("countersink", 38), ("Segmenting", 48), ("Building", 62),
    ("Gap-filling", 68), ("local patch", 70), ("merging large patch", 72),
    ("gap patches merged", 76), ("Sewing", 82), ("sewShape", 86),
    ("watertight faceted solid", 88), ("faceted solid", 88),
    ("Exporting", 94), ("Done", 100),
]

# If the worker emits no output for this long, warn that it may be stalled.
_STALL_SECONDS = 25.0

# Palette — a clean flat light theme with a blue accent and a console-style log.
BG = "#eef1f5"
CARD = "#ffffff"
ACCENT = "#3b82f6"
ACCENT_ACTIVE = "#2563eb"
TEXT = "#1f2937"
MUTED = "#6b7280"
BORDER = "#dfe3e8"
LOG_BG = "#0f172a"
LOG_FG = "#cbd5e1"
OK_GREEN = "#16a34a"
ERR_RED = "#dc2626"


def _package_src() -> str:
    """Directory to put on PYTHONPATH so FreeCAD's Python can import this pkg."""
    if getattr(sys, "frozen", False):
        return str(Path(getattr(sys, "_MEIPASS", ".")) / "mesh2step_src")
    return str(Path(__file__).resolve().parent.parent)


class WorkerError(RuntimeError):
    pass


def run_worker(job: dict, freecad_python: str, on_line=None, timeout: float = 1800) -> dict:
    """Run one worker job out-of-process, streaming stdout lines to ``on_line``."""
    with tempfile.TemporaryDirectory() as tmp:
        job_file = Path(tmp) / "job.json"
        res_file = Path(tmp) / "result.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _package_src() + (os.pathsep + existing if existing else "")

        proc = subprocess.Popen(
            [freecad_python, "-m", "mesh2step.worker",
             "--job", str(job_file), "--result", str(res_file)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=_NO_WINDOW,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line and on_line:
                on_line(line)
        proc.wait(timeout=timeout)
        if not res_file.exists():
            raise WorkerError(f"worker produced no result (exit {proc.returncode})")
        return json.loads(res_file.read_text(encoding="utf-8-sig"))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("mesh2step")
        root.geometry("760x820")
        root.minsize(680, 720)
        root.configure(bg=BG)
        self.q: queue.Queue = queue.Queue()
        self.busy = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.units_var = tk.StringVar(value="mm")
        self.detect_var = tk.BooleanVar(value=True)
        self.faceted_var = tk.BooleanVar(value=False)
        self.repair_var = tk.BooleanVar(value=False)
        self.closed_var = tk.BooleanVar(value=False)
        self.freecad_var = tk.StringVar(value=find_freecad_python() or "")
        self._longest_mm = None
        self._t0 = 0.0
        self._last_line_t = 0.0
        self._stall_noted = False
        self.last_stl = None
        self.last_step = None

        self._init_style()
        self._build()
        self.root.after(80, self._drain_queue)
        if not self.freecad_var.get():
            self._log("⚠  FreeCAD not found — set its python path below.", "err")
        else:
            self._log(f"FreeCAD: {self.freecad_var.get()}", "muted")

    # ---- styling ----------------------------------------------------------
    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Card.TFrame", background=CARD)
        style.configure("Bg.TFrame", background=BG)
        style.configure("TLabel", background=CARD, foreground=TEXT)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Head.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI Semibold", 11))
        style.configure("Value.TLabel", background=CARD, foreground=TEXT, font=("Consolas", 10))
        style.configure("TCheckbutton", background=CARD, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", CARD)])
        style.configure("TCombobox", fieldbackground="#ffffff")
        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                        font=("Segoe UI Semibold", 11), borderwidth=0, padding=10)
        style.map("Accent.TButton",
                  background=[("active", ACCENT_ACTIVE), ("disabled", "#9ab6f0")])
        style.configure("TButton", padding=6)
        style.configure("Accent.Horizontal.TProgressbar", background=ACCENT)

    def _card(self, parent, title: str) -> ttk.Frame:
        outer = tk.Frame(parent, bg=BORDER)  # 1px border via padding
        outer.pack(fill="x", pady=(0, 12))
        inner = ttk.Frame(outer, style="Card.TFrame", padding=14)
        inner.pack(fill="x", padx=1, pady=1)
        ttk.Label(inner, text=title, style="Head.TLabel").pack(anchor="w", pady=(0, 8))
        return inner

    # ---- layout -----------------------------------------------------------
    def _build(self):
        # Header band
        header = tk.Frame(self.root, bg=ACCENT)
        header.pack(fill="x")
        tk.Label(header, text="mesh2step", bg=ACCENT, fg="white",
                 font=("Segoe UI Semibold", 18)).pack(anchor="w", padx=18, pady=(12, 0))
        tk.Label(header, text="STL mesh → STEP solid  ·  surface & hole reconstruction",
                 bg=ACCENT, fg="#dbeafe", font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(0, 12))

        body = ttk.Frame(self.root, style="Bg.TFrame", padding=16)
        body.pack(fill="both", expand=True)

        # --- Input card ---
        c1 = self._card(body, "1  ·  Input mesh")
        self.drop = tk.Label(
            c1, text="Drag an STL here" + ("" if _DND else "  (or use Browse)"),
            bg="#f8fafc", fg=MUTED, font=("Segoe UI", 10),
            relief="flat", height=3, bd=1, highlightbackground=BORDER, highlightthickness=1,
        )
        self.drop.pack(fill="x")
        if _DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)
        row = ttk.Frame(c1, style="Card.TFrame"); row.pack(fill="x", pady=(8, 0))
        ttk.Entry(row, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse_input).pack(side="left", padx=(6, 0))

        # mesh info grid
        self.info = ttk.Frame(c1, style="Card.TFrame"); self.info.pack(fill="x", pady=(10, 0))
        self._info_labels = {}
        for i, key in enumerate(["Triangles", "AABB X·Y·Z", "OBB (oriented)", "Mesh health"]):
            ttk.Label(self.info, text=key, style="Muted.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 10))
            v = ttk.Label(self.info, text="—", style="Value.TLabel")
            v.grid(row=i, column=1, sticky="w")
            self._info_labels[key] = v

        # --- Units & options ---
        c2 = self._card(body, "2  ·  Units & options")
        urow = ttk.Frame(c2, style="Card.TFrame"); urow.pack(fill="x")
        ttk.Label(urow, text="Source units").pack(side="left")
        cb = ttk.Combobox(urow, textvariable=self.units_var, values=UNIT_CHOICES,
                          width=6, state="readonly")
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_units())
        self.units_preview = ttk.Label(urow, text="STEP output is always mm", style="Muted.TLabel")
        self.units_preview.pack(side="left", padx=8)
        ttk.Checkbutton(c2, text="Detect cylindrical holes / bosses (best-fit radius)",
                        variable=self.detect_var).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(c2, text="Repair mesh (fix self-intersections, duplicates, normals) — "
                                 "recovers holes on defective meshes",
                        variable=self.repair_var).pack(anchor="w")
        ttk.Checkbutton(c2, text="Fully closed (guarantee watertight; slower, faceted holes on "
                                 "organic parts)",
                        variable=self.closed_var).pack(anchor="w")
        ttk.Checkbutton(c2, text="Faceted only (skip reconstruction)",
                        variable=self.faceted_var).pack(anchor="w")

        # --- Output ---
        c3 = self._card(body, "3  ·  Output")
        orow = ttk.Frame(c3, style="Card.TFrame"); orow.pack(fill="x")
        ttk.Entry(orow, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(orow, text="Browse…", command=self._browse_output).pack(side="left", padx=(6, 0))
        frow = ttk.Frame(c3, style="Card.TFrame"); frow.pack(fill="x", pady=(8, 0))
        ttk.Label(frow, text="FreeCAD Python", style="Muted.TLabel").pack(side="left")
        ttk.Entry(frow, textvariable=self.freecad_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(frow, text="…", width=3, command=self._browse_freecad).pack(side="left")

        # --- Convert + progress ---
        self.convert_btn = ttk.Button(body, text="Convert  →  STEP",
                                      style="Accent.TButton", command=self._convert)
        self.convert_btn.pack(fill="x")
        prow = ttk.Frame(body, style="Bg.TFrame"); prow.pack(fill="x", pady=(8, 4))
        self.progress = ttk.Progressbar(prow, mode="determinate", maximum=100,
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True)
        self.elapsed = tk.Label(prow, text="0.0s", bg=BG, fg=MUTED, width=8,
                                font=("Consolas", 9))
        self.elapsed.pack(side="left", padx=(8, 0))
        self.status = tk.Label(body, text="Ready", bg=BG, fg=MUTED, anchor="w",
                               font=("Segoe UI", 9))
        self.status.pack(fill="x")
        self.quality = tk.Label(body, text="", bg=BG, fg=MUTED, anchor="w",
                                font=("Segoe UI Semibold", 11))
        self.quality.pack(fill="x", pady=(2, 6))

        # --- Result actions ---
        arow = ttk.Frame(body, style="Bg.TFrame"); arow.pack(fill="x", pady=(0, 6))
        self.view_btn = ttk.Button(arow, text="View 3D deviation", state="disabled",
                                   command=self._view_result)
        self.view_btn.pack(side="left")
        ttk.Button(arow, text="Save log…", command=self._save_log).pack(side="left", padx=6)

        # --- Log ---
        logcard = self._card(body, "Log")
        self.log = ScrolledText(logcard, height=9, bg=LOG_BG, fg=LOG_FG,
                                insertbackground=LOG_FG, font=("Consolas", 9),
                                relief="flat", bd=0, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("muted", foreground="#64748b")
        self.log.tag_config("ok", foreground="#4ade80")
        self.log.tag_config("err", foreground="#f87171")
        self.log.tag_config("stage", foreground="#93c5fd")

    # ---- actions ----------------------------------------------------------
    def _browse_input(self):
        p = filedialog.askopenfilename(filetypes=[("STL mesh", "*.stl"), ("All", "*.*")])
        if p:
            self._set_input(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".step",
                                         filetypes=[("STEP", "*.step *.stp")])
        if p:
            self.output_var.set(p)

    def _browse_freecad(self):
        p = filedialog.askopenfilename(title="FreeCAD's python executable")
        if p:
            self.freecad_var.set(p)
            self._log(f"FreeCAD: {p}", "muted")

    def _on_drop(self, event):
        self._set_input(event.data.strip().strip("{}"))

    def _set_input(self, path: str):
        self.input_var.set(path)
        self.drop.config(text=Path(path).name, fg=TEXT)
        if not self.output_var.get():
            self.output_var.set(str(Path(path).with_suffix(".step")))
        self._inspect(path)

    def _refresh_units(self):
        if self._longest_mm is None:
            return
        factor = UNIT_SCALE_MM[self.units_var.get()]
        self.units_preview.config(text=f"longest edge → {self._longest_mm * factor:,.2f} mm")

    def _freecad(self) -> str | None:
        fc = self.freecad_var.get().strip()
        if not fc or not Path(fc).is_file():
            self._log("⚠  Set the path to FreeCAD's python executable first.", "err")
            return None
        return fc

    def _inspect(self, path: str):
        fc = self._freecad()
        if not fc or self.busy:
            return
        self._start("Inspecting mesh…")
        self._run({"mode": "inspect", "input": path, "config": {}}, fc, "inspect")

    def _convert(self):
        if self.busy:
            return
        path = self.input_var.get().strip()
        if not path or not Path(path).is_file():
            self._log("⚠  Choose an STL file first.", "err")
            return
        fc = self._freecad()
        if not fc:
            return
        self._start("Converting…")
        job = {
            "mode": "convert",
            "input": path,
            "output": self.output_var.get().strip() or None,
            "config": {
                "source_units": self.units_var.get(),
                "detect_cylinders": self.detect_var.get(),
                "faceted": self.faceted_var.get(),
                "repair_mesh": self.repair_var.get(),
                "full_closed": self.closed_var.get(),
            },
        }
        self._run(job, fc, "convert")

    # ---- worker plumbing --------------------------------------------------
    def _run(self, job, fc, kind):
        def worker():
            try:
                result = run_worker(job, fc, on_line=lambda ln: self.q.put(("log", ln)))
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": str(exc)}
            self.q.put((kind, result))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._on_log_line(payload)
                elif kind == "inspect":
                    self._on_inspect(payload)
                else:
                    self._on_convert(payload)
        except queue.Empty:
            pass
        if self.busy:
            now = time.monotonic()
            self.elapsed.config(text=f"{now - self._t0:.1f}s")
            quiet = now - self._last_line_t
            if quiet > _STALL_SECONDS:
                # No worker output for a while — flag as possibly stalled (a big
                # mesh sew/faceted build can genuinely be quiet for minutes).
                self.status.config(
                    text=f"⏳ still working — no update for {quiet:.0f}s "
                         f"(large meshes can be slow; watch the log)", fg="#b45309")
                self._stall_noted = True
            elif self._stall_noted:
                self._stall_noted = False
        self.root.after(80, self._drain_queue)

    def _on_log_line(self, line: str):
        self._last_line_t = time.monotonic()
        if line.startswith("PROGRESS:"):
            msg = line[len("PROGRESS:"):].strip()
            self.status.config(text=msg, fg=MUTED)
            for key, pct in _MILESTONES:
                if key in msg:
                    self.progress["value"] = max(self.progress["value"], pct)
                    break
            self._log(msg, "stage")
        else:
            self._log(line, "muted")
            if "Traceback" in line or "Error" in line:
                self._log("   (worker reported an error — see above)", "err")

    def _on_inspect(self, result: dict):
        self._stop()
        if not result.get("ok"):
            self._log(f"Inspect failed: {result.get('error','')}", "err")
            return
        aabb = result["aabb"]["dimensions"]
        xyz = result["aabb"].get("extents_xyz", aabb)
        obb = result["obb"]["dimensions"]
        self._longest_mm = aabb[0]
        self._refresh_units()
        self._info_labels["Triangles"].config(
            text=f"{result['triangle_count']:,}   ({result['vertex_count']:,} verts)")
        self._info_labels["AABB X·Y·Z"].config(
            text=f"X {xyz[0]:.2f}   Y {xyz[1]:.2f}   Z {xyz[2]:.2f}  mm-units")
        self._info_labels["OBB (oriented)"].config(
            text=f"{obb[0]:.2f} × {obb[1]:.2f} × {obb[2]:.2f}  (principal)")

        health = result.get("health", {})
        issues = []
        if health.get("non_manifold"):
            issues.append("non-manifold")
        if health.get("self_intersections"):
            issues.append("self-intersections")
        if health.get("watertight") is False:
            issues.append("not watertight")
        if "error" in health:
            self._info_labels["Mesh health"].config(text="(unavailable)")
        elif issues:
            self._info_labels["Mesh health"].config(text="⚠ " + ", ".join(issues))
            self.repair_var.set(True)  # auto-recommend repair
            self._log(f"⚠  Mesh defects: {', '.join(issues)} — "
                      f"'Repair mesh' enabled (recovers holes lost to these defects).", "err")
        else:
            self._info_labels["Mesh health"].config(text="✔ clean")
        self.status.config(text="Mesh inspected — set units and convert.")

    def _on_convert(self, result: dict):
        took = time.monotonic() - self._t0
        self._stop()
        if not result.get("ok"):
            self._log(f"✖  Conversion failed after {took:.1f}s: "
                      f"{result.get('error','unknown error')}", "err")
            self.status.config(text="Conversion failed.", fg=ERR_RED)
            self.quality.config(text="✖  FAILED", fg=ERR_RED)
            return
        s = result.get("stats", {})
        cyls = s.get("cylinders", [])
        holes = sum(1 for c in cyls if c.get("role") == "hole")
        radii = sorted({round(c["radius"] * 2, 3) for c in cyls})

        quality = s.get("quality", "good")
        badge = {"good": ("✔  GOOD", OK_GREEN),
                 "warnings": ("⚠  OK — with warnings", "#b45309"),
                 "problems": ("✖  PROBLEMS", ERR_RED)}.get(quality, ("done", MUTED))
        outputs = result.get("outputs") or [result["output"]]
        if len(outputs) > 1:
            self._log(f"✔  Wrote {len(outputs)} files:", "ok")
            for pth in outputs:
                tag = ("watertight (may have artifacts)" if "_watertight" in pth
                       else "artifact-free, open" if "_clean" in pth else "")
                self._log(f"     • {Path(pth).name}  — {tag}", "ok")
        else:
            self._log(f"✔  Wrote {result['output']}", "ok")
        self._log(f"   method={result['method']}  faces {s.get('faces_in')}→{s.get('faces_out')} "
                  f"({s.get('planar_faces',0)} planar, {s.get('cylinder_faces',0)} cyl)", "muted")
        self._log(f"   holes={holes}  bosses={len(cyls)-holes}  diameters(mm)={radii}", "muted")
        if s.get("gap_faces"):
            self._log(f"   gap-fill: {s.get('gap_patches', 0)} local patch(es) "
                      f"merged to {s['gap_faces']:,} faces (closing the solid)", "muted")
        cones = s.get("cones", [])
        if cones:
            angles = sorted({round(c["half_angle_deg"], 1) for c in cones})
            self._log(f"   countersinks: {s.get('cone_faces', 0)}/{len(cones)} built as "
                      f"cone faces (half-angles {angles}°)", "muted")
        bi, bo = s.get("bbox_input_mm"), s.get("bbox_output_mm")
        if bi and bo:
            self._log(f"   bbox in {bi} → out {bo}  (Δ{s.get('bbox_delta_pct',0)}%)", "muted")
        for w in s.get("warnings", []):
            self._log(f"   ⚠ {w}", "err")
        # Prominent quality verdict.
        self.quality.config(text=f"{badge[0]}   ·   {result['method']}, {len(cyls)} cylinders, "
                                 f"watertight={s.get('is_solid')}", fg=badge[1])
        self.status.config(text=f"Done in {took:.1f}s → {Path(result['output']).name}", fg=badge[1])

        # Enable the deviation viewer for this result.
        self.last_stl = self.input_var.get().strip()
        self.last_step = result["output"]
        self.view_btn.config(state="normal")

    # ---- result actions ---------------------------------------------------
    def _view_result(self):
        if not (self.last_stl and self.last_step and Path(self.last_step).exists()):
            self._log("⚠  No result to view yet.", "err")
            return
        self._log("Opening 3D deviation viewer…", "muted")
        self._launch_viewer(self.last_stl, self.last_step)

    def _launch_viewer(self, stl: str, step: str):
        """Launch the pyvista deviation viewer as its own process."""
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--view", stl, step]
            env = None
        else:
            env = dict(os.environ)
            env["PYTHONPATH"] = _package_src() + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", "mesh2step.viewer", stl, step]
        try:
            subprocess.Popen(cmd, env=env, creationflags=_NO_WINDOW)
        except Exception as exc:  # noqa: BLE001
            self._log(f"⚠  Could not open viewer: {exc}", "err")

    def _save_log(self):
        text = self.log.get("1.0", "end").strip()
        if not text:
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Text", "*.txt")])
        if p:
            Path(p).write_text(text, encoding="utf-8")
            self._log(f"Log saved to {p}", "muted")

    # ---- ui helpers -------------------------------------------------------
    def _start(self, text: str):
        self.busy = True
        self._t0 = time.monotonic()
        self._last_line_t = self._t0
        self._stall_noted = False
        self.convert_btn.config(state="disabled")
        self.status.config(text=text, fg=MUTED)
        self.progress.config(mode="determinate")
        self.progress["value"] = 2

    def _stop(self):
        self.busy = False
        self.convert_btn.config(state="normal")
        self.progress["value"] = 100 if not self._stall_noted else self.progress["value"]
        self.elapsed.config(text=f"{time.monotonic() - self._t0:.1f}s")

    def _log(self, text: str, tag: str = ""):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")


def main() -> int:
    root = TkinterDnD.Tk() if _DND else tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
