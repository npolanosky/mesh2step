"""Drag-and-drop GUI for STL -> STEP conversion.

Runs under an ordinary Python with tkinter (no numpy/FreeCAD needed). It shells
out to :mod:`mesh2step.worker` using FreeCAD's bundled Python for the actual
mesh inspection and conversion, so the heavy CAD kernel stays out of this
process and the GUI can be packaged on its own.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import UNIT_SCALE_MM
from .freecad_env import find_freecad_python

# Optional drag-and-drop support.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND = True
except Exception:  # noqa: BLE001
    _DND = False

UNIT_CHOICES = ["mm", "cm", "m", "in"]


def _package_src() -> str:
    """Directory to put on PYTHONPATH so FreeCAD's Python can import this pkg."""
    if getattr(sys, "frozen", False):
        # PyInstaller: the package source is shipped as bundled data.
        return str(Path(getattr(sys, "_MEIPASS", ".")) / "mesh2step_src")
    return str(Path(__file__).resolve().parent.parent)


class WorkerError(RuntimeError):
    pass


def run_worker(job: dict, freecad_python: str, timeout: float = 600) -> dict:
    """Run one worker job out-of-process under FreeCAD's Python."""
    with tempfile.TemporaryDirectory() as tmp:
        job_file = Path(tmp) / "job.json"
        res_file = Path(tmp) / "result.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _package_src() + (os.pathsep + existing if existing else "")

        proc = subprocess.run(
            [freecad_python, "-m", "mesh2step.worker",
             "--job", str(job_file), "--result", str(res_file)],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
        if not res_file.exists():
            raise WorkerError(
                f"worker produced no result (exit {proc.returncode}).\n{proc.stderr[-2000:]}"
            )
        return json.loads(res_file.read_text(encoding="utf-8-sig"))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("mesh2step — STL to STEP")
        root.geometry("620x640")
        self.q: queue.Queue = queue.Queue()
        self.busy = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.units_var = tk.StringVar(value="mm")
        self.detect_var = tk.BooleanVar(value=True)
        self.faceted_var = tk.BooleanVar(value=False)
        self.freecad_var = tk.StringVar(value=find_freecad_python() or "")

        self._build()
        self.root.after(100, self._drain_queue)

    # ---- layout -----------------------------------------------------------
    def _build(self):
        pad = dict(padx=8, pady=4)
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        # Input
        ttk.Label(frm, text="Input STL", font=("", 10, "bold")).grid(row=0, column=0, sticky="w")
        row = ttk.Frame(frm); row.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Entry(row, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse_input).pack(side="left", padx=4)

        drop_hint = "drag an STL here or use Browse" if _DND else "use Browse to pick an STL"
        self.drop = tk.Label(frm, text=f"⬇  {drop_hint}", relief="groove",
                             height=2, fg="#555")
        self.drop.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        if _DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)

        # Mesh info
        self.info = tk.Text(frm, height=7, width=70, state="disabled",
                            bg="#f6f6f6", relief="flat")
        self.info.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)

        # Units
        ttk.Label(frm, text="Source units (STEP is always mm)").grid(row=4, column=0, sticky="w", **pad)
        units = ttk.Combobox(frm, textvariable=self.units_var, values=UNIT_CHOICES,
                             width=6, state="readonly")
        units.grid(row=4, column=1, sticky="w", **pad)
        units.bind("<<ComboboxSelected>>", lambda _e: self._refresh_units())
        self.units_preview = ttk.Label(frm, text="")
        self.units_preview.grid(row=4, column=2, sticky="w", **pad)

        # Options
        ttk.Checkbutton(frm, text="Detect cylindrical holes/bosses (best-fit radius)",
                        variable=self.detect_var).grid(row=5, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Faceted only (skip reconstruction)",
                        variable=self.faceted_var).grid(row=6, column=0, columnspan=3, sticky="w", **pad)

        # Output
        ttk.Label(frm, text="Output STEP", font=("", 10, "bold")).grid(row=7, column=0, sticky="w")
        row2 = ttk.Frame(frm); row2.grid(row=8, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Entry(row2, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Browse…", command=self._browse_output).pack(side="left", padx=4)

        # FreeCAD
        ttk.Label(frm, text="FreeCAD Python").grid(row=9, column=0, sticky="w", **pad)
        row3 = ttk.Frame(frm); row3.grid(row=10, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Entry(row3, textvariable=self.freecad_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row3, text="Browse…", command=self._browse_freecad).pack(side="left", padx=4)

        # Convert + status
        self.convert_btn = ttk.Button(frm, text="Convert  →  STEP", command=self._convert)
        self.convert_btn.grid(row=11, column=0, columnspan=3, sticky="ew", pady=10, padx=8)
        self.status = ttk.Label(frm, text="Ready.", foreground="#333")
        self.status.grid(row=12, column=0, columnspan=3, sticky="w", **pad)

        frm.columnconfigure(0, weight=1)

    # ---- actions ----------------------------------------------------------
    def _browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("STL mesh", "*.stl"), ("All", "*.*")])
        if path:
            self._set_input(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".step",
                                            filetypes=[("STEP", "*.step *.stp")])
        if path:
            self.output_var.set(path)

    def _browse_freecad(self):
        path = filedialog.askopenfilename(title="FreeCAD's python executable")
        if path:
            self.freecad_var.set(path)

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")  # tkdnd wraps spaced paths in braces
        self._set_input(path)

    def _set_input(self, path: str):
        self.input_var.set(path)
        if not self.output_var.get():
            self.output_var.set(str(Path(path).with_suffix(".step")))
        self._inspect(path)

    def _refresh_units(self):
        longest = getattr(self, "_longest_mm", None)
        if longest is None:
            self.units_preview.config(text="")
            return
        factor = UNIT_SCALE_MM[self.units_var.get()]
        self.units_preview.config(text=f"longest edge → {longest * factor:,.2f} mm")

    def _freecad(self) -> str | None:
        fc = self.freecad_var.get().strip()
        if not fc or not Path(fc).is_file():
            messagebox.showerror("FreeCAD not found",
                                 "Set the path to FreeCAD's bundled python executable "
                                 "(e.g. C:\\Program Files\\FreeCAD 1.1\\bin\\python.exe).")
            return None
        return fc

    def _inspect(self, path: str):
        fc = self._freecad()
        if not fc or self.busy:
            return
        self._set_busy("Inspecting mesh…")
        job = {"mode": "inspect", "input": path, "config": {}}
        threading.Thread(target=self._bg, args=(job, fc, "inspect"), daemon=True).start()

    def _convert(self):
        if self.busy:
            return
        path = self.input_var.get().strip()
        if not path or not Path(path).is_file():
            messagebox.showerror("No input", "Choose an STL file first.")
            return
        fc = self._freecad()
        if not fc:
            return
        self._set_busy("Converting…")
        job = {
            "mode": "convert",
            "input": path,
            "output": self.output_var.get().strip() or None,
            "config": {
                "source_units": self.units_var.get(),
                "detect_cylinders": self.detect_var.get(),
                "faceted": self.faceted_var.get(),
            },
        }
        threading.Thread(target=self._bg, args=(job, fc, "convert"), daemon=True).start()

    # ---- worker plumbing --------------------------------------------------
    def _bg(self, job: dict, fc: str, kind: str):
        try:
            result = run_worker(job, fc)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
        self.q.put((kind, result))

    def _drain_queue(self):
        try:
            while True:
                kind, result = self.q.get_nowait()
                (self._on_inspect if kind == "inspect" else self._on_convert)(result)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _on_inspect(self, result: dict):
        self._set_busy(None)
        if not result.get("ok"):
            self._info(f"Inspect failed:\n{result.get('error', '')}")
            return
        aabb = result["aabb"]["dimensions"]
        obb = result["obb"]["dimensions"]
        self._longest_mm = aabb[0]
        self._refresh_units()
        self._info(
            f"Triangles : {result['triangle_count']:,}\n"
            f"Vertices  : {result['vertex_count']:,}\n"
            f"AABB (axis-aligned) : {aabb[0]:.3f} × {aabb[1]:.3f} × {aabb[2]:.3f}\n"
            f"OBB (oriented)      : {obb[0]:.3f} × {obb[1]:.3f} × {obb[2]:.3f}\n"
            f"(dimensions in the mesh's own units — pick the source units below)"
        )

    def _on_convert(self, result: dict):
        self._set_busy(None)
        if not result.get("ok"):
            messagebox.showerror("Conversion failed", result.get("error", "unknown error"))
            self.status.config(text="Conversion failed.")
            return
        s = result.get("stats", {})
        cyls = s.get("cylinders", [])
        holes = sum(1 for c in cyls if c.get("role") == "hole")
        bosses = len(cyls) - holes
        msg = (
            f"Wrote {result['output']}\n\n"
            f"Method: {result['method']}\n"
            f"Faces: {s.get('faces_in')} triangles → {s.get('faces_out')} "
            f"({s.get('planar_faces', 0)} planar, {s.get('cylinder_faces', 0)} cylindrical)\n"
            f"Cylinders: {holes} hole(s), {bosses} boss(es)\n"
            f"Watertight solid: {s.get('is_solid')}"
        )
        self.status.config(text=f"Done → {Path(result['output']).name}")
        messagebox.showinfo("Conversion complete", msg)

    # ---- ui helpers -------------------------------------------------------
    def _set_busy(self, text: str | None):
        self.busy = text is not None
        self.convert_btn.config(state="disabled" if self.busy else "normal")
        if text:
            self.status.config(text=text)

    def _info(self, text: str):
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.config(state="disabled")


def main() -> int:
    root = TkinterDnD.Tk() if _DND else tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
