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
from .embedded_viewer import EmbeddedViewer
from .freecad_env import find_freecad_python

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND = True
except Exception:  # noqa: BLE001
    _DND = False

# Whether drag-and-drop is actually usable on the live root (the tkinterdnd2
# import can succeed while the native tkdnd library fails to load at runtime,
# especially in a packaged macOS app — set once the root is created).
DND_ACTIVE = False

UNIT_CHOICES = ["mm", "cm", "m", "in"]


def _log_dir() -> Path:
    """Per-user, writable log directory (so a windowed app still leaves a log)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "mesh2step"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "mesh2step" / "logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "mesh2step"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        base = Path(tempfile.gettempdir())
    return base


def _make_root() -> "tk.Tk":
    """Create the Tk root, using drag-and-drop if it actually loads, else plain.

    tkinterdnd2's ``Tk()`` loads a native library; if that fails (common in a
    frozen macOS app) we must not let it take the whole app down — fall back to
    a normal Tk window with drag-and-drop disabled.
    """
    global DND_ACTIVE
    root: "tk.Tk"
    if _DND:
        try:
            root = TkinterDnD.Tk()
            DND_ACTIVE = True
            _disable_mac_window_tabs(root)
            return root
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("mesh2step").warning(
                "drag-and-drop backend failed to load; using a plain window", exc_info=True)
            _destroy_zombie_default_root()
    root = tk.Tk()
    _disable_mac_window_tabs(root)
    return root


def _destroy_zombie_default_root() -> None:
    """Tear down the half-built Tk left behind by a failed TkinterDnD.Tk().

    TkinterDnD.Tk() runs the full ``tkinter.Tk.__init__`` (which registers the
    new interp as ``tkinter._default_root``) and only THEN tries to load the
    native tkdnd library. When that load fails (always, in the frozen macOS
    app), the exception leaves the abandoned interp as the default root. Every
    ``StringVar``/``BooleanVar`` created afterwards without an explicit master
    silently binds to that ZOMBIE interp instead of the real window — so
    checkboxes render as indeterminate dashes and entry fields (FreeCAD path,
    output, units) look blank even though the values were set. Destroying the
    zombie makes the real root the default so everything binds where it shows.
    """
    zombie = getattr(tk, "_default_root", None)
    if zombie is None:
        return
    try:
        zombie.destroy()
    except Exception:  # noqa: BLE001
        pass
    try:
        tk._default_root = None  # destroy() usually clears it; make sure
    except Exception:  # noqa: BLE001
        pass


def _disable_mac_window_tabs(root: "tk.Tk") -> None:
    """Stop macOS from adding an empty window-tab bar above our single window.

    On macOS the window manager can group Tk windows into native tabs, which
    surfaces as a stray blank "( )" tab strip at the top of the app's only
    window. Setting the NSWindow tabbing mode to *disallowed* removes it. Wrapped
    in try/except: the Tcl command only exists on macOS Aqua Tk and is a no-op
    (or absent) elsewhere.
    """
    if sys.platform != "darwin":
        return
    # Tk 8.7+ exposes a per-window tabbing mode; "disallowed" removes the native
    # tab bar. Tk 8.6 (what we bundle) has no such subcommand — the effective
    # opt-out there is the AppleWindowTabbingMode user default, set once before
    # any window is created (see packaging/app.py). This call is a harmless no-op
    # on 8.6 and a belt-and-braces removal on 8.7+.
    try:
        root.tk.eval("tk::unsupported::MacWindowStyle tabbingMode . disallowed")
    except tk.TclError:
        pass

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
    from . import provision

    with tempfile.TemporaryDirectory() as tmp:
        job_file = Path(tmp) / "job.json"
        res_file = Path(tmp) / "result.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")

        # Prepend our package source AND the auto-provisioned prep-deps dir
        # (pymeshlab/manifold3d) so FreeCAD's interpreter can import both.
        env = provision.prep_env(freecad_python)
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
        # Cap the initial height to the screen (leaving room for the taskbar) so
        # the window — and its scrollbar — are never taller than the display.
        # The body scrolls, so all controls stay reachable at any size/DPI.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        init_h = min(1080, max(560, screen_h - 90))
        init_w = min(1300, max(760, screen_w - 120))
        root.geometry(f"{init_w}x{init_h}")
        root.minsize(620, 420)
        root.configure(bg=BG)
        self.q: queue.Queue = queue.Queue()
        self.busy = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.units_var = tk.StringVar(value="mm")
        self.detect_var = tk.BooleanVar(value=True)
        self.faceted_var = tk.BooleanVar(value=False)
        self.repair_var = tk.BooleanVar(value=True)
        self.closed_var = tk.BooleanVar(value=True)
        self.freecad_var = tk.StringVar(value=find_freecad_python() or "")
        self.output_dir = None  # user-chosen output folder; None = use input's folder
        # Failure-corpus option (persisted across runs in the app settings).
        from . import provision
        self._settings = provision.load_settings()
        self.savefail_var = tk.BooleanVar(
            value=bool(self._settings.get("save_failures", False)))
        self.failures_dir = self._settings.get("failures_dir") or None
        self._longest_mm = None
        self._t0 = 0.0
        self._last_line_t = 0.0
        self._stall_noted = False
        self.last_stl = None
        self.last_step = None
        self._last_result = None

        self._prep_ready = False   # prep deps provisioned this session?

        self._init_style()
        self._build()
        self.root.after(80, self._drain_queue)
        if not self.freecad_var.get():
            self._log("⚠  FreeCAD not found.", "err")
            # Offer to auto-download/install it (no admin needed). Do it after the
            # window is up so the consent dialog has a parent.
            self.root.after(300, self._offer_freecad_install)
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

    @staticmethod
    def _checkbutton(parent, text: str, variable: "tk.BooleanVar") -> ttk.Checkbutton:
        """A ttk.Checkbutton that shows its real on/off state, never a dash.

        A fresh ttk.Checkbutton starts in the tri-state 'alternate' look (renders
        as '-' on the macOS/clam themes) until the user clicks it — even though
        its BooleanVar already holds True/False. Clearing 'alternate' and pushing
        the variable's value through makes the box read its intended default the
        moment the window opens.
        """
        cb = ttk.Checkbutton(parent, text=text, variable=variable,
                             onvalue=True, offvalue=False)
        cb.state(["!alternate"])
        variable.set(variable.get())  # force the check/uncheck to match the var
        return cb

    # ---- layout -----------------------------------------------------------
    def _build(self):
        # Header band
        header = tk.Frame(self.root, bg=ACCENT)
        header.pack(fill="x")
        from . import DISPLAY_VERSION
        tk.Label(header, text=f"mesh2step  {DISPLAY_VERSION}", bg=ACCENT, fg="white",
                 font=("Segoe UI Semibold", 18)).pack(anchor="w", padx=18, pady=(12, 0))
        tk.Label(header, text="STL mesh → STEP solid  ·  surface & hole reconstruction",
                 bg=ACCENT, fg="#dbeafe", font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(0, 12))

        # Split the window: scrollable controls on the left, the live 3D preview
        # on the right (which expands as the window grows — the responsive part).
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True)

        # Scrollable body: the frozen app is DPI-aware, so on a scaled Windows
        # display the content is taller than the physical window and the lower
        # controls (Convert, result actions, log) would clip off-screen with no
        # way to reach them. A canvas + scrollbar keeps everything reachable.
        scroll_area = ttk.Frame(main_pane, style="Bg.TFrame")
        canvas = tk.Canvas(scroll_area, bg=BG, highlightthickness=0, bd=0, width=560)
        vbar = ttk.Scrollbar(scroll_area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        if sys.platform == "darwin":
            # Pixel-ish scrolling. With no yscrollincrement, one scroll "unit" is
            # a tenth of the canvas height — and an aqua trackpad swipe delivers a
            # stream of delta events, so one swipe slammed the panel from top to
            # bottom. 3 px per unit × the per-event delta gives the standard
            # macOS feel (several swipes to traverse the panel).
            canvas.configure(yscrollincrement=3)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = ttk.Frame(canvas, style="Bg.TFrame", padding=16)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")
        # Inner frame tracks the canvas width; scrollregion tracks its height.
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(body_id, width=e.width))
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _on_wheel(e):
            # Leave the log's own text area free to scroll itself, and don't
            # steal wheel events from the preview panel on the right.
            w = self.root.winfo_containing(e.x_root, e.y_root)
            if w is None or isinstance(w, tk.Text):
                return
            area = str(scroll_area)
            if not (str(w) == area or str(w).startswith(area + ".")):
                return
            if sys.platform == "darwin":
                # Aqua Tk reports small per-line deltas (±1..±10); the Windows
                # convention divides by 120, which floors to 0 here and made
                # trackpad scrolling a no-op on macOS.
                step = -e.delta
            else:
                step = int(-e.delta / 120)
            if step:
                canvas.yview_scroll(int(step), "units")

        canvas.bind_all("<MouseWheel>", _on_wheel)

        # --- Input card ---
        c1 = self._card(body, "1  ·  Input mesh")
        self.drop = tk.Label(
            c1, text="Drag an STL here" + ("" if DND_ACTIVE else "  (or use Browse)"),
            bg="#f8fafc", fg=MUTED, font=("Segoe UI", 10),
            relief="flat", height=3, bd=1, highlightbackground=BORDER, highlightthickness=1,
        )
        self.drop.pack(fill="x")
        if DND_ACTIVE:
            try:
                self.drop.drop_target_register(DND_FILES)
                self.drop.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:  # noqa: BLE001 - drag-drop is a convenience, not required
                self.drop.config(text="Drag an STL here  (or use Browse)")
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
        self._checkbutton(c2, "Detect cylindrical holes / bosses (best-fit radius)",
                          self.detect_var).pack(anchor="w", pady=(8, 0))
        self._checkbutton(c2, "Repair mesh (fix self-intersections, duplicates, normals) — "
                          "recovers holes on defective meshes",
                          self.repair_var).pack(anchor="w")
        self._checkbutton(c2, "Fully closed (guarantee watertight; slower, faceted holes on "
                          "organic parts)",
                          self.closed_var).pack(anchor="w")
        self._checkbutton(c2, "Faceted only (skip reconstruction)",
                          self.faceted_var).pack(anchor="w")
        sf = self._checkbutton(c2, "Save failing models for regression testing",
                               self.savefail_var)
        sf.config(command=self._on_savefail_toggle)
        sf.pack(anchor="w", pady=(8, 0))
        sfrow = ttk.Frame(c2, style="Card.TFrame")
        sfrow.pack(fill="x")
        self.faildest_label = ttk.Label(sfrow, text="→ " + str(self._failures_dest()),
                                        style="Muted.TLabel")
        self.faildest_label.pack(side="left", padx=(24, 0))
        ttk.Button(sfrow, text="…", width=3,
                   command=self._choose_failures_dir).pack(side="left", padx=(6, 0))

        # --- Output ---
        c3 = self._card(body, "3  ·  Output")
        orow = ttk.Frame(c3, style="Card.TFrame"); orow.pack(fill="x")
        ttk.Entry(orow, textvariable=self.output_var, state="readonly").pack(
            side="left", fill="x", expand=True)
        ttk.Button(orow, text="Choose folder…", command=self._browse_output).pack(
            side="left", padx=(6, 0))
        ttk.Label(c3, text="File name follows the input; pick a folder to change where it lands.",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
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
        self.view_btn = ttk.Button(arow, text="Pop out 3D view ↗", state="disabled",
                                   command=self._view_result)
        self.view_btn.pack(side="left")
        self.flag_btn = ttk.Button(arow, text="Flag for improvement", state="disabled",
                                   command=self._flag_result)
        self.flag_btn.pack(side="left", padx=6)
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

        # Assemble the split: controls (fixed-ish) + preview (grows).
        self.viewer = EmbeddedViewer(
            main_pane,
            freecad_python_getter=lambda: (self.freecad_var.get().strip() or None),
            on_popout=self._view_result)
        main_pane.add(scroll_area, weight=0)
        main_pane.add(self.viewer, weight=1)

    # ---- actions ----------------------------------------------------------
    def _browse_input(self):
        p = filedialog.askopenfilename(filetypes=[("STL mesh", "*.stl"), ("All", "*.*")])
        if p:
            self._set_input(p)

    def _browse_output(self):
        # Pick a destination FOLDER; the file name is derived from the input.
        start = self.output_dir or (str(Path(self.input_var.get()).parent)
                                    if self.input_var.get().strip() else None)
        d = filedialog.askdirectory(title="Choose output folder", initialdir=start or None)
        if d:
            self.output_dir = d
            self._update_output_path()

    def _browse_freecad(self):
        p = filedialog.askopenfilename(title="FreeCAD's python executable")
        if p:
            self.freecad_var.set(p)
            self._log(f"FreeCAD: {p}", "muted")

    # ---- failure corpus ----------------------------------------------------
    def _failures_dest(self):
        from . import failstore

        return failstore.resolve_dest(self.failures_dir)

    def _on_savefail_toggle(self):
        self._persist_settings()
        if self.savefail_var.get():
            self._log(f"Failing inputs will be copied to {self._failures_dest()} "
                      f"(sorted by failure category).", "muted")

    def _choose_failures_dir(self):
        d = filedialog.askdirectory(title="Failure corpus folder",
                                    initialdir=str(self._failures_dest()))
        if d:
            self.failures_dir = d
            self.faildest_label.config(text="→ " + d)
            self._persist_settings()
            self._log(f"Failure corpus folder: {d}", "muted")

    def _persist_settings(self):
        from . import provision

        self._settings["save_failures"] = bool(self.savefail_var.get())
        self._settings["failures_dir"] = self.failures_dir
        provision.save_settings(self._settings)

    def _on_drop(self, event):
        self._set_input(event.data.strip().strip("{}"))

    @staticmethod
    def _validate_stl(path: str) -> str | None:
        """Return an error message if ``path`` is not a usable STL, else None."""
        p = Path(path)
        if not p.is_file():
            return "File not found."
        if p.suffix.lower() != ".stl":
            got = p.suffix or "no extension"
            return f"Not an STL file ({got}). Please choose a .stl file."
        try:
            size = p.stat().st_size
            if size < 84:
                return "File is too small to be a valid STL."
            with open(p, "rb") as fh:
                head = fh.read(80)
                n = int.from_bytes(fh.read(4), "little")
            if head[:5].lower() == b"solid":          # ASCII STL
                return None
            if size == 84 + 50 * n:                    # binary STL
                return None
            return "File does not look like a valid STL (bad header/size)."
        except Exception:  # noqa: BLE001 - sniff is best-effort; don't block on IO quirks
            return None

    def _update_output_path(self):
        """Derive the output .step path from the current input + chosen folder."""
        inp = self.input_var.get().strip()
        if not inp:
            return
        folder = self.output_dir or str(Path(inp).parent)
        self.output_var.set(str(Path(folder) / (Path(inp).stem + ".step")))

    def _set_input(self, path: str):
        path = path.strip().strip('"')
        err = self._validate_stl(path)
        if err:
            self._log(f"✖  {err}", "err")
            self.status.config(text=err, fg=ERR_RED)
            return
        self.input_var.set(path)
        self.drop.config(text=Path(path).name, fg=TEXT)
        # Always re-derive the output path from the new input (into the chosen
        # output folder if one was set), so converting a second file doesn't
        # keep writing to the first file's name.
        self._update_output_path()
        # Show the mesh in the preview immediately — it needs no FreeCAD, so it
        # must not wait for the inspect worker (which can take seconds). If the
        # inspection finds defects, _on_inspect re-shows it with the markers.
        self.viewer.show_stl(path)
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

    # ---- self-provisioning ------------------------------------------------
    def _offer_freecad_install(self):
        """If FreeCAD is missing, ask consent and auto-download it (no admin)."""
        from tkinter import messagebox

        if self.freecad_var.get().strip() or self.busy:
            return
        ok = messagebox.askyesno(
            "FreeCAD not found",
            "mesh2step needs FreeCAD to convert meshes, but no install was found.\n\n"
            "Download the official FreeCAD (macOS) now and install it into your "
            "~/Applications folder? No administrator password is required.\n\n"
            "(You can also click No and point at an existing FreeCAD python "
            "manually in the Output section.)",
            parent=self.root,
        )
        if not ok:
            self._log("FreeCAD auto-install declined — set its python path below, "
                      "or install from https://www.freecad.org/downloads.php", "muted")
            return
        self._start("Downloading & installing FreeCAD…")

        def work():
            from . import freecad_env, provision

            app = provision.install_freecad(log=lambda m: self.q.put(("log", m)))
            if app:
                py = freecad_env.find_freecad_python()
                self.q.put(("freecad_installed", py or ""))
            else:
                self.q.put(("freecad_installed", ""))

        threading.Thread(target=work, daemon=True).start()

    def _on_freecad_installed(self, py: str):
        self._stop()
        if py and Path(py).is_file():
            self.freecad_var.set(py)
            self._log(f"✔  FreeCAD ready: {py}", "ok")
            self.status.config(text="FreeCAD installed — choose an STL to convert.")
        else:
            self._log("⚠  Could not auto-install FreeCAD. Install it manually from "
                      "https://www.freecad.org/downloads.php, then set its python "
                      "path below.", "err")
            self.status.config(text="FreeCAD not available.", fg=ERR_RED)

    def _ensure_prep_deps(self, fc: str) -> bool:
        """Provision pymeshlab/manifold3d into the user pydeps dir if needed.

        Runs synchronously on the worker thread (callers are already off the UI
        thread). Streams pip output into the log. Best-effort: returns True when
        the deps import, False otherwise — conversion still proceeds either way
        (the pipeline degrades gracefully without them).
        """
        from . import provision

        if self._prep_ready:
            return True
        try:
            target = provision.ensure_prep_deps(
                fc, log=lambda m: self.q.put(("log", m)))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("log", f"⚠ prep-dep provisioning error: {exc}"))
            target = None
        self._prep_ready = target is not None
        return self._prep_ready

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
                # A conversion needs the prep deps (pymeshlab/manifold3d) for the
                # watertight path; provision them once, on first use, so the user
                # never has to pip-install anything. Inspect doesn't need them.
                if kind == "convert":
                    self._ensure_prep_deps(fc)
                result = run_worker(job, fc, on_line=lambda ln: self.q.put(("log", ln)))
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": str(exc)}
            if kind == "convert" and self.savefail_var.get():
                # Keep inputs that fail to convert watertight (and record later
                # passes on files already in the corpus). Off the UI thread —
                # the copy/hash can take a moment on big meshes.
                from . import failstore

                failstore.record_result(job["input"], result, dest=self.failures_dir,
                                        log=lambda m: self.q.put(("log", m)))
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
                elif kind == "freecad_installed":
                    self._on_freecad_installed(payload)
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

        # The preview already shows the mesh (kicked off at selection); only
        # re-render it when there are defect regions to mark.
        if result.get("problem_points"):
            self.viewer.show_stl(self.input_var.get().strip(),
                                 problem_points=result.get("problem_points"))

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

        # Enable the deviation viewer for this result. Point it at a file that
        # was actually written: in dual-output mode the base path is not written
        # (only the suffixed _watertight/_clean files are), so prefer an existing
        # output over result["output"].
        self.last_stl = self.input_var.get().strip()
        written = [p for p in outputs if Path(p).exists()]
        self.last_step = written[0] if written else result["output"]
        self.view_btn.config(state="normal" if written else "disabled")
        # "Flag for improvement" applies to a finished watertight result whose
        # remaining faceted surfaces the user wants a future version to rebuild.
        self._last_result = result
        self.flag_btn.config(
            state="normal" if (written and s.get("is_solid")) else "disabled")

        # Show the result (STEP + deviation heatmap) in the embedded preview.
        if written and self.last_stl:
            self.viewer.show_result(self.last_stl, self.last_step)

    # ---- result actions ---------------------------------------------------
    def _flag_result(self):
        """Copy the input STL into the failure corpus as user-flagged
        (watertight, but with faceted surfaces worth improving)."""
        stl = self.last_stl
        if not (stl and Path(stl).is_file()):
            self._log("⚠  No input to flag.", "err")
            return
        self.flag_btn.config(state="disabled")  # one flag per result
        result = getattr(self, "_last_result", None)

        def work():
            from . import failstore

            failstore.record_flag(stl, result, dest=self.failures_dir,
                                  log=lambda m: self.q.put(("log", m)))

        threading.Thread(target=work, daemon=True).start()

    def _view_result(self):
        if self.last_stl and self.last_step and Path(self.last_step).exists():
            self._log("Opening 3D viewer (1: STL · 2: STEP · 3: heatmap)…", "muted")
            # Open on whatever view the embedded panel is showing.
            self._launch_viewer(self.last_stl, self.last_step,
                                self.viewer._active or "heatmap")
            return
        # Pre-conversion: pop out just the input mesh (no FreeCAD needed).
        stl = self.input_var.get().strip()
        if stl and Path(stl).is_file():
            self._log("Opening 3D viewer (input STL — convert to add STEP/heatmap)…",
                      "muted")
            self._launch_viewer(stl, None, "stl")
        else:
            self._log("⚠  Choose an STL first.", "err")

    def _launch_viewer(self, stl: str, step: str | None, show: str = "heatmap"):
        """Launch the pyvista viewer as its own process."""
        files = [stl] + ([step] if step else [])
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--view", *files, "--show", show]
            env = None
        else:
            env = dict(os.environ)
            env["PYTHONPATH"] = _package_src() + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", "mesh2step.viewer", *files, "--show", show]
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


def _setup_logging():
    """Log to a per-user file so a windowed (no-console) crash is still findable."""
    import logging

    logpath = _log_dir() / "mesh2step.log"
    handlers = [logging.FileHandler(str(logpath), encoding="utf-8")]
    if sys.stderr is not None:  # None under a frozen windowed app
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logpath


def _show_fatal_dialog(message: str, logpath) -> None:
    """Best-effort error dialog; safe to call even if the GUI is broken."""
    try:
        import tkinter as _tk
        from tkinter import messagebox

        r = _tk.Tk()
        r.withdraw()
        messagebox.showerror(
            "mesh2step failed to start",
            f"{message}\n\nDetails were written to:\n{logpath}")
        r.destroy()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    import logging
    import traceback

    logpath = _setup_logging()
    log = logging.getLogger("mesh2step")
    log.info("starting (frozen=%s, platform=%s)", getattr(sys, "frozen", False), sys.platform)
    try:
        root = _make_root()
        App(root)
        log.info("GUI up (drag-and-drop=%s)", DND_ACTIVE)
        root.mainloop()
        return 0
    except Exception:  # noqa: BLE001 - top-level guard: log + surface, never silent-crash
        tb = traceback.format_exc()
        log.error("fatal error\n%s", tb)
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        _show_fatal_dialog(last, logpath)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
