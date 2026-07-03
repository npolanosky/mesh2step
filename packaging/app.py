"""Frozen-app entry point (used by PyInstaller).

Runs the GUI by default. When invoked as ``mesh2step --view STL STEP`` it opens
the pyvista deviation viewer in its own process.

This is the OUTERMOST guard: it catches even import-time failures (e.g. Tcl/Tk
or a native lib not bundling correctly) that would otherwise make a windowed
``.app`` close instantly with no message — writing them to a log file and
showing a dialog so the user isn't left blind.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path


def _log_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "mesh2step"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "mesh2step" / "logs"
    else:
        base = Path.home() / ".local" / "state" / "mesh2step"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        base = Path(tempfile.gettempdir())
    return base


def _fatal(tb: str, logpath: Path) -> None:
    try:
        with open(logpath, "a", encoding="utf-8") as fh:
            fh.write(tb + "\n")
    except Exception:
        pass
    if sys.stderr is not None:
        try:
            sys.stderr.write(tb)
        except Exception:
            pass
    try:  # last-ditch GUI dialog (may itself fail if tkinter is the problem)
        import tkinter as tk
        from tkinter import messagebox

        r = tk.Tk()
        r.withdraw()
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        messagebox.showerror("mesh2step failed to start",
                             f"{last}\n\nDetails were written to:\n{logpath}")
        r.destroy()
    except Exception:
        pass


def _disable_window_tabbing() -> None:
    """Opt this app out of macOS automatic window tabbing, before Tk starts.

    If the user's system setting is "Prefer tabs: Always", AppKit adds a native
    tab bar to *every* window — which surfaces as a stray blank tab strip above
    mesh2step's single window. The per-app ``AppleWindowTabbingMode = manual``
    default suppresses it. It must be set before the first NSWindow is created,
    so we do it here at process start (the bundled Tk is 8.6, which has no Tcl
    command to change tabbing after the fact). Best-effort and macOS-only.
    """
    if sys.platform != "darwin":
        return
    try:
        import subprocess

        subprocess.run(
            ["defaults", "write", "com.mesh2step.app",
             "AppleWindowTabbingMode", "manual"],
            check=False, capture_output=True, timeout=10,
        )
    except Exception:
        pass


def _run() -> int:
    if "--selfcheck" in sys.argv:
        # Import the whole GUI/viewer stack (the parts most likely to fail to
        # bundle) without opening a window, so the build can verify the app.
        # pyvista/PIL are imported lazily at runtime, so pull them in explicitly
        # here — otherwise a missing bundle would only surface when the user
        # first opens the 3D preview.
        import mesh2step.embedded_viewer  # noqa: F401
        import mesh2step.gui  # noqa: F401
        import mesh2step.viewer  # noqa: F401
        import pyvista  # noqa: F401
        import vtkmodules.all  # noqa: F401  (the embedded viewer needs the full VTK)
        from PIL import Image, ImageTk  # noqa: F401

        print("selfcheck ok")
        return 0
    if "--provision-check" in sys.argv:
        # Exercise the self-provisioning code paths from inside the frozen app:
        # resolve the FreeCAD download URL (dry-run, no download) and report the
        # per-user pydeps target dir. Lets the build verify provisioning ships
        # and works without opening a window or installing anything.
        from mesh2step import freecad_env, provision

        url = provision.freecad_download_url()
        print(f"freecad_download_url: {url}")
        fc = freecad_env.find_freecad_python()
        print(f"freecad_python: {fc}")
        if fc:
            print(f"pydeps_dir: {provision.pydeps_dir(fc)}")
            print(f"prep_deps_present: {provision.prep_deps_present(fc)}")
        return 0
    if "--vtkcheck" in sys.argv:
        # Confirm the in-window 3D preview stack works when frozen. On Windows the
        # embedded viewer creates a live reparented VTK window; on macOS/Linux it
        # renders preview images off-screen, so we check the path that platform
        # actually uses.
        import time

        import tkinter as tk
        from tkinter import ttk

        import mesh2step.embedded_viewer as ev
        from mesh2step.embedded_viewer import EmbeddedViewer

        root = tk.Tk()
        root.geometry("640x480")
        ttk.Style().configure("Bg.TFrame", background="#0f172a")
        ttk.Style().configure("Muted.TLabel", background="#0f172a", foreground="#94a3b8")
        viewer = EmbeddedViewer(root)
        viewer.pack(fill="both", expand=True)
        for _ in range(40):
            root.update()
        if ev._LIVE_EMBED:
            ok = viewer._vtk is not None
        else:
            # Render a sample STL (passed as an argument) to a preview image and
            # confirm it lands; with a STEP too, also exercise the deviation
            # heatmap path (tessellation + implicit distance + off-screen render).
            # With no sample, just confirm the static image label exists.
            def _wait_for(key, seconds):
                deadline = time.time() + seconds
                while time.time() < deadline and key not in viewer._images:
                    root.update()
                    time.sleep(0.05)
                return key in viewer._images

            stls = [a for a in sys.argv if a.lower().endswith(".stl")]
            steps = [a for a in sys.argv if a.lower().endswith(".step")]
            if stls:
                viewer.show_stl(stls[0])
                ok = _wait_for("stl", 120)
                print(f"  stl preview: {'ok' if ok else 'FAILED'}")
                if ok and steps:
                    viewer.show_result(stls[0], steps[0])
                    ok = _wait_for("heatmap", 300)
                    print(f"  heatmap preview: {'ok' if ok else 'FAILED'}")
                    if ok:
                        viewer.set_view("step")
                        ok = _wait_for("step", 120)
                        print(f"  step preview: {'ok' if ok else 'FAILED'}")
            else:
                ok = viewer._image_label is not None
        print("vtkcheck ok" if ok else f"vtkcheck FAILED (failed={viewer._vtk_failed})")
        root.destroy()
        return 0 if ok else 1
    if "--render-preview" in sys.argv:
        # Off-screen preview renderer for the embedded viewer: the GUI re-launches
        # this same binary to render a scene spec to a PNG in its own process
        # (VTK off-screen contexts can't be made from a Tk worker thread on
        # macOS, and a subprocess gives the GUI a hard render timeout).
        i = sys.argv.index("--render-preview")
        from mesh2step.preview_render import main as render_main

        return render_main(sys.argv[i + 1:i + 2])
    if "--view" in sys.argv:
        i = sys.argv.index("--view")
        from mesh2step.viewer import main as viewer_main

        # Pass everything after --view through so the frozen app supports the
        # same flags as the CLI (--screenshot, --clamp, --deflection, ...).
        return viewer_main(sys.argv[i + 1:])
    _disable_window_tabbing()
    from mesh2step.gui import main

    return main()


if __name__ == "__main__":
    try:
        sys.exit(_run())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - never let the app vanish without a trace
        _fatal(traceback.format_exc(), _log_dir() / "mesh2step.log")
        sys.exit(1)
