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


def _run() -> int:
    if "--selfcheck" in sys.argv:
        # Import the whole GUI/viewer stack (the parts most likely to fail to
        # bundle) without opening a window, so the build can verify the app.
        import mesh2step.gui  # noqa: F401
        import mesh2step.viewer  # noqa: F401

        print("selfcheck ok")
        return 0
    if "--view" in sys.argv:
        i = sys.argv.index("--view")
        from mesh2step.viewer import view

        view(sys.argv[i + 1], sys.argv[i + 2])
        return 0
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
