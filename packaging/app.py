"""Frozen-app entry point (used by PyInstaller).

Runs the GUI by default. When invoked as ``mesh2step.exe --view STL STEP`` the
GUI shells out to this same executable to open the pyvista deviation viewer in
its own process (so VTK gets its own main loop).
"""

import sys

if __name__ == "__main__":
    if "--view" in sys.argv:
        i = sys.argv.index("--view")
        from mesh2step.viewer import view

        view(sys.argv[i + 1], sys.argv[i + 2])
        sys.exit(0)
    from mesh2step.gui import main

    sys.exit(main())
