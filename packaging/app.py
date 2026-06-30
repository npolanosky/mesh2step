"""Frozen-app entry point for the mesh2step GUI (used by PyInstaller)."""

import sys

from mesh2step.gui import main

if __name__ == "__main__":
    sys.exit(main())
