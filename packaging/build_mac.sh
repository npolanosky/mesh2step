#!/usr/bin/env bash
# Build the mesh2step macOS app bundle.
#
# PyInstaller cannot cross-compile, so run this ON a Mac. Produces
# dist/mesh2step.app. FreeCAD is NOT bundled — it is located at runtime; the
# machine running the app needs FreeCAD 0.20+ installed (the app auto-detects
# /Applications/FreeCAD.app, or you can point it at FreeCAD's python).
#
# Usage:
#   cd <repo root>
#   ./packaging/build_mac.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

PY="${PYTHON:-python3}"
echo "Using Python: $($PY --version)"

# Isolated build venv so we don't touch the system Python.
if [ ! -d ".venv-build" ]; then
  "$PY" -m venv .venv-build
fi
# shellcheck disable=SC1091
source .venv-build/bin/activate

python -m pip install --quiet --upgrade pip
# GUI + viewer + packaging deps. (numpy comes with pyvista; tkinter ships with
# the python.org macOS Python — if you use Homebrew python, `brew install python-tk`.)
python -m pip install --quiet -e ".[gui,viewer,build]"

echo "Building app bundle with PyInstaller..."
pyinstaller --clean --noconfirm \
  --distpath dist --workpath build/pyi \
  packaging/mesh2step.spec

echo
echo "Done -> dist/mesh2step.app"
echo "Run it:      open dist/mesh2step.app"
echo "Zip to ship: (cd dist && zip -r mesh2step-mac.zip mesh2step.app)"
echo
echo "Note: the deviation viewer needs pyvista/VTK bundled (it is). The target"
echo "Mac still needs FreeCAD installed for the conversion worker."
