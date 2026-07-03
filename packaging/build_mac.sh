#!/usr/bin/env bash
# Build the mesh2step macOS app bundle -> dist/mesh2step.app
#
# PyInstaller cannot cross-compile, so run this ON a Mac. The resulting .app is
# fully standalone: it bundles Python, Tcl/Tk, pyvista/VTK and everything else,
# so the machine that RUNS it needs nothing but macOS + FreeCAD installed. This
# BUILD machine only needs a Python that has working tkinter.
#
# Usage:
#   cd <repo root>
#   ./packaging/build_mac.sh
#
# If ./build_mac.sh won't run: `chmod +x packaging/build_mac.sh` (and, if it was
# edited on Windows, `sed -i '' $'s/\r$//' packaging/build_mac.sh`).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# --- pick a Python that has tkinter (no Homebrew required) -------------------
pick_python() {
  # Prefer an explicit $PYTHON, then python.org framework builds (they bundle
  # Tcl/Tk), then whatever python3 is on PATH.
  local candidates=()
  [ -n "${PYTHON:-}" ] && candidates+=("$PYTHON")
  candidates+=(
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
    python3
  )
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import tkinter" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  cat >&2 <<'EOF'
ERROR: could not find a Python 3 with working tkinter.

The app bundles tkinter for its users, but the BUILD needs a Python that has it.
macOS no longer ships a usable system python3, so the simplest no-Homebrew fix:

  1. Download the macOS installer from https://www.python.org/downloads/macos/
     (the python.org build includes Tcl/Tk — nothing else to install).
  2. Re-run this script, or set PYTHON explicitly, e.g.:
       PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
         ./packaging/build_mac.sh

(If you prefer Homebrew instead: `brew install python-tk@3.12`.)
EOF
  exit 1
fi
echo "Using Python: $PY ($("$PY" --version 2>&1))"

# --- isolated build venv (don't touch the system Python) --------------------
if [ ! -d ".venv-build" ]; then
  "$PY" -m venv .venv-build
fi
# shellcheck disable=SC1091
source .venv-build/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[gui,viewer,build]"

# tkinter must survive into the venv (it does for framework/py builds).
python -c "import tkinter" || {
  echo "ERROR: tkinter not importable in the build venv." >&2; exit 1; }

echo "Building app bundle with PyInstaller..."
pyinstaller --clean --noconfirm \
  --distpath dist --workpath build/pyi \
  packaging/mesh2step.spec

APP="dist/mesh2step.app"
if [ ! -d "$APP" ]; then
  echo "ERROR: build finished but $APP was not produced." >&2; exit 1
fi

# Smoke-check the bundled binary launches far enough to import its GUI stack.
BIN="$APP/Contents/MacOS/mesh2step"
echo "Verifying the bundle can import its GUI stack..."
if "$BIN" --selfcheck >/dev/null 2>&1; then
  echo "  self-check OK"
else
  echo "  (self-check unavailable or failed — not fatal; run the app to confirm)"
fi

cat <<EOF

Done -> $APP
Run it:        open "$APP"
Debug a crash: "$BIN"          # runs with console output attached
Crash log:     ~/Library/Logs/mesh2step/mesh2step.log
Zip to share:  (cd dist && zip -r mesh2step-mac.zip mesh2step.app)

The app self-provisions on the target Mac — no manual dependency installs:
  * FreeCAD: auto-detected; if missing, the GUI offers to download + install the
    official macOS build into ~/Applications (no admin needed).
  * Prep deps (manifold3d, pymeshlab): auto-installed on first conversion into
    ~/Library/Application Support/mesh2step/pydeps/ using FreeCAD's own pip
    (FreeCAD.app is never modified), and injected onto the worker's PYTHONPATH.
EOF
