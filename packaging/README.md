# Packaging

The GUI is packaged as a small standalone app. **FreeCAD is not bundled** — it
is located on the user's machine at runtime and the conversion runs in FreeCAD's
own Python out-of-process. The frozen app therefore only contains the tkinter
GUI plus our package source (shipped so FreeCAD's Python can import the worker),
so the build is ~2 MB rather than hundreds.

## Windows

Requires a Python (3.11+) with `pyinstaller` and `tkinterdnd2`:

```bash
pip install -e ".[gui,build]"
pyinstaller --clean --noconfirm packaging/mesh2step.spec
```

Output: `dist/mesh2step/mesh2step.exe` (a one-folder app). Zip the `mesh2step`
folder to distribute. The target machine still needs FreeCAD 0.20+ installed;
the app auto-detects it (or the user points at FreeCAD's `python.exe`).

### How it works when frozen

- `sys._MEIPASS/mesh2step_src` holds our package source (bundled via the spec's
  `src_datas`).
- The GUI launches `freecad_python -m mesh2step.worker` with `PYTHONPATH` set to
  that folder, so FreeCAD's separate interpreter imports the conversion code.
- `numpy`, `FreeCAD`, `Part`, `Mesh` are **excluded** from the frozen app — they
  belong to FreeCAD's Python, not ours.

## macOS

PyInstaller **cannot cross-compile**, so build the mac app *on a Mac*. A helper
script does the whole thing (creates an isolated build venv, installs deps,
runs PyInstaller):

```bash
cd <repo root>
./packaging/build_mac.sh          # -> dist/mesh2step.app
open dist/mesh2step.app
```

Or manually:

```bash
python3 -m pip install -e ".[gui,viewer,build]"
pyinstaller --clean --noconfirm packaging/mesh2step.spec   # spec makes a .app on macOS
```

The same spec builds both platforms — on macOS its `BUNDLE` step wraps the
one-folder app into `dist/mesh2step.app`. The target Mac still needs **FreeCAD
0.20+ installed**; the app auto-detects `/Applications/FreeCAD.app` (see the
macOS globs in `freecad_env.py`) or you can point it at FreeCAD's `python` in the
GUI. The deviation viewer's pyvista/VTK are bundled into the app.

Notes:
- **tkinter**: the python.org macOS Python includes it. With Homebrew Python run
  `brew install python-tk` first.
- Gatekeeper: an unsigned `.app` needs right-click → Open the first time (or
  `xattr -dr com.apple.quarantine dist/mesh2step.app`). Code-signing +
  notarization are optional follow-ups for distribution.
- To share it: `(cd dist && zip -r mesh2step-mac.zip mesh2step.app)`.
