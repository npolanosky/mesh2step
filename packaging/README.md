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

## macOS (deferred)

Planned once the Windows build is confirmed in real use. The same architecture
applies (locate FreeCAD's bundled Python under `FreeCAD.app/Contents/Resources`,
shell out to the worker). Build with PyInstaller on a Mac:

```bash
pip install -e ".[gui,build]"
pyinstaller --clean --noconfirm --windowed packaging/mesh2step.spec
```

`freecad_env.py` already contains macOS search paths for both the library dir
and the bundled Python. A `.app` bundle / DMG and notarization are the remaining
steps. PyInstaller cannot cross-compile, so the Mac build must run on macOS.
