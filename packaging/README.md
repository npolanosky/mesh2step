# Packaging

The GUI is packaged as a standalone app. **FreeCAD is not bundled** — it is
located on the user's machine at runtime and the conversion runs in FreeCAD's
own Python out-of-process. The frozen app contains the tkinter GUI, our package
source (shipped so FreeCAD's Python can import the worker), and the pyvista/VTK
stack for the embedded 3D deviation viewer.

## Self-provisioning (no manual installs for the end user)

The app provisions its own runtime dependencies on first use, so a fresh machine
needs nothing hand-installed:

- **FreeCAD** — auto-detected via the globs in `freecad_env.py`. If it is not
  found, the GUI shows a consent dialog and (on approval) downloads the official
  FreeCAD macOS **arm64** release from the FreeCAD GitHub *latest release* and
  unpacks it into `~/Applications` — no administrator password needed. On
  failure it shows a clear manual-install message. See
  `provision.freecad_download_url` / `provision.install_freecad`.
- **Prep deps (`manifold3d`, `pymeshlab`)** — installed on first conversion into
  a per-user, per-(python-version+arch) cache
  `~/Library/Application Support/mesh2step/pydeps/py<X.Y>-<arch>/` using
  **FreeCAD's own `pip --target`**. FreeCAD.app itself is never modified (that
  would break its code signature). The worker subprocess (and the headless CLI)
  pick the cache up via `PYTHONPATH` / `sys.path`. See `provision.py`.
  - `manifold3d` is the load-bearing dep (resolves self-intersecting /
    overlapping-body meshes into one watertight solid); it is required.
  - `pymeshlab` is optional (planar decimation only). Its wheels bundle a Qt
    whose plugin loader hard-crashes (SIGTRAP) on import on some macOS builds —
    uncatchable in-process — so `meshprep._pymeshlab_importable()` probes it in
    a throwaway subprocess and simply **skips decimation** if it would crash.
    A conversion never fails because of pymeshlab.

Verify the provisioning paths from a built bundle without opening a window:

```bash
BIN="dist/mesh2step.app/Contents/MacOS/mesh2step"
"$BIN" --selfcheck          # imports the GUI + pyvista/VTK viewer stack
"$BIN" --provision-check    # resolves the FreeCAD download URL + pydeps dir
```

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
one-folder app into `dist/mesh2step.app`. The deviation viewer's pyvista/VTK are
bundled into the app. The target Mac needs **nothing pre-installed**: FreeCAD is
auto-detected and, if absent, offered for one-click download/install into
`~/Applications`; the prep deps are auto-installed on first conversion (see
**Self-provisioning** above). You can still point the GUI at an existing
FreeCAD `python` manually if you prefer.

## Running FreeCAD's Python as a worker

The GUI drives FreeCAD's bundled interpreter as `<freecad_python> -m
mesh2step.worker`. FreeCAD only wires up `import FreeCAD` automatically when *it*
launches the interpreter (freecadcmd / its GUI) — a plain `python -m` does not
get it. So `provision.prep_env()` puts FreeCAD's own library directory on
`PYTHONPATH` (resolved from the interpreter path by
`freecad_env.freecad_lib_dir()`: macOS `.../Resources/lib`, Windows the `bin`
dir). Without this the worker fails with `No module named 'FreeCAD'`, which
surfaces as "Mesh health: (unavailable)" and viewer/tessellation failures. The
same env also carries the provisioned pydeps dir.

## 3D preview (embedded + pop-out)

- **Windows** embeds a live, GPU VTK render window directly in the Tk panel
  (reparented via `SetParentInfo`), with trackball interaction.
- **macOS / Linux** cannot reparent an NSView/VTK window into Tk, so the embedded
  panel renders each view (STL / STEP / deviation heatmap) **off-screen to an
  image** and shows it in the panel (switchable, re-rendered on resize). The
  render runs in a short-lived **`--render-preview` subprocess** (see
  `preview_render.py`): creating a VTK off-screen context from a Tk worker
  thread deadlocks on macOS, and the subprocess gives the GUI a hard timeout —
  a stuck render becomes an error message in the panel, never a hang. Live
  camera interaction is one click away via **Pop out ↗**, which opens the same
  scene in its own interactive VTK window (frozen: the app re-launches itself
  as `mesh2step --view STL STEP`; see `packaging/app.py`).

`"$BIN" --vtkcheck sample.stl [result.step]` exercises whichever path the build
platform uses (with a STEP it also verifies the deviation-heatmap preview).

## Failure corpus (regression testing)

With **"Save failing models for regression testing"** enabled (GUI checkbox,
persisted; CLI `--save-failures[=DIR]`), any conversion that doesn't end in a
single watertight solid — including worker crashes — copies the input STL into
`tests/data/community/failures/<category>/` (source checkout) or the per-user
support dir `failed_corpus/` (frozen app), deduped by sha256 and indexed in a
`manifest.json` that also records later passes on saved files. **Flag for
improvement** (button, enabled after a watertight result) files the input under
`faceted_improvable/` with the surface stats from the quality report. The
category taxonomy lives in one place, `failstore.py`; regression sweeps should
enumerate the corpus via `failstore.iter_corpus()` so `failures/**` is included.

Notes:
- **tkinter**: the python.org macOS Python includes it. With Homebrew Python run
  `brew install python-tk` first.
- **Drag-and-drop** is disabled in the frozen macOS app: the bundled `tkdnd`
  native library fails to load under the app's Tcl/Tk ("incompatible stubs
  mechanism"), so the GUI falls back to a plain window and the **Browse…** button.
  This is handled, not a crash. Gotcha for future work: the failed
  `TkinterDnD.Tk()` fully initialises a plain Tk first, leaving a half-built
  interp registered as `tkinter._default_root`; unless that zombie is destroyed
  (see `gui._destroy_zombie_default_root`), every Variable created without an
  explicit master binds to it and the UI shows blank fields and indeterminate
  checkboxes.
- **Window tabbing**: if the user's system setting is "Prefer tabs: Always",
  AppKit would add a blank tab strip above the single window. The app writes
  `AppleWindowTabbingMode = manual` to its own defaults domain at startup (before
  any window is created) to suppress it.
- Gatekeeper: an unsigned `.app` needs right-click → Open the first time (or
  `xattr -dr com.apple.quarantine dist/mesh2step.app`). Code-signing +
  notarization are optional follow-ups for distribution.
- To share it: `(cd dist && zip -r mesh2step-mac.zip mesh2step.app)`.
