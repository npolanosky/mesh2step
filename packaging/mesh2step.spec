# PyInstaller spec for the mesh2step GUI.
#
# Build (from the repo root, with a Python that has pyinstaller + tkinterdnd2):
#     pyinstaller packaging/mesh2step.spec
#
# Produces dist/mesh2step/mesh2step.exe (a one-folder app). FreeCAD is NOT
# bundled — it is located on the user's machine at runtime. Our own package
# source is bundled as data ("mesh2step_src/") so that FreeCAD's separate Python
# can import the conversion worker out-of-process.

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH is injected by PyInstaller = the directory containing this spec.
REPO = os.path.dirname(SPECPATH)
SRC = os.path.join(REPO, "src")

# tkinterdnd2 ships native tkdnd libraries that must be collected explicitly.
dnd_datas, dnd_bins, dnd_hidden = collect_all("tkinterdnd2")

# The deviation viewer needs pyvista + VTK (large). Collect everything they need.
pv_datas, pv_bins, pv_hidden = collect_all("pyvista")
vtk_datas, vtk_bins, vtk_hidden = collect_all("vtkmodules")
vtk_hidden += collect_submodules("vtkmodules")

# Ship our package as plain source so FreeCAD's interpreter can import it.
src_datas = [(os.path.join(SRC, "mesh2step"), os.path.join("mesh2step_src", "mesh2step"))]

a = Analysis(
    [os.path.join(SPECPATH, "app.py")],
    pathex=[SRC],
    binaries=dnd_bins + pv_bins + vtk_bins,
    datas=dnd_datas + src_datas + pv_datas + vtk_datas,
    hiddenimports=dnd_hidden + pv_hidden + vtk_hidden,
    hookspath=[],
    # FreeCAD/Part/Mesh belong to FreeCAD's Python; numpy is kept because the
    # viewer/pyvista run in this process.
    excludes=["FreeCAD", "Part", "Mesh"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="mesh2step",
    console=False,
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    name="mesh2step",
)
