# Viewer & deviation heatmap (IMPLEMENTED — `mesh2step.viewer`)

A 3D viewer that shows the **input STL** and overlays the **output STEP**, with a
**deviation heatmap** colouring the STEP by its geometric distance from the mesh.
Both a user feature ("see the result quality") and a **development/QA tool** — it
shows *which* holes/faces failed or drifted.

## Usage (implemented)

```bash
pip install -e ".[viewer]"          # pyvista
mesh2step-view input.stl output.step                 # interactive window
mesh2step-view input.stl output.step --screenshot dev.png   # off-screen PNG
```

`viewer.build_scene()` returns `(stl_poly, step_poly, stats)` with per-point
deviation and `{max, rms, p95, mean}` (mm) — usable headlessly for automated QA.
On the real Blank Topper part it reports max ≈ 0.82 mm with ~2.4% of STEP points
deviating, pinpointing the imperfect regions.

Architecture: FreeCAD's Python tessellates the STEP (worker `tessellate` mode);
pyvista reads both meshes and computes point-to-surface distance
(`compute_implicit_distance`) for the heatmap.

## Original design notes

## Why it helps development

- Visualises exactly which features were reconstructed vs left faceted.
- The heatmap flags where analytic faces deviate from the mesh (bad fits,
  wrong-radius holes, mis-placed cylinders) — turns "looks almost right" into a
  measurable number per region.
- Lets us regression-test conversions visually across the sample library.

## Deviation heatmap approach (FreeCAD-native)

FreeCAD already provides the primitives, so no new heavy dependency is needed:

1. Tessellate each STEP face (`shape.tessellate(dev)`) into sample points.
2. Project those points onto the input mesh: `MeshPart.projectPointsOnMesh(points, mesh, dir)`
   or nearest-point queries; distance = deviation.
3. Colour-map distances (e.g. blue 0 → red ≥ tol) per vertex and render.
4. Report max/RMS deviation per face and overall — a hard quality metric.

## Rendering options

| Option | Pros | Cons |
|--------|------|------|
| **pyvista** (VTK) | Easy 3D, per-vertex scalars/heatmaps, mesh+brep, screenshots for automated QA | extra dependency (VTK is large) |
| **trimesh + pyglet** | light, good for meshes | weaker BREP/STEP handling |
| **three.js (web)** | shareable, no install; tessellate to glTF | more plumbing; separate stack |
| **FreeCAD GUI headless** | native STEP/mesh; offscreen render | heavier to script |

**Recommendation:** a `mesh2step-view` command using **pyvista** — load the STL
as one actor, the STEP (tessellated) as another with the deviation scalar field,
a slider for the deviation clamp, and a "save screenshot" for automated QA. It
can reuse the worker pattern (run under FreeCAD's Python for tessellation +
projection, hand vertices/scalars to pyvista).

## How it plugs in

- Reuses `worker`/`freecad_env` to tessellate + project under FreeCAD's Python.
- Adds `deviation.py` (numpy/FreeCAD): sample STEP, project to mesh, return
  per-point distances + summary stats. These stats also feed the GUI quality
  report (max/RMS deviation as a first-class number).
- Optional "View result" button in the GUI once the standalone viewer works.

## Kickoff prompt for a parallel session

Start a second Claude Code session in this repo and paste:

> Build a `mesh2step` viewer with a deviation heatmap, per docs/VIEWER.md. Add
> `src/mesh2step/deviation.py` that, under FreeCAD's Python, tessellates a STEP
> shape, projects the sample points onto the source STL mesh
> (`MeshPart.projectPointsOnMesh`), and returns per-point deviation distances +
> {max, rms, p95} summary stats (numpy). Add a `mesh2step-view` entry point
> using pyvista that shows the STL and the STEP overlaid, colouring the STEP by
> deviation (blue→red, clamp slider) and printing the summary. Test on
> tests/data/*.stl and tests/data/real/blank_topper.stl. Keep the FreeCAD parts
> isolated (only deviation.py / worker import FreeCAD) so the viewer packages
> like the GUI. Wire the deviation summary into the conversion stats so the GUI
> quality report can show max/RMS deviation.

Work it on a branch/worktree to avoid colliding with core development, then we
merge once the core hole-reconstruction work settles.
