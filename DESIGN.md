# Design & Architecture

## Goal

Turn a triangle-mesh STL into a STEP **solid** whose topology reflects the
*surfaces* of the part, not its tessellation. The headline differentiator vs.
prior art is surface reconstruction: coplanar facets become single planar
faces, and (eventually) curved regions become analytic surfaces.

## Constraints that shape the design

- **FreeCAD's Python is a separate interpreter.** `import FreeCAD` only works
  under FreeCAD's bundled Python *or* if FreeCAD's `bin/` is on `sys.path`. To
  keep the project testable on any machine, all geometry analysis that does
  *not* require OpenCASCADE is written in plain numpy and isolated from the
  FreeCAD-dependent code.
- **STL has no topology.** Vertices are duplicated per triangle. We must weld
  coincident vertices before we can build adjacency.
- **STL is faceted and noisy.** "Coplanar" must be tolerance-based, not exact.

## Module layout

```
src/mesh2step/
├── config.py        ConversionConfig dataclass — all tolerances/flags
├── mesh_io.py       STL load + vertex welding -> (vertices, faces) numpy arrays
├── segmentation.py  planar region growing over the welded mesh  [pure numpy]
├── boundary.py      boundary-loop extraction + collinear simplification [numpy]
├── fitting.py       analytic surface fitting (plane/cylinder/...) [roadmap]
├── builder.py       regions -> FreeCAD faces -> sewn solid -> STEP  [FreeCAD]
├── pipeline.py      orchestration: ties the stages together
├── freecad_env.py   locate FreeCAD bin/ and inject into sys.path
└── cli.py           argparse entry point
```

The dependency rule: `mesh_io`, `segmentation`, `boundary`, `fitting` must
**never** `import FreeCAD`. Only `builder` (and `freecad_env`) touch
OpenCASCADE. This keeps ~80% of the logic unit-testable without FreeCAD.

## Pipeline stages

### 1. Load & weld (`mesh_io`)
Parse binary or ASCII STL into raw triangles, then weld vertices whose
coordinates match within `weld_tol` (quantize → hash → dedupe). Output:
`vertices (V,3) float64`, `faces (F,3) int`.

### 2. Planar segmentation (`segmentation`)
- Per-face normals and areas.
- Edge→faces adjacency map (manifold edges have 2 incident faces).
- **Region growing:** seed from the largest unvisited face; BFS to neighbors
  whose face normal is within `angle_tol` of the seed plane normal *and* whose
  centroid is within `dist_tol` of the seed plane. Coplanar facets of a CAD
  part are exactly planar, so tolerances only absorb STL float noise.
- Output: a list of `Region(face_indices, plane_point, plane_normal)`.

Regions that are a single triangle (or fail planarity) are flagged for the
faceted fallback / future analytic fitting.

### 3. Boundary extraction (`boundary`)
For each region:
- Boundary edges = edges incident to exactly one face *within the region*.
- Chain boundary edges into ordered vertex loops.
- Project to the region plane (2D), compute signed areas: the largest-|area|
  loop is the outer wire; opposite-winding loops are holes.
- **Collinear simplification:** drop interior vertices where consecutive
  segments are collinear within tolerance — turns a meshed rectangle's edge
  (many points) back into a single straight segment.

### 4. Geometry build (`builder`, FreeCAD)
- For each planar region: build a `Part.Wire` per loop from the 3D boundary
  points, then `Part.Face([outer, *holes])`.
- `Part.Shell(faces)` → `Part.Solid(shell)`; validate with `shape.isValid()`
  / `check()`. If the shell isn't closed, `sewShape` then retry; if still bad,
  fall back to `Part.Shape.makeShapeFromMesh` for the whole part.
- `shape.removeSplitter()` to coalesce any residual coplanar splits.
- Export with `shape.exportStep(path)`.

### 5. Faceted fallback
The proven `makeShapeFromMesh → sew → Part.Solid` path. Always available via
`--faceted`, and used automatically when reconstruction yields an invalid solid
so the tool never fails to produce *a* watertight result.

## Roadmap

- [x] STL load + vertex welding
- [x] Planar region growing (numpy)
- [x] Boundary loop extraction + collinear simplification
- [x] FreeCAD builder validated on ground-truth prismatic + cylindrical parts
- [x] Cylinder detection + best-fit radius, rebuilt as analytic faces
- [x] Hole vs boss classification + correct face orientation
- [x] Unit scaling (mm/cm/m/in) and bounding-box inspection
- [x] GUI (drag-and-drop) + worker bridge + Windows executable
- [ ] Validation on real-world STLs (only synthetic ground-truth so far)
- [ ] Cone / sphere RANSAC fitting (`fitting.py`)
- [ ] Curved-surface face rebuild (B-spline fallback for free-form regions)
- [ ] Cylinders whose axis is not perpendicular to the end faces (blind/angled)
- [ ] Multi-body STL → multiple solids / compound
- [ ] macOS app bundle (deferred until Windows is confirmed in real use)
- [ ] CI matrix that runs the FreeCAD stage in a container

## Open questions

- Best heuristic to decide planar-vs-curved before fitting (curvature
  histogram vs. trial fits).
- Whether to sew in OCC or build a manifold ourselves from region adjacency
  (sewing is robust but can silently drop faces; manifold build is exact but
  fiddly at non-manifold edges).
- Tolerance auto-scaling from mesh bounding-box / edge-length statistics.
