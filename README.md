# mesh2step

Convert triangle-mesh STL files into **STEP solid models** using FreeCAD's
geometry kernel (OpenCASCADE) — but smarter than the usual "one STEP face per
triangle" approach.

## Why another STL→STEP tool?

Every existing open-source converter ([mesh2solid][m2s],
[stl_reverse_engineering][sre], Stepifi, …) does fundamentally the same thing:
it wraps the triangle mesh as a shell and calls it a solid. The result is a
*valid but faceted* STEP file where **every triangle becomes its own STEP
face**. A simple cube ends up with dozens of faces instead of six; a part with
10k triangles produces a 10k-face B-rep that no CAD tool can edit sanely.

`mesh2step` instead **reconstructs surfaces** before exporting:

1. **Coplanar-facet merging** (implemented) — triangles lying in a common
   plane are grouped into regions and rebuilt as a *single* planar STEP face
   with proper boundary loops (including holes). A meshed cube comes out with
   6 faces, not 12+.
2. **Analytic surface fitting** (roadmap) — cylindrical, conical and spherical
   regions are detected and rebuilt as true analytic surfaces instead of
   facets.

The faceted pipeline is kept as an automatic fallback for regions that can't be
reconstructed, so you always get a watertight solid.

[m2s]: https://github.com/Charles-Garrison/mesh2solid
[sre]: https://github.com/tsebukas/stl_reverse_engineering

## Status

> ⚠️ Early scaffold. The numpy segmentation core is implemented and tested; the
> FreeCAD geometry builder is implemented against the FreeCAD 1.x API but needs
> validation on real parts. See [DESIGN.md](DESIGN.md) for the roadmap.

## Requirements

- [FreeCAD](https://www.freecad.org/) 0.20+ (1.x recommended). The conversion
  runs inside FreeCAD's bundled Python via OpenCASCADE.
- Python 3.9+ with `numpy` for the segmentation core / tests (any interpreter).

The mesh segmentation core (`mesh_io`, `segmentation`, `boundary`) is pure
numpy and runs under *any* Python. Only the geometry builder and STEP export
need FreeCAD.

## Usage

Two ways to run, depending on whether you want to use your own Python or
FreeCAD's:

```bash
# 1. Let mesh2step find FreeCAD and inject it (uses your own venv + numpy):
python -m mesh2step input.stl -o output.step

# 2. Run directly under FreeCAD's interpreter (no path juggling):
freecadcmd -c "import mesh2step.cli as c; c.main()" -- input.stl -o output.step
```

Common options:

```
--angle-tol DEG     Max normal deviation to treat facets as coplanar (default 1.0)
--dist-tol MM       Max point-to-plane distance within a region   (default 0.01)
--weld-tol MM       Coincident-vertex welding tolerance            (default 1e-5)
--faceted           Skip reconstruction; emit the classic faceted solid
--freecad-bin PATH  Explicit path to FreeCAD bin/ (overrides auto-detect)
```

## How it works

```
STL ─▶ load + weld vertices ─▶ planar region growing ─▶ boundary loops
                                                              │
                          faceted fallback ◀──┐               ▼
                                              └── rebuild planar faces (FreeCAD)
                                                              │
                                          sew ─▶ shell ─▶ solid ─▶ STEP
```

See [DESIGN.md](DESIGN.md) for algorithm details and tolerances.

## Development

```bash
pip install -e ".[dev]"
pytest            # runs the numpy segmentation tests (no FreeCAD needed)
```

## License

MIT — see [LICENSE](LICENSE).
