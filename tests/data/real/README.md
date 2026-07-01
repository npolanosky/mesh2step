# Real-world test meshes (local only)

This folder holds large real-world STL/STEP files used for manual testing. It is
**git-ignored** (the meshes are tens of MB each) — drop files here locally.

Findings so far (`blank_topper.stl`, an organic part with drilled holes):

- The provided `*_reference.step` is another tool's (Fusion) conversion of the
  same mesh. It contains **19,858 planar facets + 39 cylinders and 0 solids**
  (not a closed solid) — including a spurious **Ø908.5 mm** cylinder fit to a
  near-flat 7-facet sliver.
- mesh2step now **rejects that phantom** (radius + angular-coverage guards) and
  detects the same real holes (Ø2.5 / 3.6 / 4.2), **harmonized** to single
  radii. There is no genuine Ø6.2 cylinder in the mesh; that feature is angled/
  organic and neither tool fits it as a clean cylinder.
- Closing the organic surface into a single watertight solid is the open
  problem (see DESIGN.md) — the faceted surface reconstructs to ~19.9k planar
  faces that don't sew shut around the analytic holes.
