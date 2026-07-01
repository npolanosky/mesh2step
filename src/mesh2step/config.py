"""Conversion configuration — all tolerances and flags in one place."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Source-unit -> millimetre scale factors. STEP output is always millimetres,
# so the mesh is scaled by these on load. STL itself is unit-less; the user
# tells us what units the mesh was exported in.
UNIT_SCALE_MM: dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
}


@dataclass
class ConversionConfig:
    """Tolerances and flags controlling the STL->STEP conversion.

    The mesh is scaled to millimetres on load (see ``source_units``), so every
    distance tolerance below is interpreted in **millimetres**.
    """

    # Units the STL was exported in; the mesh is scaled to mm on load. One of
    # UNIT_SCALE_MM ("mm", "cm", "m", "in"). Use ``scale_override`` for a custom
    # factor (takes precedence when set).
    source_units: str = "mm"
    scale_override: float | None = None

    # Coincident-vertex welding tolerance. STL stores each triangle's vertices
    # independently, so we merge vertices closer than this to recover topology.
    weld_tol: float = 1e-5

    # Max angle (degrees) between a facet normal and its region's plane normal
    # for the facet to be considered coplanar.
    angle_tol_deg: float = 1.0

    # Max point-to-plane distance for a facet to join a planar region.
    dist_tol: float = 1e-2

    # Tolerance for dropping collinear vertices from a boundary loop. A vertex
    # is removed when its perpendicular distance to the chord of its neighbours
    # is below this value.
    collinear_tol: float = 1e-4

    # Minimum facets a region must have to be rebuilt as a single planar face.
    # Below this we leave facets for the faceted fallback (avoids spurious
    # micro-faces from mesh noise).
    min_region_facets: int = 1

    # Skip reconstruction entirely and emit the classic faceted solid.
    faceted: bool = False

    # Guarantee a watertight solid. If surface reconstruction can't close (common
    # for organic meshes, where analytic hole edges can't meet the faceted
    # surrounding surface), fall back to a watertight faceted solid. Slower, and
    # holes stay faceted on organic parts — but the body is closed.
    full_closed: bool = False

    # Emit un-mergeable facets as locally-merged patches so the reconstructed
    # shell has no gaps and sews watertight — keeping merged planar faces +
    # analytic holes while staying manifold. Enabled by the fully-closed path.
    fill_faceted_gaps: bool = False

    # Sewing tolerance (mm) when stitching faces into a solid. Analytic faces
    # (exact circles/planes) and raw mesh-derived patches meet at edges that are
    # coordinate-identical in theory but can differ by FP noise; a small nonzero
    # tolerance lets OCC bridge that without needing bit-exact vertices.
    sew_tolerance: float = 1e-3

    # Boolean clean-up (fully-closed tier 2) cuts each analytic hole into the
    # faceted base solid; every cut costs O(base faces), so on very dense meshes
    # this becomes minutes. Above this triangle count, skip boolean clean-up and
    # fall through to the plain faceted solid. Raise it if you're willing to wait
    # (or, better, decimate the mesh first). None disables the guard.
    boolean_max_base_faces: int | None = 60000

    # Explicit path to FreeCAD's bin/ directory (overrides auto-detection).
    freecad_bin: str | None = None

    # Detect cylindrical regions and rebuild them as analytic cylinder faces
    # with a best-fit radius (clean holes/bores) instead of facets.
    detect_cylinders: bool = True

    # Max RMS residual (mm) of facet vertices to a fitted cylinder for the fit
    # to be accepted.
    cylinder_tol: float = 5e-2

    # Minimum facets a curved region must have to attempt a cylinder fit.
    min_cylinder_facets: int = 8

    # Reject fitted cylinders/bosses smaller than this radius (mm). Tiny curved
    # facet clusters on organic surfaces fit near-zero-radius circles and would
    # otherwise appear as dozens of spurious micro-holes; real holes are larger.
    min_cylinder_radius: float = 0.4

    # How many flat-face-normal directions to try as cylinder axes (by area).
    # More axes catch holes drilled perpendicular to small faces (e.g. pocket
    # floors) at the cost of some speed.
    max_candidate_axes: int = 12

    # Also derive candidate axes from isolated curved regions, so holes drilled
    # at an arbitrary angle (axis not perpendicular to any flat face) are found.
    detect_angled: bool = True

    # A facet is on a curved surface if an edge-neighbour's normal differs by
    # more than the coplanar tolerance but less than this (a smooth transition);
    # a larger difference is a sharp feature edge (a flat-face boundary), not
    # curvature. Separates hole walls from flat faces regardless of facet count.
    curve_max_deg: float = 50.0

    # Minimum fraction of the full circle the facets must cover (0..1). Holes
    # and bosses wrap the whole way around (~1.0); this rejects shallow arcs and
    # slivers that algebraically fit a huge circle (the classic false positive).
    # Set below 0.5 to admit partial arcs (holes clipped by intersecting holes);
    # the radius + centroid-radius + RMS guards keep false positives out.
    min_cylinder_coverage: float = 0.33

    # Reject fitted radii larger than this (mm). None -> the mesh's largest
    # bounding-box dimension. A full cylinder of radius r spans 2r across, so
    # 2r <= (a part dimension) <= largest dimension; using the largest dimension
    # as the cap still admits round parts whose outside diameter equals the part
    # size (radius = size/2), while rejecting shallow-arc mega-circles.
    max_cylinder_radius: float | None = None

    # Mesh preparation. Repair (FreeCAD mesh kernel) fixes duplicate
    # points/facets, degenerate facets, normals and non-manifold edges.
    repair_mesh: bool = False

    # Planar-preserving decimation (pymeshlab quadric edge-collapse). Collapses
    # over-tessellated flat regions while keeping holes/curves dense and edges
    # sharp — it both shrinks the file and, crucially, makes the boolean
    # clean-up tractable (its cost is O(base faces) per hole). If the mesh has
    # more than ``decimate_target_faces`` triangles it is decimated down toward
    # that count. Set to None to disable. The fully-closed path enables a
    # default target automatically when needed.
    decimate_target_faces: int | None = None

    # Snap near-equal detected radii to a shared rounded value, so triangulation
    # noise doesn't yield 6.04/6.05/6.06 for what is really one 6.05 hole.
    harmonize_radii: bool = True
    harmonize_rel_tol: float = 0.03   # radii within 3% are treated as the same
    harmonize_round: float = 0.05     # snap the shared radius to this grid (mm)

    @property
    def angle_tol_cos(self) -> float:
        """Pre-computed cosine of the angle tolerance for dot-product gating."""
        return math.cos(math.radians(self.angle_tol_deg))

    @property
    def scale_to_mm(self) -> float:
        """Factor to multiply mesh coordinates by to obtain millimetres."""
        if self.scale_override is not None:
            return self.scale_override
        try:
            return UNIT_SCALE_MM[self.source_units.lower()]
        except KeyError as exc:
            raise ValueError(
                f"unknown source_units {self.source_units!r}; "
                f"expected one of {sorted(UNIT_SCALE_MM)}"
            ) from exc
