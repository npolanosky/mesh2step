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

    # How many flat-face-normal directions to try as cylinder axes (by area).
    # More axes catch holes drilled perpendicular to small faces (e.g. pocket
    # floors) at the cost of some speed.
    max_candidate_axes: int = 12

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

    # Mesh preparation (uses FreeCAD's mesh kernel). Repair fixes duplicate
    # points/facets, degenerate facets, normals and non-manifold edges. Decimate
    # reduces triangle count (reduction fraction 0..1) to speed up heavy meshes.
    repair_mesh: bool = False
    decimate: float | None = None      # e.g. 0.5 => reduce by up to 50%
    decimate_tol: float = 0.1          # max geometric error (mm) when decimating

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
