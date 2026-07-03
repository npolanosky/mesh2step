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

    # Hard cost ceiling for the fully-closed boolean path, in faces. The
    # fully-closed tier relies on decimation to shrink the boolean base, so it
    # normally lifts ``boolean_max_base_faces`` to attempt the cuts even on dense
    # meshes. But if decimation is unavailable, or the post-decimation base is
    # still huge, an unbounded boolean run can grind for many minutes (DESIGN.md:
    # ~26 s/cut at 174k faces). This ceiling caps the base the fully-closed tier
    # will ever hand to the boolean cuts: above it, we skip booleans and fall
    # through to the watertight faceted solid (tier 3) instead of grinding.
    # ~130k keeps it well under a minute per cut while admitting most decimated
    # bases. None disables the ceiling (unbounded — the old behaviour).
    fully_closed_boolean_ceiling_faces: int | None = 130000

    # Post-export re-validation. After writing the STEP, re-read it and confirm
    # the solid still loads, is valid, is closed, and the solid count matches the
    # in-memory result. Some defects (self-intersecting wires from sliver
    # triangles) only surface through the STEP write/read round-trip: in-memory
    # the shape passes isValid(), but the exported file re-reads invalid. When a
    # decimation rung fails this check the fully-closed path backs off to the
    # next gentler rung, so a silent invalid export is impossible. One extra STEP
    # read per conversion — cheap. Set False to skip (e.g. for enormous files).
    revalidate_export: bool = True

    # Skip export re-validation automatically when the exported STEP exceeds this
    # size (bytes); re-reading a very large STEP can be slow. None -> never skip
    # on size grounds (only ``revalidate_export=False`` disables it).
    revalidate_export_max_bytes: int | None = 80_000_000

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

    # --- Curved-feature reconstruction (M1: resolution-scaled tolerances,
    # tangency prior, straight-edge fillets/chamfers). See docs/CURVED_FEATURES.md.

    # Fit tolerances scale with local mesh resolution so coarse STLs (big facets
    # on curved surfaces) still get analytic fits. Chord error scales with
    # edge_length^2/(8R): a 2 mm-edge facet on an R=1 fillet has ~0.5 mm sagitta,
    # 10x over the absolute cylinder_tol, so coarse bands otherwise stay faceted.
    # The accepted RMS-about-fit is max(curve_fit_tol_abs, curve_fit_tol_rel *
    # local_edge_length). curve_fit_tol_abs is a floor so fine meshes don't
    # over-tighten. The centroid/coverage guards in _fit_circle_for_facets scale
    # the same way, keeping false positives out while admitting coarse fits.
    curve_fit_tol_rel: float = 0.15   # accept RMS up to 15% of local edge length
    curve_fit_tol_abs: float = 0.02   # floor (fine meshes shouldn't over-tighten)

    # Detect straight-edge fillets (partial-arc cylinder sections between two
    # planes) and route residual chamfer strips back to planar faces. Behind this
    # flag so the whole feature can be disabled if it ever regresses a part.
    detect_fillets: bool = True

    # Tangency prior. If a fitted fillet is *nearly* tangent to its adjacent
    # flats, tangency is design intent: snap to exact tangency and derive the
    # radius from the plane-plane constraint (coarse meshes chord-cut the surface
    # and under-read radii). If *far* from tangent, it is intentional geometry:
    # keep the best-effort fit radius. The near/far threshold is resolution-scaled
    # (even a true tangent blend reads a defect of order median_dihedral/2 on a
    # coarse mesh): tangency_threshold_deg = max(tangency_floor_deg, tangency_k *
    # median_dihedral_deg).
    tangency_floor_deg: float = 3.0
    tangency_k: float = 1.0

    # Reject fillets whose derived radius is below this many local edge lengths
    # (sub-facet fillets can't be built cleanly). The absolute floor is
    # min_fillet_radius; the effective floor is the larger of the two.
    min_fillet_radius: float = 0.2
    min_fillet_radius_edges: float = 0.5

    # A straight-edge fillet is a partial arc: its facet centroids cover well
    # under the full circle. This is the minimum coverage a fillet band must span
    # (fillets are exempt from min_cylinder_coverage, which they never meet).
    min_fillet_coverage: float = 0.04

    # Maximum coverage for a band to still be treated as a partial-arc fillet
    # (above this it is a real, near-full cylinder that detect_cylinders owns).
    max_fillet_coverage: float = 0.6

    # --- Swept / extruded curved-wall reconstruction (M4). See
    # docs/CURVED_FEATURES.md §6a — 60%+ of residual facetedness on the corpus.
    # A swept wall is a constant-cross-section extrusion: its facet normals are
    # all perpendicular to one direction d, and the profile repeats along d. It
    # arrives tessellated as a fan of thin planar strips (a smooth chain). We
    # detect the region (fixed-d chain growth), fit the 2D profile (line + arc +
    # B-spline with 2D tangency snapping), and extrude it along d.

    # Detect and rebuild swept curved walls. Behind this flag so the whole
    # feature can be disabled if it ever regresses a part.
    detect_swept_walls: bool = True

    # A member strip's normal must stay within this |cos| of perpendicular to the
    # fixed sweep direction d (|n·d| <= this) to join the sweep. Small so end
    # caps (n parallel to d) and non-swept curvature are excluded.
    swept_axis_perp_tol: float = 0.08

    # Minimum planar strips (arc rows) a swept chain must contain, and minimum
    # total facets — below these it is noise, not a genuine tessellated sweep.
    swept_min_regions: int = 4
    swept_min_facets: int = 12

    # The sweep must span a meaningful extent along d (mm) — a sub-millimetre
    # "sweep" is a sliver, not a wall. Also expressed relative to the profile
    # size so a tiny feature isn't rebuilt as an extrusion.
    swept_min_extent: float = 1.0

    # Max RMS (mm) of the fitted 2D profile curve to the region's rail points,
    # resolution-scaled the same way as fillets (max of this and
    # curve_fit_tol_rel * local_edge). Above it the sweep is left faceted.
    swept_profile_tol_abs: float = 0.05

    # 2D tangency snap: where a fitted arc/spline meets a straight profile segment
    # within this angle (deg, resolution-scaled by median dihedral) of tangency,
    # snap to exact tangency — the product-owner rule applied in 2D. Reuses the
    # same near/far policy as fillets (tangency_floor_deg / tangency_k).

    # Minimum profile-segment length (mm) to fit as its own line/arc; shorter
    # runs are folded into the B-spline. Keeps micro-noise from spawning
    # segments.
    swept_min_segment_len: float = 0.8

    # Arc acceptance: a run of profile points is an arc when the circle fit RMS
    # is below swept_profile_tol_abs + this fraction of the radius. Rail points
    # sit ON the true circle, so this is noise-scale (NOT chord-error scale — a
    # loose relative tolerance would let one giant circle "fit" a whole
    # line+arc+line profile). Otherwise the run goes to a line/B-spline.
    swept_arc_tol_rel: float = 0.001

    # After the swept lens ops, remove micro-sliver planar faces (below this
    # area, mm^2) that drag a large flat into a smooth chain — decimation and
    # boolean-seam wedges of negligible area that read as residual tessellation.
    # Only chains of one dominant face plus a few such slivers are touched, via
    # OCC defeaturing with validity/volume guards (reverts wholesale).
    swept_defeature_slivers: bool = True
    swept_sliver_max_area: float = 0.5

    # --- RTAF: Residual Tessellation Area Fraction (post-conversion quality
    # metric). See docs/CURVED_FEATURES.md §6a. Fraction of the output solid's
    # surface AREA that sits in "smooth chains" — runs of >=3 connected planar
    # faces whose neighbour normals step by rtaf_angle_lo..rtaf_angle_hi degrees
    # (a tessellated-curve fan; exactly-coplanar splits and real feature edges
    # are excluded). Area-weighted so a curve shipped as a fan of thin flats
    # reads high even when skipped_facets is zero (the strips technically
    # "reconstructed"). Higher = more faceted-looking. Computed on the final
    # shape, added to stats["rtaf"], surfaced in the quality report + failstore.

    # Compute RTAF at all. Off skips it entirely (no cost).
    compute_rtaf: bool = True

    # A neighbour normal step below rtaf_angle_lo (deg) is effectively coplanar
    # (a genuine split), above rtaf_angle_hi is a real feature edge. In between
    # is a near-tangent tessellation step.
    rtaf_angle_lo: float = 0.5
    rtaf_angle_hi: float = 30.0
    # Minimum chain length (number of connected planar faces) to count.
    rtaf_min_chain: int = 3

    # Skip the RTAF computation when the output solid has more than this many
    # faces (the O(faces * edges) adjacency scan gets slow on enormous shells).
    # The corpus's largest chain analysis (tweezer, ~10.5k faces) stays in
    # single-digit seconds well under this cap. None disables the guard.
    rtaf_max_faces: int | None = 40000

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
