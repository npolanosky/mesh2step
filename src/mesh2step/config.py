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

    # Resolution-scaled planar-merge tolerances (mirror curve_fit_tol_rel). The
    # absolute angle_tol_deg / dist_tol above are floors; when the rel factors are
    # non-zero the effective planar-merge tolerance grows with the mesh's own
    # tessellation noise (median smooth-dihedral step, median edge length):
    #
    #   effective angle = clamp(angle_tol_deg .. planar_angle_tol_cap_deg,
    #                           planar_angle_tol_rel * median_smooth_dihedral)
    #   effective dist  = clamp(dist_tol .. planar_dist_tol_cap,
    #                           planar_dist_tol_rel * median_edge)
    #
    # BOTH default OFF (rel = 0 -> legacy pure-absolute behaviour). Investigation
    # on the reported coarse scans showed loosening segmentation is the WRONG lever
    # for merging their flats: (a) the 1.0° angle boundary is ALSO the flat/curved
    # discriminator the swept-wall and dome-consensus detectors key off, so any
    # angle loosening big enough to absorb the scans' ~1.7° flat-normal noise also
    # merges the fine curved detectors' arc rows and drops those analytic features;
    # and (b) even pure distance loosening perturbs the freeform/swept detector
    # inputs enough to trigger a pathologically slow boolean on some meshes while
    # merging almost no real flat area. On the reported coarse organic scans the
    # "flats" are additionally warped by decimation (raw median facet step 0.8°,
    # post-decimation 1.7°), so they are not truly planar and cannot be merged
    # into valid planar faces at all — the output is the correct watertight
    # representation of a genuinely faceted mesh. The rel fields are kept as an
    # explicit opt-in for callers with a mesh known to be prismatic-flat and free
    # of curved analytic features (where a modest loosening is safe).
    planar_angle_tol_rel: float = 0.0       # x median smooth-dihedral step (off)
    planar_angle_tol_cap_deg: float = 1.5   # hard ceiling on the effective angle
    planar_dist_tol_rel: float = 0.0        # x median edge length (off)
    planar_dist_tol_cap: float = 0.1        # hard ceiling on the effective dist (mm)

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

    # Multi-body support. A mesh can contain several disjoint bodies that touch
    # nowhere (a print-in-place hinge's knuckle + pin, a snap-fit lid + base).
    # The reconstruction/boolean pipeline assumes one connected body, so a
    # multi-body mesh never closes into a single watertight solid. When enabled,
    # disjoint connected components are detected up front and each is converted
    # independently through the full pipeline, then combined into one STEP
    # compound of N solids (watertightness is required per body). A single-body
    # mesh is unaffected — it takes the ordinary path byte-for-byte.
    multi_body: bool = True

    # Junk-body filtering before multi-body dispatch. A welded-and-split STL can
    # leave tiny degenerate non-bodies beside the real part (an 8-facet sliver, a
    # stray shell). Left in, such junk (a) flips the "auto" combine/separate
    # heuristic to "separate" on a part that is really single-body, and (b)
    # hard-aborts the worker later (uncaught C++ CADKernelError when the sew is
    # handed the 2 degenerate facets it collapses to). A component is dropped only
    # when it is BOTH below ``min_body_facets`` facets AND below ``min_body_area_frac``
    # of the total mesh area — both must be tiny, so a small-but-real body (a
    # print-in-place hinge pin) is never dropped for being small in one axis alone.
    # The corpus gap is wide: real secondary bodies run 892+ facets / 33%+ area
    # (snap_fit is the smallest); junk is <=12 facets / <=1.2% area (sharpie's
    # 8-facet flake, labstack_keystone's two 12-facet flakes). 32 facets / 2% sits
    # in that gap with large margin. If EVERY component is junk the split is kept
    # unchanged (never end up with nothing to convert). min_body_facets=0 disables.
    min_body_facets: int = 32
    min_body_area_frac: float = 0.02

    # How a multi-body mesh (>= 2 disjoint connected components) is handled.
    # Only consulted when ``multi_body`` is True and the mesh actually splits:
    #
    #   "auto" (default): a conservative heuristic per file. If the bodies'
    #       bounding boxes overlap or touch (they are almost certainly one part
    #       that was exported as several shells — e.g. a lid modelled through its
    #       base, tabs interpenetrating a wall), attempt to COMBINE them into a
    #       single solid via the manifold3d winding-number union. If every body
    #       is bbox-disjoint (a genuinely separate print-in-place hinge pin, two
    #       loose parts on one plate), keep them SEPARATE (one compound of N
    #       solids). On any failure of the combine attempt it falls back to
    #       separate — auto never regresses relative to the old behaviour.
    #
    #   "combine": always union all bodies into ONE solid via manifold3d (the
    #       same winding-number boolean the self-intersection resolver uses).
    #       For bodies that are really one part split into coincident/near-
    #       coincident shells. A small weld/merge pass first bridges tiny gaps so
    #       near-coincident (not bit-exact) faces still fuse.
    #
    #   "separate": always convert each body independently and emit a STEP
    #       compound of N solids (the historical multi-body behaviour).
    multibody_mode: str = "auto"

    # Gap (mm) up to which "combine" welds near-coincident vertices across bodies
    # before the union, so shells that meet with sub-tolerance FP gaps still fuse
    # into one solid. manifold3d's own merge handles bit-exact/quantised
    # coincidence; this widens it slightly for meshes exported with tiny seams.
    multibody_combine_weld: float = 1e-3

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

    # Bounding-box growth guard for boolean cut/fuse-back ops. A legitimate hole
    # cut removes material and a boss/fillet fuse-back trues up a wall over its
    # own extent — neither should enlarge the part's overall silhouette. A
    # mis-detected feature (a spurious giant tilted cylinder fitted to a curved
    # corner, an over-radius fillet) DOES grow the box, silently distorting the
    # exported dimensions by 10-30%. Any boolean-clean step that expands a side of
    # the solid's bounding box by more than this relative fraction is reverted
    # (the feature is left faceted). None disables the guard.
    boolean_max_bbox_growth: float | None = 0.02

    # Hard bounding-box distortion ceiling (fraction). A boolean/reconstruction
    # tier can silently ship a watertight, valid-on-reread solid whose dimensions
    # are catastrophically wrong (gridfinity_base_lid: a degenerate sphere fuse
    # collapsed a 210x126x12mm plate to a 6mm cube — 97% off — yet it passed every
    # validity gate). The per-op guards catch this at the source, but this is the
    # last-line safety net: if the adopted tier's output bbox differs from the
    # input mesh by MORE than this fraction on any axis, that tier's result is
    # REJECTED (never shipped), the pipeline falls back to a watertight faceted
    # solid (dimensionally faithful by construction), and the quality verdict is
    # forced to "problems" with a loud error. 0.25 (25%) sits well above the
    # corpus's worst legitimate drift (~16% on the carabiner) yet far below any
    # real collapse. None disables the gate.
    bbox_reject_delta: float | None = 0.25

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

    # Designed-polygon guard (P0: hexagonal bores must NOT become circular holes).
    # A regular polygon bore (hex/pentagon/octagon...) is a handful of FLAT facet
    # panels meeting at exact, uniform, SHARP dihedral steps (a hexagon: 6 distinct
    # normals 60 deg apart), and its vertices lie exactly on the circumscribed
    # circle (RMS ~ 0) — so the algebraic circle fit is *perfect* and the
    # resolution-scaled centroid guard (sized for a coarse CURVED surface's chord
    # error) admits it. But it is a designed feature that is ALREADY ideal (N flat
    # planes); it must stay as planar faces, never be replaced by an analytic
    # cylinder. This guard discriminates a designed polygon from a genuine coarse
    # circle by two resolution-INDEPENDENT signals (a polygon's are fixed by its
    # side count, a circle's shrink with mesh density):
    #   (a) the median angular step between consecutive DISTINCT wall-facet normals
    #       about the axis. A regular N-gon steps 360/N (hex = 60 deg) — far above
    #       the smooth-curvature ceiling; a tessellated circle steps a few degrees.
    #   (b) the facet-centroid-radius deficit: an N-gon's centroids sit at the
    #       apothem = circumradius * cos(180/N) (hex = 0.866 r, a fixed ~13.4%
    #       deficit); a circle's centroids sit at ~r * (1 - edge^2/8r^2), a small
    #       deficit that tracks the sagitta. A large deficit that MATCHES a
    #       low-N-gon apothem AND exceeds the plausible chord sagitta is a polygon.
    # Detect polygons up to this many sides; above it a coarse regular polygon is
    # indistinguishable from a circle and legitimately rounds (design intent).
    detect_polygons: bool = True
    polygon_max_sides: int = 12
    # A wall-facet-normal step above this (deg) is a designed sharp edge, not a
    # curvature step. Mirrors curve_max_deg; kept separate so the polygon guard can
    # be tuned without moving the curvature band.
    polygon_min_facet_step_deg: float = 24.0
    # Distinct normals are merged when within this angle (deg): triangulation of a
    # single flat panel yields ~coplanar normals that must collapse to one side.
    polygon_normal_merge_deg: float = 8.0
    # Accept the apothem-deficit corroboration when the measured centroid deficit
    # is within this fraction of the ideal cos(180/N) for the detected side count.
    polygon_apothem_tol: float = 0.25
    # Decisive discriminator (c): a designed polygon's facets are far coarser than
    # the local mesh would tessellate a real curve into. Treat the band as a real
    # tessellated circle (keep the analytic fit) when the measured per-facet
    # angular step is LESS than this factor times the circle-at-this-resolution
    # step (degrees(local_edge / radius)). A designed hexagon steps 60 deg while a
    # circle at the same mesh steps a few degrees, so this cleanly separates them;
    # the factor leaves margin so a genuinely coarse circle is never called a
    # polygon.
    polygon_coarseness_factor: float = 2.5

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

    # Planarity-damage back-off (task §1). Planar-preserving decimation collapses
    # over-tessellated flats cheaply, but on a COARSE organic scan the quadric
    # collapse warps the flats past the 1.0° coplanar gate — the measured Patton
    # case: raw flats step 0.8° (segment into large regions) but the 12k-face
    # decimation warps them to 1.7-1.9°, shattering ~2800 large flats into <40 and
    # dropping the area in large flats from 48% to 32% (coverage ratio 0.67). The
    # user sees "everything faceted" and RTAF stays ~0.70. Loosening the planar
    # gate to re-absorb the warped noise regresses the curved detectors (config
    # above), and post-hoc merging the warped geometry can't produce valid faces.
    # The un-attacked lever is to NOT ship geometry decimation destroyed: after
    # each decimation rung, compare area-weighted planar coverage (fraction of
    # surface area in planar regions >= planarity_min_region_facets, via
    # segmentation.planar_coverage) against the RAW mesh's; if the ratio falls
    # below planarity_damage_min_ratio, the rung is treated as FAILED and the
    # boolean back-off ladder steps to the next gentler rung (2x target, then
    # undecimated) — exactly like the export-revalidation criterion. On the Patton
    # files this backs 12k off to 24k (ratio 0.84/0.74) where the flats survive as
    # large merged faces. A gentler rung means a larger boolean base (each cut is
    # O(base faces)), so the cost ceiling still bounds the tradeoff. Prismatic
    # parts whose flats survive decimation cleanly (ratio ~1.0) never trip it and
    # keep the fast 12k base byte-for-byte.
    planarity_damage_check: bool = True
    # Coverage ratio (decimated/raw) below which a rung is treated as flat-damaged
    # and backed off. Calibrated from the measured pipeline ratios: the Patton 12k
    # rung is 0.64-0.67 (must reject — the "everything faceted" rung) and its 24k
    # rung is 0.74-0.84 (must accept — flats survive as large faces there); every
    # prismatic corpus part is >= 0.87 at 12k (gear 0.87, drive_bay 1.0) so they
    # never trip it and keep the fast 12k base. 0.70 sits in that window: it backs
    # 12k off to 24k on the damaged organic scans while landing on a rung that
    # still builds a boolean-clean solid — crucially NOT cascading all the way to
    # the undecimated base (which, above the boolean ceiling, would ship a plain
    # faceted solid — worse). None disables the gate (same as check off).
    planarity_damage_min_ratio: float | None = 0.70
    # A planar region must have at least this many facets to count as a "real"
    # flat in the coverage metric (below it is mesh-noise; a warped flat shatters
    # into exactly these sub-threshold micro-regions, which is what we detect).
    planarity_min_region_facets: int = 8

    # Snap near-equal detected radii to a shared rounded value, so triangulation
    # noise doesn't yield 6.04/6.05/6.06 for what is really one 6.05 hole.
    harmonize_radii: bool = True
    harmonize_rel_tol: float = 0.03   # radii within 3% are treated as the same
    harmonize_round: float = 0.05     # snap the shared radius to this grid (mm)

    # --- Threads (M5.2). See docs/M5_HELICAL_PATTERNED.md §1. A thread is a
    # single-family helix on a cylindrical band, whose wall vertices satisfy
    # z = (pitch/2pi)*phi + z0. We fit that invariant, gate on coverage/RMS/pitch,
    # and suppress to the pitch-diameter cylinder (metadata: pitch/starts/hand/
    # crest/root in stats["threads"] + the <name>_features.json sidecar). True
    # helical B-reps are deferred (build_true_threads stays off — M6 stretch).

    # Detect and suppress threaded bands. Behind this flag so it can be disabled.
    detect_threads: bool = True

    # Rebuild threads as true helical B-reps (Part.makeHelix + MakePipe). OFF:
    # a helical sweep booleaned against a faceted base is OCC's worst self-
    # intersection case (design §governing decision) — deferred M6 stretch.
    build_true_threads: bool = False

    # Minimum wall facets a thread band must contain to attempt a helix fit.
    thread_min_facets: int = 60

    # Angular coverage gate (turns): the primary false-positive guard. One turn
    # is a chamfer/ramp; a real thread wraps well past 1.5 turns. Turns are
    # measured as axial_extent / fitted_pitch (robust — no fragile phi unwrap).
    thread_min_turns: float = 1.5

    # Upper turn bound: a real thread has a handful to a couple dozen turns. A
    # band read along a WRONG axis wraps the whole part and phase-collapses a
    # spurious very-fine "thread" of many dozens of turns — reject those.
    thread_max_turns: float = 20.0

    # The band must wrap the full circle about the axis (a thread groove goes all
    # the way round); a partial-arc band is a ramp/chamfer, not a thread.
    thread_min_coverage: float = 0.75

    # How many dominant flat-normal / principal directions to try as thread axes
    # (beyond the detected cylinders' own axes). Kept small: trying every
    # candidate axis lets a wrong (e.g. horizontal) direction phase-collapse a
    # spurious full-diameter "thread" out of a cap's flat faces.
    thread_max_axes: int = 3

    # The band facets must genuinely face radially (like a cylinder wall); a band
    # seen along a wrong axis is really the part's flat top/sides with weak radial
    # alignment. Require the mean |normal·radial| to clear this floor.
    thread_min_radial_align: float = 0.55

    # When a thread's axis coincides with a detected cylinder, the thread band's
    # radius must be within this relative tolerance of a seed cylinder's radius (a
    # thread is ON its cylinder). Rejects a spurious thread found at a very
    # different radius from a tiny cross-hole's axis (knurled_knob's r=1 side holes
    # seeding a bogus r=3.5 thread). Only applied when seeds exist for the axis.
    thread_seed_radius_tol: float = 0.6

    # Pitch sanity window, as a fraction of the band radius. A thread's pitch is
    # a small-to-moderate fraction of its diameter; outside this the helix fit is
    # spurious (a shallow ramp reads as a giant pitch, noise as a tiny one).
    thread_min_pitch_rel: float = 0.03
    thread_max_pitch_rel: float = 1.5

    # Phase-collapse gate (THE thread discriminator). On a single-start thread the
    # phase theta = phi - (2pi/pitch)*z collapses to one value, so its resultant
    # length R = 1 - circular_variance is high (bottle-cap thread ~0.66). A knurl
    # (two crossing helix families) or gear teeth (no helix) never collapse at any
    # pitch (R stays near 0). Require R at the fitted pitch to clear this floor.
    # 0.35 sits well below a real thread's R and far above a knurl/gear's.
    thread_min_resultant: float = 0.35

    # Accept the helix fit when its RMS residual (mm) is within this multiple of
    # the resolution-scaled surface tolerance (a secondary sanity check; the
    # phase-collapse resultant gate above is the primary false-positive net).
    thread_rms_mult: float = 6.0

    # --- Knurling (M5.1). See docs/M5_HELICAL_PATTERNED.md §3. A knurl is a
    # high-frequency micro-roughness on a cylindrical grip: hundreds of tiny
    # facets whose radial deviation from the wall is small but whose normals tilt
    # in a *bimodal* pattern (a diamond knurl's two crossing helix families give
    # two symmetric axial-tilt lobes). It is never rebuilt as bumps (absurd cost);
    # it is suppressed to its median-radius mid-surface cylinder via
    # _boolean_clean_cylinder, with the pattern kept as metadata (stats["knurling"]).

    # Detect and suppress knurled bands. Behind this flag so it can be disabled.
    detect_knurling: bool = True

    # Minimum facets a knurled band must contain. A knurl is a DENSE micro-facet
    # field wrapping a grip; a real one runs into the thousands, so this floor is
    # well above any incidental cluster of tilted wall facets.
    knurl_min_facets: int = 200

    # A wall facet joins the knurl band only if its radial distance is within this
    # tolerance of the band's median radius (max of abs mm and rel fraction). The
    # crest/root excursion of a knurl is small vs the radius; this isolates the
    # knurl from the hub/bore and any coaxial smooth land.
    knurl_band_tol_abs: float = 0.3
    knurl_band_tol_rel: float = 0.06

    # Radial-excursion ceilings (the decisive knurl/gear discriminator). A knurl
    # is a micro-roughness: its crest-to-root depth (p10..p90 radial span of the
    # wall facets near the median radius) is a small fraction of the radius AND
    # under a facet edge length. Gear teeth and coarse grip ribs have a LARGE
    # intrinsic radial excursion (teeth ~9% of R and ~2.3 edges; ribs deeper) and
    # must NOT be flattened to a cylinder — they route to the gear/swept path.
    # Measured on the corpus: knob knurl 2.0% of R / 0.77 edges; gear teeth 9.0% /
    # 2.3 edges; bottle grip 22% / 25 edges. Both ceilings must hold.
    knurl_max_excursion_rel: float = 0.04
    knurl_max_excursion_edges: float = 1.5

    # Minimum fraction of the full circle the band must span (a knurl wraps the
    # whole grip); rejects a partial tilted-facet sliver.
    knurl_min_coverage: float = 0.75

    # Bimodality gate (the discriminator). The wall-facet normals' axial-tilt
    # component must score at least this on the bimodal-symmetric-dip metric
    # (_knurl_bimodality): a plain cylinder wall scores ~0, a diamond knurl ~0.6+.
    # Set high enough that a smooth wall or a gentle sweep never trips it.
    knurl_min_normal_bimodality: float = 0.35

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

    # Organic-surface guard for straight-edge fillet detection, by border reuse.
    # A *real* straight-edge fillet rounds between two flats that belong to it
    # alone: on a prismatic part each fillet has two dedicated bordering planar
    # regions (each serves that one fillet). A smooth freeform / vase-mode wall
    # segments into stacked rings, so a few large panels each border many "fillet"
    # slices — the same border region is reused across a dozen candidates. That
    # reuse is what cleanly separates a hex-planter vase wall (105 candidates over
    # 22 border regions, max reuse 13) from a genuine multi-fillet part (a T-slot
    # connector: 12 fillets, zero border reuse). A candidate whose either border
    # region serves more than this many fillet candidates is rejected as a
    # misclassified organic strip rather than built (which is what crashed OCC on
    # the vase). 2 tolerates a flat legitimately shared by two adjacent fillets.
    fillet_max_border_reuse: int = 2

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

    # --- Gears / whole-outline extrusion (M5.3). See docs/M5_HELICAL_PATTERNED.md
    # §2. A gear IS an extrusion; M4 recognizes the shape but drops it because
    # per-arc lens ops are O(arcs × base) and never converge. Instead we claim a
    # repeated-arc CLOSED profile roughly centered on the axis WHOLESALE: assemble
    # the full 2D cross-section (lines + arcs + spline spans) into ONE closed
    # Part.Wire -> Face -> extrude -> ONE guarded boolean. O(base) once. The
    # central bore is cut after the gear fuse (ladder discipline).

    # Route repeated-arc closed centered profiles to whole-outline extrusion.
    reconstruct_gears: bool = True

    # A profile is a "ring" (a gear cross-section / splined shaft) when its
    # outline wraps the full circle about its OWN centroid and its points sit at a
    # roughly consistent radius — the radial spread (std/mean about the centroid)
    # is at most this. A gear's tip/root teeth vary the radius modestly (~0.15);
    # a one-sided wall panel has huge spread and fails. Origin-frame independent
    # (the profile-plane origin is an arbitrary rail point, not the axis).
    gear_ring_spread_rel: float = 0.35

    # Sanity ceiling on the number of profile segments a whole-outline extrusion
    # will assemble into a wire. A noisy outline with thousands of micro-segments
    # is rejected (left faceted) rather than handed to a pathological wire build.
    gear_max_profile_segments: int = 2000

    # Force the gear outline's flank spans to build as B-splines rather than
    # short arcs: jagged tessellated involute flanks make many tiny arcs whose
    # exact tangency is fragile; a spline through the span's points is robust.
    gear_flanks_as_spline: bool = True

    # Max whole-outline fuse attempts per part (largest-extent outlines first).
    # Each fuse is an O(base_faces) boolean; a decimated gear can fragment its
    # tooth ring into several axial regions, so cap the attempts to bound the time
    # budget (a fuse that doesn't lower RTAF reverts anyway).
    gear_max_ops: int = 2

    # Repeated-tooth guard (M4 gear regression). An involute gear / splined shaft
    # fits as one swept region whose profile is dozens of near-identical short
    # arcs (the tessellated tooth flanks). Building one boolean lens op per arc is
    # O(arcs × base_faces) and never converges (gear_box_gear_v2 timed out >2 min
    # at 456 arcs). Such a profile is dropped wholesale (teeth stay faceted by
    # design for now). A profile trips the guard when it has >= swept_repeat_arc_min
    # arcs whose radii cluster into only <= swept_repeat_distinct_frac of that
    # count of distinct values (radii within swept_repeat_radius_rel are one).
    swept_repeat_arc_min: int = 12
    swept_repeat_distinct_frac: float = 0.5
    swept_repeat_radius_rel: float = 0.05

    # Per-part swept lens-op cost budget (M4 gear regression). Each lens op is a
    # boolean against the faceted base, cost ~O(base_faces); the whole batch is
    # O(distinct_arcs × base_faces). When that product exceeds this budget the
    # batch is skipped wholesale with a clear log (the walls stay faceted) rather
    # than grinding for minutes on a pathological mesh. The repeated-arc guard
    # already drops the gear-tooth profiles (0 arcs), so this is a belt-and-braces
    # ceiling for any residual blow-up — set well ABOVE the corpus's real swept
    # parts (tweezer ~50 arcs × ~9 k faces = 447 k built fine; USB-holder /
    # drive_bay similar) so it never vetoes a legitimate reconstruction, while
    # still catching a runaway (a gear's raw 456 arcs × 12 k = 5.5 M). None
    # disables the budget (unbounded — the old behaviour).
    swept_op_budget: int | None = 1_500_000

    # After the swept lens ops, remove micro-sliver planar faces (below this
    # area, mm^2) that drag a large flat into a smooth chain — decimation and
    # boolean-seam wedges of negligible area that read as residual tessellation.
    # Only chains of one dominant face plus a few such slivers are touched, via
    # OCC defeaturing with validity/volume guards (reverts wholesale).
    swept_defeature_slivers: bool = True
    swept_sliver_max_area: float = 0.5

    # --- Freeform B-spline sheets (Candidate B). See
    # docs/ORGANIC_CONVERSION_RESEARCH.md (Candidate B). The residual after all
    # analytic + swept + sphere detectors can include genuinely doubly-curved
    # regions (a curved lid, an ergonomic shell, a camera-adapter panel) that no
    # analytic fit and no constant-cross-section sweep claims. Where such a
    # region is a *height field* (injective under a projection axis), we resample
    # the mesh on a (u,v) grid, fit a single trimmed B-spline face, and replace
    # the faceted region via a guarded boolean cut/fuse (the M4 rollback net) —
    # adopted ONLY when the result is watertight, bbox-stable, and lowers RTAF.
    # Strongly-wrapping surfaces (a closed organic blob) fail the injectivity
    # gate and are left faceted (no regression) — they need the quad-remesh
    # Candidate A pipeline, not a single sheet.

    # Detect and rebuild doubly-curved height-field regions as B-spline sheets.
    # Behind this flag so the whole feature can be disabled if it ever regresses.
    fit_freeform_sheets: bool = True

    # A neighbour facet joins the growing height field only while its normal
    # stays on the +axis side by at least this dot product. Larger keeps the
    # region flatter/smaller (safer injectivity); smaller lets it wrap further.
    freeform_ndot_tol: float = 0.2

    # Minimum facets and surface area (mm^2) a residual region must have to be
    # worth fitting as a sheet — small residuals stay faceted (negligible RTAF).
    freeform_min_facets: int = 40
    freeform_min_area: float = 50.0

    # Injectivity gate: the fraction of region facet area facing *away* from the
    # projection axis (foldover) must stay below this. ~0 is a clean height
    # field; a strongly-curved surface that wraps past its silhouette exceeds it
    # and is deferred (Candidate A territory).
    freeform_max_foldover: float = 0.06

    # Doubly-curved gate: peak-to-peak height (mm, about the region's mean plane)
    # must exceed max(this, freeform_min_curvature_edges * local_edge). A flat
    # residual strip reads ~0 and is left to the planar path, not fit as a sheet.
    freeform_min_curvature: float = 0.3
    freeform_min_curvature_edges: float = 1.0

    # Double-curvature gate: a freeform sheet must bend in BOTH in-plane
    # directions (a paraboloid-like height field), else it is a single-curvature
    # wall the swept detector owns (and de-facets more cheaply). We fit a
    # quadratic height field and require the smaller principal curvature term to
    # be at least this fraction of the peak-to-peak height. A cylinder/sweep has
    # one principal curvature ~0 and is rejected here; a genuine doubly-curved
    # shell passes. Runs BEFORE swept detection so true freeform is claimed first.
    freeform_double_curve_frac: float = 0.08

    # (u,v) grid resolution for resampling the mesh under a freeform region. The
    # B-spline is approximated (not interpolated) from this grid; the fit is
    # rejected if the pole count saturates near the grid size (a signal it could
    # not approximate within tolerance and merely interpolated mesh noise).
    freeform_grid: int = 26

    # Region splitting (task §1): a large cast surface can be locally a height
    # field but curve too much to be ONE clean field — a single B-spline fit
    # would miss it and ship faceted. Splitting is driven at BUILD time by the
    # true B-spline deviation to the real facets (see builder._apply_freeform_
    # sheets): a sheet whose fitted surface misses the mesh is bisected along its
    # dominant curvature ridge and each half re-fitted, up to
    # ``freeform_max_split_depth`` levels (0 -> no split; 2 -> up to 4 sub-sheets).
    # Build-time deviation is the honest trigger — the segmentation-time quadratic
    # residual over-fires on gentle single bumps a quadratic under-models but a
    # B-spline fits perfectly (freeform_bump), so it is NOT used to split.
    freeform_max_split_depth: int = 2

    # Accepted deviation (mm) of the fitted sheet to the region's facets,
    # resolution-scaled: max(this, freeform_dev_tol_rel * local_edge). Above it
    # the fit is rejected and the region left faceted.
    freeform_dev_tol_abs: float = 0.25
    freeform_dev_tol_rel: float = 0.6

    # Reject a sampled region whose (u,v) grid has more than this fraction of
    # cells outside the footprint. With ``freeform_inpaint`` on, missing cells
    # (a ragged boundary, an interior notch, an L-shaped corner) are filled by a
    # smooth Laplace solve from the covered values and the extrapolated skirt is
    # trimmed by the builder's boolean CUT — so a partly-covered region can still
    # fit cleanly. The ceiling stays a guard against a region so sparse its grid
    # is mostly fabricated (below half the cells are real surface).
    freeform_max_missing: float = 0.55

    # Fill (u,v) grid cells outside the region footprint by a discrete Laplace
    # solve from the covered cells (minimal-curvature smooth extension) instead
    # of a nearest-centroid step. The smooth grid fits a clean B-spline; the
    # extrapolated skirt lands past the true surface and the boolean cut trims
    # it. Disable to restore the historical nearest-centroid fallback.
    freeform_inpaint: bool = True

    # A freeform sheet claims its facets (removing them from the swept-wall pool)
    # only when its ``missing`` fraction is at or below this — a well-covered,
    # confident height field the sweep would mis-fit. A marginal / heavily-
    # extrapolated sheet (e.g. a split sub-region with a large inpainted skirt)
    # is still attempted for building but does NOT pre-empt swept walls, which
    # de-facet more reliably; both ops are guarded + RTAF-gated so whichever
    # improves the surface wins. Keeps aggressive region splitting from starving
    # the swept detector (port_cover regression: 44 swept walls -> 0).
    freeform_claim_max_missing: float = 0.30

    # Let a confident freeform sheet PRE-EMPT the swept-wall pool (claim its facets
    # so the swept detector doesn't fragment a genuine doubly-curved bump into
    # empty 0-arc "swept walls" that claim-and-build-nothing, starving the sheet).
    # Necessary for a true freeform bump (freeform_bump: without the claim, swept
    # grabs the 1237-facet dome as 6 arc-less walls and the B-spline never builds).
    # BUT a naive claim regresses prismatic parts whose "freeform" regions are
    # really swept walls the double-curvature detector mis-fired on (insert_cable
    # 0.20 -> 0.40). The two are separated by two gates below:
    # ``freeform_claim_requires_buildable`` (the sheet's own fit must land) AND
    # ``freeform_claim_max_swept_arc_frac`` (the sheet must NOT overlap arc-bearing
    # swept walls that would de-facet it better). With both, a bump claims (no
    # competing arcs) while a mis-detected prismatic panel does not (swept builds
    # its arcs). False disables pre-emption entirely (swept-first, sheets attempted
    # after, RTAF-arbitrated).
    freeform_claim_swept_pool: bool = True

    # Only claim a sheet that will ACTUALLY BUILD — its undivided B-spline fit
    # lands within tolerance on the real facets. Coverage confidence (``missing``)
    # alone is not enough (a well-covered region can still fit a B-spline that
    # misses the mesh), so this pre-fits each candidate and rejects a claim whose
    # deviation exceeds tol. Requires OCC (Part) at claim time.
    freeform_claim_requires_buildable: bool = True

    # Do NOT let a sheet pre-empt the swept pool when arc-bearing swept walls
    # already cover more than this fraction of its facets — those walls de-facet
    # the region more reliably than a single B-spline sheet, and letting the sheet
    # claim them starves the swept build (the insert_cable / countersink / bin_2x1x2
    # P1 regressions: a buildable sheet mis-detected on prismatic swept walls). A
    # genuine freeform bump has ~0 swept-arc coverage (its curvature yields no
    # constant-cross-section arcs) and so still claims. None disables this veto.
    freeform_claim_max_swept_arc_frac: float = 0.15

    # Cost ceiling: the guarded boolean of a doubly-curved sheet against the
    # faceted base is O(base_faces) and can be slow. Skip freeform integration
    # when the base exceeds this many faces (leave the region faceted) rather
    # than grinding. None disables the guard.
    freeform_max_base_faces: int | None = 20000

    # Max freeform boolean attempts per part (largest-area sheets first). Each
    # attempt is an O(base_faces) boolean; this caps the time a part with many
    # small doomed candidates can spend on rejected attempts.
    freeform_max_ops: int = 4

    # --- Organic multi-patch (Candidate A). See
    # docs/ORGANIC_CONVERSION_RESEARCH.md (Candidate A). A genuinely organic body
    # that wraps past any single projection (a sculpted cat, an ergonomic handle,
    # a curved shell) is not a height field, so the freeform-sheet path (Candidate
    # B) declines it and it ships faceted. Candidate A rebuilds the WHOLE body as a
    # quad-patch network: quad-remesh the mesh into a coarse all-quad control cage
    # (pynanoinstantmeshes), least-squares shrink-wrap the cage so its Catmull-Clark
    # LIMIT surface approximates the original mesh, subdivide 1-2x to isolate
    # extraordinary vertices, then extract one exact bicubic B-spline patch per
    # regular quad (Stam 1998) and cap the EV faces — sewing the patches into a
    # shell. Because every face is a B-spline (not a planar strip), the RTAF of a
    # successful result is ~0. Whole-body only for now (no analytic seam); routed
    # by the after-analytic residual-area fraction (see organic_multipatch_min_
    # residual). On ANY failure — remesh unavailable, cage not closed-manifold,
    # shell won't close, or the result doesn't lower RTAF — the pipeline keeps its
    # existing output (never regress).

    # Attempt whole-body organic multi-patch reconstruction. Behind this flag so
    # the whole feature can be disabled. Requires the optional pynanoinstantmeshes
    # dependency; declines gracefully (falls back) when it is unavailable.
    organic_multipatch: bool = True

    # Route to Candidate A only when the after-analytic residual covers at least
    # this fraction of the part's surface area (a mostly-organic body). Below it
    # the part is prismatic-with-features and the analytic + Candidate-B tiers own
    # it; whole-body quad remeshing would destroy their clean analytic faces.
    organic_multipatch_min_residual: float = 0.6

    # Prismatic-signal veto (P0 gate right-sizing). RTAF alone mis-routes a
    # prismatic part whose reconstruction merely failed to SEW into a closed solid:
    # an open shell reads a high RTAF (its faceted chains), so gridfinity_bin_1x1x3
    # (a boxy bin) crossed the residual gate and entered whole-body quad remeshing
    # — where the native remesher ran memory away. But a genuinely-organic body (a
    # sculpted cat) has essentially NO constant-cross-section swept walls, whereas
    # bin_1x1x3 detected 11 (plus spheres). So a reconstruction that found at least
    # this many swept walls is prismatic-with-features and is NOT routed to the
    # whole-body organic tier regardless of RTAF (the region-level pass still runs
    # in the boolean tier and is separately budgeted). None disables the veto.
    organic_multipatch_max_swept_walls: int | None = 6

    # Target quad count for the control cage (before Catmull-Clark subdivision).
    # Coarser = fewer patches + fewer extraordinary vertices (cleaner sew) but
    # higher deviation; the remesher treats this as an edge-length target, so the
    # realised count can differ. Scaled modestly by mesh size in the builder.
    organic_multipatch_target_quads: int = 220

    # Catmull-Clark subdivisions applied to the cage before patch extraction. Each
    # step isolates extraordinary vertices (new vertices are valence-4), shrinking
    # the irregular region geometrically; 1 is enough for most bodies, 2 for
    # heavily-irregular cages. More steps = more patches (4x per step).
    organic_multipatch_subdiv: int = 1

    # Gauss-Newton/projection iterations for the cage shrink-wrap fit (step 3).
    # 0 disables the fit (cage = raw remesh — the limit surface then shrinks
    # inside the mesh); 2-3 lands the limit surface on the original mesh.
    organic_multipatch_fit_iters: int = 3

    # Accepted deviation (mm) of the reconstructed limit surface to the original
    # mesh, resolution-scaled: max(this, this_rel * bbox_diagonal). Above it the
    # multipatch result is rejected and the existing output kept.
    organic_multipatch_dev_tol_abs: float = 1.0
    organic_multipatch_dev_tol_rel: float = 0.02   # 2% of bbox diagonal

    # Skip Candidate A when the input mesh exceeds this many triangles (the quad
    # remesh + per-patch OCC build gets slow on huge scans; decimation upstream
    # usually keeps it well under). None disables the guard.
    organic_multipatch_max_faces: int | None = 200000

    # --- Organic-pass safety budgets (P0 net) ---------------------------------
    # The organic tiers (whole-body multi-patch AND region-level / multi-chart)
    # call the native quad remesher and run subdivision + limit-fit + boolean
    # ladders. On adversarial inputs those can hang, explode memory, or grind for
    # tens of minutes for zero correctness gain (the pass always falls back to the
    # already-good pre-pass solid). These budgets are the safety net: a pass that
    # exceeds them bails and KEEPS the existing solid (never regress, never hang).
    #
    # Wall-clock ceiling for ONE organic pass (whole-body build, or the whole
    # region/multi-chart loop). Checked between attempts/regions; a pass that has
    # already spent this long stops issuing new work and returns what it has. The
    # native remesh itself is additionally hard-bounded by its own subprocess
    # timeout (organic_remesh_timeout) since a wall-clock check can't interrupt a
    # C-extension call. None disables the ceiling.
    organic_pass_time_budget: float | None = 120.0

    # Hard wall-clock timeout (seconds) for a SINGLE native quad-remesh call. The
    # remesher (pynanoinstantmeshes) can hang or run away on memory for some
    # meshes (gridfinity_bin_1x1x3: a 894-facet prepared mesh drove it into a
    # multi-GB runaway). The call runs in a separate subprocess (quadremesh_runner)
    # with this timeout and a memory ceiling; on breach it is killed and the caller
    # treats the remesh as failed (declines the pass). None disables isolation
    # (runs the remesh in-process, unguarded — the old behaviour).
    organic_remesh_timeout: float | None = 45.0

    # Memory ceiling (MB) for the isolated quad-remesh subprocess (RLIMIT_AS). A
    # remesh that tries to allocate past this is killed by the OS instead of
    # exhausting the machine (the bin_1x1x3 runaway climbed past 20 GB). Generous
    # for a legitimate coarse cage (which needs well under a GB) while catching a
    # blow-up. None leaves the subprocess uncapped.
    organic_remesh_max_memory_mb: int | None = 4096

    # Hard wall-clock timeout (seconds) for a single organic-tier boolean CUT
    # (freeform-sheet integration and organic-region integration). A B-spline tool
    # cut against a faceted base can send OCC's face-face intersection into a
    # pathological multi-minute grind even on a valid, small tool (the usb_holder
    # freeform P0 hang) — and the call is uninterruptible in-process, so a caller
    # wall-clock budget can't stop it. When set, these cuts run in a separate
    # process (boolean_runner) under this timeout; on breach the op is treated as
    # failed and reverted (the region/sheet stays faceted — never regress, never
    # hang). The trade-off: a fresh child must import FreeCAD before the cut, which
    # costs tens of seconds PER op, so this is only worth paying when the cut is
    # known-risky (see ``organic_boolean_isolate_min_base_faces`` — the cut is
    # isolated only on a base large enough for the pathological grind, and run
    # in-process on small bases where it is always fast). None disables isolation
    # entirely (all cuts in-process, unguarded — the old, hang-prone behaviour).
    organic_boolean_timeout: float | None = 90.0

    # Only pay the subprocess-isolation cost for an organic-tier boolean CUT when
    # the base solid has at least this many faces — the pathological multi-minute
    # grind is a dense-base phenomenon (usb_holder's freeform cut ground on its
    # boolean base), while a cut against a small base is always sub-second and is
    # far cheaper to run in-process than to isolate (isolation adds a FreeCAD
    # cold-import per op). Below this the cut runs in-process (fast, unisolated);
    # at or above it the cut is isolated under ``organic_boolean_timeout``. None
    # isolates every cut (when a timeout is set) regardless of base size.
    organic_boolean_isolate_min_base_faces: int | None = 3000

    # Cap the whole-body organic subdivision escalation by the resulting quad
    # count: each Catmull-Clark step is 4x the faces, and the escalation ladder
    # (subdiv .. subdiv+3) can reach thousands of quads on a cage that never
    # closes, paying a large per-level shell build (one Stam B-spline patch + EV
    # cap per quad) AND a per-face tessellation for the deviation/RTAF checks —
    # both uninterruptible OCC calls that a wall-clock budget can't stop mid-call
    # (animal_figurine hung in BRepMesh tessellating a 3952-quad shell). A
    # subdivision level whose quad count would exceed this is not attempted (the
    # escalation stops early; if no level is small enough the pass declines and
    # keeps the existing solid). 2500 admits the corpus's real organic cages (dog
    # ~250, cat ~1000 at the level they close) while stopping the pathological
    # escalation before it grinds. The whole-body tier has never ADOPTED on the
    # corpus (it always declines after a strict RTAF/bbox gate), so a tight cap
    # costs no real reconstruction. None disables the cap.
    organic_multipatch_max_subdiv_quads: int | None = 2500

    # --- Region-level Candidate A (organic patch regions). See
    # docs/ORGANIC_CONVERSION_RESEARCH.md (Candidate A, region-level). The
    # whole-body organic tier only fires on a body that is ORGANIC in its entirety
    # (a sculpted cat). Real engineering parts mix analytic features with ONE (or a
    # few) large residual organic region(s) — the port_cover cast top, a curved
    # cast pad — that neither the analytic tiers, nor the freeform height-field
    # sheet (Candidate B, which requires an injective projection), nor whole-body
    # remeshing (the whole part isn't a closed organic body) can claim, so the
    # region ships faceted (high RTAF). This pass rebuilds each such region as a
    # Catmull-Clark patch network (quad-remesh the OPEN region -> fit the cage's
    # limit surface to the region facets -> exact bicubic Stam patches) and
    # integrates it with the analytic base by the SAME guarded extrude+cut boolean
    # the freeform sheet uses (organic patch shell -> extrude along the region axis
    # into a tool -> cut, so the shell's smooth patches become the new surface).
    # Runs AFTER the freeform sheet pass in the boolean-clean tier, on whatever
    # residual it left. STRICT rollback: each region is adopted only when its
    # boolean validates, stays bbox-stable, and LOWERS the RTAF; otherwise the
    # region keeps its faceted output (never regress).

    # Attempt the region-level organic patch pass. Behind this flag so it can be
    # disabled. Requires the optional pynanoinstantmeshes remesher + OCC bindings;
    # declines gracefully (regions stay faceted) when either is unavailable.
    organic_region_patches: bool = True

    # A residual smooth region must have at least this many facets (and this
    # surface area, mm^2) to be worth rebuilding — smaller residuals stay faceted
    # (negligible RTAF, and a tiny cage remeshes poorly).
    organic_region_min_facets: int = 300
    organic_region_min_area: float = 200.0

    # Reject a near-FLAT region: an organic patch shell is only worth building (and
    # only produces a well-behaved boolean tool) when the region genuinely curves.
    # A large flat/gently-warped panel (the fan_panel residual: a 269 mm-diagonal
    # face that reads foldover 0 and fits a 2 mm-deviation B-spline) is NOT organic
    # — it is a planar face the sew/planar path owns, and extruding its huge flat
    # B-spline into a boolean tool grinds OCC for minutes (the fan_panel P0 hang).
    # Require the region's peak-to-peak height about its mean plane to be at least
    # this fraction of its in-plane diagonal; below it the region is flat and is NOT
    # routed to the organic patch pass (stays faceted/planar). None disables the
    # gate. Calibrated: a genuine cast top (port_cover, tweezer shell) curves well
    # past 3% of its span; a flat panel is ~0.
    organic_region_min_curve_frac: float | None = 0.03

    # Reject a region whose foldover (facet-area fraction facing away from its mean
    # normal) exceeds this. A region must project INJECTIVELY along its mean normal
    # for its single fitted B-spline surface (and the extruded boolean tool) to be
    # non-self-intersecting — a wrapping region folds the (u,v) projection, giving a
    # degenerate surface and a pathological cut. So this stays tight (near the
    # freeform gate). The pass's value over freeform on these injective regions is
    # the Catmull-Clark limit SAMPLER: it denoises the mesh into a smooth dense grid,
    # so a region whose raw-mesh B-spline fit freeform rejects on deviation can still
    # be reconstructed here from the clean limit surface.
    organic_region_max_foldover: float = 0.08

    # Target quad count for a region's control cage (edge-length target; the
    # realised count differs). Scaled up modestly for larger regions in the builder.
    organic_region_target_quads: int = 400

    # Catmull-Clark subdivisions of the region cage before the limit-fit. One step
    # densifies the limit-point sample enough for a clean single-surface fit.
    organic_region_subdiv: int = 1

    # (u,v) grid resolution for resampling the region's Catmull-Clark limit points
    # into ONE B-spline surface. The limit cloud is smooth + dense, so a moderate
    # grid captures it; the fit is rejected if the pole count saturates (a signal it
    # interpolated noise instead of approximating).
    organic_region_grid: int = 30

    # Projection iterations for the region cage shrink-wrap fit (applied to the
    # already-subdivided dense cage — see organic_region.build_region_patch_faces).
    # 0 disables (the limit surface then shrinks inside the region); ~6 lands the
    # dense limit surface on the mesh to sub-mm on the target corpus.
    organic_region_fit_iters: int = 6

    # Accepted deviation of the region's fitted surface to its real facets. The
    # tolerance is max(abs, rel*local_edge, diag_frac*region_diagonal): the
    # edge-scaled term recovers correct fits on a coarse STL (chord sagitta scales
    # with edge length), and the diagonal-fraction term (like the whole-body
    # organic tier's 2%-of-diag gate) admits a resolution-honest fit on a large
    # cast region. Above it the region's boolean is not attempted (the surface
    # misses the mesh -> would distort the part).
    organic_region_dev_tol_abs: float = 0.6
    organic_region_dev_tol_rel: float = 2.0
    organic_region_dev_tol_diag_frac: float = 0.02

    # Cap on the number of region booleans attempted per part (each is O(base
    # faces)); and skip the pass when the base solid exceeds this many faces (the
    # boolean of a dense patch tool against a dense base is slow).
    organic_region_max_ops: int = 6
    organic_region_max_base_faces: int | None = 40000

    # --- Multi-chart parametrization (organic_region.region_charts) ------------
    # A thin WRAPPING residual shell (the port_cover cast top, the tweezer shell,
    # the patton_pad residual) has no single injective projection — its facet
    # normals span far past a hemisphere, so the single-surface region pass rejects
    # it (surface folds; extrude tool self-intersects). When enabled, such a region
    # is split into connected SINGLE-SIDED charts, each of which IS individually
    # parametrizable, and each chart is reconstructed + cut in by the same guarded
    # extrude+cut boolean the single-region pass uses (processed sequentially, each
    # revertable). Behind this flag so it can be disabled independently.
    organic_region_multichart: bool = True

    # A chart admits a connected region facet only while its normal is within this
    # many degrees of BOTH the chart axis and the fixed seed normal — the injectivity
    # ("stop at the fold") criterion, seed-anchored so each chart is a COMPACT
    # geodesic cap (not an annular ring that wraps around the axis and never projects
    # single-valued). ~50 deg keeps each cap comfortably inside the silhouette so its
    # single fitted B-spline surface and the extruded boolean tool do not
    # self-intersect (measured: the open Catmull-Clark limit-fit balloons a
    # near-hemisphere chart, so the cap must stay well under 90 deg).
    organic_region_chart_half_angle: float = 50.0

    # A chart must have at least this many facets (and this surface area, mm^2, or
    # this fraction of the parent region's area) to be worth reconstructing; smaller
    # charts stay faceted (negligible RTAF, and a tiny cage remeshes poorly).
    organic_region_chart_min_facets: int = 300
    organic_region_chart_min_area: float = 150.0
    organic_region_chart_min_area_frac: float = 0.02

    # Cap on charts built per region (each is a full remesh + fit + boolean). A
    # wrapping shell is 2-8 natural charts; beyond this the region is left faceted.
    organic_region_chart_max: int = 8

    # A region is only routed to the multi-chart pass when its foldover is at least
    # this high (i.e. it genuinely wraps) — below it the single-surface region pass
    # already claims it (or correctly declines a gentle residual). Keeps the extra
    # remesh/boolean cost off the injective regions the single pass handles.
    organic_region_multichart_min_foldover: float = 0.02

    # A wrapping region must have at least this many facets total to bother splitting
    # (the split + per-chart remesh is only worth it on a large residual).
    organic_region_multichart_min_facets: int = 900

    # --- Spheres: domes and corner blends (M3). See docs/CURVED_FEATURES.md §3,
    # §4. A dome (grille cap, rounded boss top) or the spherical blend where three
    # fillets meet is a spherical cap: its facet normals fan out in all directions
    # (_region_axis returns None) and its vertices lie on one sphere. We fit the
    # 4-parameter linear sphere, sagitta-bias-correct, gate, and snap to adjacent
    # flats when near-tangent. Domes that tessellate into many strips are routed
    # here via a cross-region sphere consensus BEFORE swept-wall fitting.

    # Detect and rebuild spherical caps / corner blends. Behind this flag so the
    # whole feature can be disabled if it ever regresses a part.
    detect_spheres: bool = True

    # Roll back a sphere cut/fuse that does not lower the RTAF (residual
    # tessellation). This is the false-positive net that lets the shallow-cap
    # fuse volume guard be relaxed (task §2): a mis-detected / bulging cap that
    # slips past the volume + bbox guards but doesn't actually de-facet the
    # surface is reverted, so only genuine caps are adopted.
    sphere_rtaf_gate: bool = True

    # --- Local deviation guard (task §3): a permanent net for the "visual
    # artifact" sphere ops. RTAF is a GLOBAL area metric — a small bogus cap next
    # to a fillet, or a dome bulging a few mm out of a large flat, barely moves the
    # aggregate RTAF yet is a glaring wrong bump on the real part (the port_cover /
    # patton P0s: the worst deviation outliers were all next to freshly-built
    # spheres). After each sphere boolean op we therefore sample points ON the
    # candidate's newly-built cap surface near the op's affected region and measure
    # their distance to the INPUT mesh (point-to-triangle, the webapp meshdata
    # pattern). An op whose surface deviates from the mesh by more than a
    # resolution-scaled threshold — i.e. it introduced geometry the mesh never had
    # — is reverted, leaving the cap faceted (Nick's principle: better faceted than
    # a wrong bump). This localises what the global RTAF gate cannot see. Freeform
    # sheets and organic regions are NOT re-checked here: they already gate each
    # fit on its true surface-to-mesh deviation before the boolean; spheres do not.
    local_deviation_guard: bool = True
    # Absolute floor (mm) and edge-relative slope for the allowed surface-to-mesh
    # deviation of a freshly-built analytic op. The tool surface (sphere cap /
    # B-spline sheet) is inscribed in / snapped to the faceted mesh, so a correct
    # op lands within a fraction of the local chord sagitta; a bulging false
    # positive lands mm off. tol = max(abs, rel * local_edge), then the op must not
    # exceed max(tol, prior_local_deviation) — never punished for pre-existing
    # facet error, only for making the surface WORSE than the mesh it replaced.
    local_deviation_max_abs: float = 0.6
    local_deviation_max_rel: float = 1.5
    # Number of sample points drawn on the candidate op surface for the check
    # (kept small — this runs per successful op; the KD-tree query dominates).
    local_deviation_samples: int = 200

    # Minimum facets a compact smooth region must have to attempt a sphere fit.
    min_sphere_facets: int = 8

    # Reject fitted spheres below this radius (mm) — sub-facet blobs on organic
    # surfaces fit tiny spheres. The effective floor is the larger of this and
    # min_sphere_radius_edges * local_edge (a sphere can't be finer than a facet).
    min_sphere_radius: float = 0.5
    min_sphere_radius_edges: float = 0.5

    # Reject fitted radii larger than this (mm). None -> derived from the part
    # size (see max_sphere_radius_frac).
    max_sphere_radius: float | None = None

    # When max_sphere_radius is None, cap the fitted radius at this fraction of
    # the part's largest bounding-box dimension. A dome / corner blend is a
    # feature ON the part, well under its overall size; a spurious fit over an
    # organic / vase-mode wall lands at a near-part-size sphere. domed_plate's
    # R=20 on a 60 mm part is 0.33; the vase's bogus R≈87 on a 130 mm part is
    # 0.67 — so 0.45 admits real caps while dropping the organic false positives.
    max_sphere_radius_frac: float = 0.45

    # Minimum solid-angle fraction (0..1) of the sphere a cap must span. Rejects
    # sliver clusters that algebraically fit a huge sphere but barely wrap it.
    # A shallow but genuine dome covers ~0.15 (domed_plate), so this stays low;
    # the organic-wall false positive (a vase ring-stack fitting a bogus
    # part-size sphere) is dropped by the radius-vs-part-size cap above, not here.
    min_sphere_coverage: float = 0.03

    # Per-part sphere boolean-op cost budget (mirrors ``swept_op_budget``). Each
    # analytic-sphere cut/fuse is a boolean against the base solid, cost
    # ~O(base_faces); with a deep BOP self-intersection re-check on the result a
    # single op can take tens of seconds on a DENSE base (a mostly-organic body —
    # a scanned tank hull — that decimation could not shrink, or a config that
    # skipped decimation, leaves a ~200k-face base where each op grinds and the
    # M3 pass can appear hung, e.g. "spheres cleaned 7/8" stalling for minutes).
    # When ``spheres × base_faces`` exceeds this budget the sphere ops are skipped
    # WHOLESALE (those caps stay faceted) rather than hanging — a graceful
    # per-feature degradation that never costs the watertight solid or the other
    # analytic features. Set well ABOVE the corpus's real domed parts (domed_plate
    # ~9 spheres × ~12k base = 108k) so it never vetoes a legitimate reconstruction
    # while still catching a runaway (9 × 200k = 1.8M). None disables the budget.
    sphere_op_budget: int | None = 1_500_000

    # --- Dome routing via cross-region sphere consensus (task §3). A tessellated
    # dome segments into many thin planar strips; no single strip reads as a
    # sphere (the per-region gate fails), but many strips share one (centre, R).
    # We fit each candidate region, cluster by (centre, R), and a dominant cluster
    # is a dome — routed to the sphere detector BEFORE swept-wall fitting so M4
    # never wastes minutes fitting doomed lens ops to a dome's latitude rows.

    # Minimum facets a strip must have to join the consensus vote.
    sphere_consensus_min_region_facets: int = 3
    # A strip joins the vote only if its own sphere fit RMS is within this
    # multiple of the surface tolerance (loose — the clustering confirms the dome).
    sphere_consensus_rms_mult: float = 4.0
    # Minimum candidate strips overall, and minimum members in a cluster, to
    # accept a dome (many regions sharing one sphere is the dome signature).
    sphere_consensus_min_regions: int = 4
    # Two strips share a dome when their fitted radii agree within this relative
    # fraction and their centres within this fraction of the radius (+ mm floor).
    sphere_consensus_radius_rel: float = 0.06
    sphere_consensus_center_rel: float = 0.06

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

    # Accepted values for ``multibody_mode``; also used by the CLI/GUI choices.
    MULTIBODY_MODES = ("auto", "combine", "separate")

    def __post_init__(self) -> None:
        if self.multibody_mode not in self.MULTIBODY_MODES:
            raise ValueError(
                f"unknown multibody_mode {self.multibody_mode!r}; "
                f"expected one of {list(self.MULTIBODY_MODES)}"
            )

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
