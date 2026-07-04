"""FreeCAD/OpenCASCADE geometry construction and STEP export.

This is the only module that imports FreeCAD. It rebuilds the mesh as real CAD
geometry — planar facets merged into single faces, and detected cylindrical
regions rebuilt as analytic cylinder faces with true circular edges — then sews
everything into a solid. A faceted fallback always yields a watertight result.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .boundary import FaceLoops, extract_face_loops
from .config import ConversionConfig
from .fitting import (
    Cylinder,
    FreeformSheet,
    KnurlBand,
    Sphere,
    SweptProfile,
    Thread,
    _connected_components,
    detect_cones,
    detect_cylinders,
    detect_fillets_straight,
    detect_knurling,
    detect_spheres,
    detect_swept_walls,
    detect_threads,
    fit_freeform_sheets,
    sphere_consensus_regions,
)
from .segmentation import (
    _axis_basis,
    build_edge_adjacency,
    face_normals_and_areas,
    mesh_resolution,
    sample_freeform_grid,
    segment_planar,
    segment_smooth_bands,
    segment_swept_walls,
)


def _vec(p):
    import FreeCAD  # type: ignore  # local import; module only runs under FreeCAD

    return FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))


def _loop_degenerate(points3d, tol: float = 1e-6) -> str | None:
    """Pre-validate a boundary loop before it reaches an OCC wire/face constructor.

    Some degenerate loops make OCC's polygon/plane/face constructors raise a
    *native* ``Standard_ConstructionError`` that abort()s the process instead of
    surfacing as a catchable Python exception — no ``except Exception:`` around
    the constructor can save the worker. The only safe fix is to reject the bad
    loop in pure Python *before* calling into OCC.

    Returns a short reason string if the loop is degenerate (caller skips it and
    logs), or ``None`` if it is safe to build. Rejects:
      * fewer than 3 points, or fewer than 3 *distinct* points (within ``tol``);
      * consecutive duplicate points (zero-length edges);
      * a near-zero coordinate span (all points coincide);
      * a collinear loop (zero enclosed area / rank < 2) — this is the case that
        aborts ``Part.Face``/``Part.Plane`` on a smooth wall misclassified into
        thin strips, since a line has no plane to build a face on.
    """
    pts = np.asarray(points3d, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return "fewer than 3 points"
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if span < tol:
        return "zero coordinate span"
    # Consecutive duplicates (including wrap-around) -> zero-length edges.
    edges = pts - np.roll(pts, -1, axis=0)
    if float(np.linalg.norm(edges, axis=1).min()) < tol:
        return "consecutive duplicate points"
    # Distinct-point count (rounded to tol grid).
    scale = max(span, 1.0)
    quant = np.round(pts / (tol * scale)).astype(np.int64)
    if len({tuple(row) for row in quant}) < 3:
        return "fewer than 3 distinct points"
    # Collinearity / zero enclosed area: SVD rank of the centred points. A loop
    # whose points all lie on one line has a single dominant singular value; OCC
    # cannot infer a plane and aborts. Compare the 2nd singular value to the 1st.
    centred = pts - pts.mean(axis=0)
    try:
        sv = np.linalg.svd(centred, compute_uv=False)
    except np.linalg.LinAlgError:
        return "SVD failed (degenerate loop)"
    if sv.size < 2 or sv[0] <= tol or sv[1] < 1e-6 * sv[0]:
        return "collinear loop (zero area)"
    return None


def _wire_from_points(points3d: np.ndarray, Part):
    """Build a closed polygonal wire from ordered 3D points."""
    vectors = [_vec(p) for p in points3d]
    vectors.append(vectors[0])  # close the loop
    return Part.makePolygon(vectors)


def _circle_wire(center, normal, radius: float, Part):
    """Build a true circular wire (one analytic edge)."""
    circle = Part.Circle(_vec(center), _vec(normal), float(radius))
    return Part.Wire([circle.toShape()])


def _analytic_circles(cylinders, cones):
    """End-circles of every analytic face, as (center, axis, radius) tuples.

    Planar boundary loops that match one of these get replaced by the exact
    circle so their edges coincide with the analytic faces and sew cleanly.
    """
    circles = []
    for cyl in cylinders:
        for za in (cyl.axial_min, cyl.axial_max):
            circles.append((cyl.axis_point + za * cyl.axis_dir, cyl.axis_dir, cyl.radius))
    for cone in cones:
        circles.append((cone.axis_point + cone.axial_min * cone.axis_dir, cone.axis_dir, cone.r_base))
        circles.append((cone.axis_point + cone.axial_max * cone.axis_dir, cone.axis_dir, cone.r_top))
    return [(c, a, r) for (c, a, r) in circles if r > 1e-6]


def _match_loop_to_circle(loop: np.ndarray, plane_normal: np.ndarray, circles):
    """If a boundary loop is a faceted circle matching an analytic end-circle,
    return the exact (center, normal, radius) to replace it with; else None."""
    centroid = loop.mean(axis=0)
    mean_radius = float(np.linalg.norm(loop - centroid, axis=1).mean())
    for center0, axis, radius in circles:
        if abs(float(plane_normal @ axis)) < 0.99:  # plane must be ⊥ axis
            continue
        if abs(mean_radius - radius) > 0.05 * radius + 0.1:
            continue
        rel = centroid - center0
        off_axis = np.linalg.norm(rel - (rel @ axis) * axis)
        if off_axis > 0.05 * radius + 0.1:  # loop centred on the axis line
            continue
        denom = float(axis @ plane_normal)
        if abs(denom) < 1e-9:
            continue
        # Slide center0 along the axis onto the loop's plane (axis ∩ plane).
        s = float((centroid - center0) @ plane_normal) / denom
        center = center0 + s * axis
        return center, plane_normal, radius
    return None


def _planar_face(loops: FaceLoops, circles, Part):
    """Build a planar face, swapping any faceted-circle loop for a true circle.

    Hole loops are wound opposite to the outer loop so OCC subtracts them; when
    a loop is replaced by an exact circle we flip its axis to match.

    ``segment_planar`` admits facets within ``dist_tol`` (0.01 mm) of the fitted
    plane, so a region's boundary-loop vertices can sit a few microns off any
    single plane. ``Part.Face(wires)`` *infers* the plane from the wire and
    rejects those loops with ``OCCError: Not planar`` even though they are
    genuinely planar to well under a micron of design intent (measured max
    deviation 0.0074 mm on the corpus). That single failure used to drop the
    whole region to faceted gap-fill — ~20% of all residual facets, and enough
    open regions to stop the sew tier ever closing on grille/slot parts.

    Fix: when the inference path fails, rebuild the face on the *explicit* fitted
    plane (``Part.Face(plane, wires)``) — OCC no longer has to infer planarity,
    it just projects the wire onto the given surface. Crucially this keeps the
    original 3-D wire vertices (shared with neighbouring faces at the sew tier),
    so **no vertex moves** and the sewing gaps that plain projection would open
    (deviation up to ``dist_tol`` 0.01 mm > ``sew_tolerance`` 0.001 mm) never
    appear. The explicit-plane face reads ``isValid()==False`` at OCC's default
    1e-7 tolerance (the wire sits microns off the surface), so ``.fix()`` widens
    only the face/edge *tolerance* to absorb that deviation — it still moves no
    vertices — making the face valid for sewing.
    """
    normal = loops.normal

    def wire_for(loop, is_hole):
        match = _match_loop_to_circle(loop, normal, circles)
        if match is not None:
            center, n, radius = match
            return _circle_wire(center, -n if is_hole else n, radius, Part)
        return _wire_from_points(loop, Part)

    # Degenerate-loop guard (see _loop_degenerate): a collinear/zero-area outer
    # loop makes Part.Face/Part.Plane abort() natively — uncatchable from Python.
    # Reject it here as an ordinary exception so the caller drops the region to
    # gap-fill instead of the whole worker crashing. Circle-matched loops are
    # exact analytic circles and safe regardless.
    if _match_loop_to_circle(loops.outer, normal, circles) is None:
        reason = _loop_degenerate(loops.outer)
        if reason is not None:
            raise RuntimeError(f"degenerate outer loop ({reason})")

    wires = [wire_for(loops.outer, is_hole=False)]
    for hole in loops.holes:
        if len(hole) < 3:
            continue
        if _match_loop_to_circle(hole, normal, circles) is None \
                and _loop_degenerate(hole) is not None:
            continue  # skip a degenerate hole loop rather than abort in OCC
        wires.append(wire_for(hole, is_hole=True))

    try:
        # Fast path: OCC infers the plane from the wire. Works for the vast
        # majority of regions (loops that lie on one plane to ~1e-7).
        face = Part.Face(wires)
        if face.isValid() and face.Area > 1e-9:
            return face
    except Exception:  # noqa: BLE001 - OCCError: Not planar (and friends)
        pass

    # Robust path: build on the explicit fitted plane, then widen the face
    # tolerance so OCC accepts the wire that sits a few microns off it. Vertices
    # are untouched, so sewing with neighbours is unaffected.
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    centroid = np.asarray(loops.outer, float).mean(axis=0)
    plane = Part.Plane(_vec(centroid), _vec(n))
    face = Part.Face(plane, wires)
    if not face.isValid():
        # Absorb the sub-dist_tol wire-vs-surface deviation into the face/edge
        # tolerance (does not move vertices); sew_tolerance still bridges the
        # shared edges between faces.
        face = face.copy()
        try:
            face.fix(1e-3, 1e-3, 1e-3)
        except Exception:  # noqa: BLE001 - fix is best-effort validation
            pass
    if face.Area <= 1e-9:
        raise RuntimeError("planar face built with zero area")
    return face


def _cylinder_face(cyl: Cylinder, Part):
    """Build an analytic cylindrical face trimmed to the region's axial span.

    A boss keeps the surface's natural outward normal; a hole is reversed so the
    solid's outward normal points into the material (out of the bore). A fillet
    is a partial arc: only its ``[u_start, u_start+u_span]`` angular sector is
    built (a full cylinder uses the whole 0..2pi range).
    """
    surf = Part.Cylinder()
    surf.Center = _vec(cyl.axis_point)
    surf.Axis = _vec(cyl.axis_dir)
    surf.Radius = float(cyl.radius)
    if cyl.is_fillet:
        u0 = float(cyl.u_start)
        u1 = u0 + float(cyl.u_span)
    else:
        u0, u1 = 0.0, 2.0 * math.pi
    face = surf.toShape(u0, u1, cyl.axial_min, cyl.axial_max)
    return face if cyl.outward else face.reversed()


def _faceted_faces(all_points: list, tri_faces: np.ndarray, Part):
    """Build merged faces for one local patch of facets.

    Wraps the patch as a raw mesh shape, then ``removeSplitter`` coalesces
    adjacent coplanar triangles within the patch into single faces — so a flat
    region inside an otherwise-unmergeable pocket still collapses to one face,
    and only genuine curvature stays faceted. ``all_points`` is the full
    pre-built FreeCAD Vector list for the mesh (shared across patches so it is
    only built once); ``tri_faces`` indexes into it.
    """
    topo = [(int(a), int(b), int(c)) for a, b, c in tri_faces]
    shape = Part.Shape()
    shape.makeShapeFromMesh((all_points, topo), 0.1)
    try:
        shape = shape.removeSplitter()
    except Exception:  # noqa: BLE001 - merge is an optimization, not required
        pass
    return list(shape.Faces)


def _cone_face(cone, Part):
    """Build an analytic conical face (countersink) from makeCone's lateral face."""
    base = cone.axis_point + cone.axial_min * cone.axis_dir
    height = cone.axial_max - cone.axial_min
    solid = Part.makeCone(float(cone.r_base), float(cone.r_top), float(height),
                          _vec(base), _vec(cone.axis_dir))
    lateral = [f for f in solid.Faces if f.Surface.TypeId == "Part::GeomCone"][0]
    return lateral if cone.outward else lateral.reversed()


# --------------------------------------------------------------------------- #
# Swept / extruded curved walls (Milestone 4). Each fitted profile segment
# becomes one extruded face: a line extrudes to a plane, an arc to an analytic
# cylinder sector, a spline to an extruded-B-spline surface — replacing the fan
# of thin planar strips the tessellation shipped.
# --------------------------------------------------------------------------- #


def _arc_mid_point2(seg) -> np.ndarray:
    """2D midpoint ON the fitted arc, halfway along its traversal direction."""
    c, r = seg.center, seg.radius
    a0 = math.atan2(seg.p0[1] - c[1], seg.p0[0] - c[0])
    a1 = math.atan2(seg.p1[1] - c[1], seg.p1[0] - c[0])
    if seg.ccw:
        while a1 <= a0:
            a1 += 2.0 * math.pi
    else:
        while a1 >= a0:
            a1 -= 2.0 * math.pi
    am = 0.5 * (a0 + a1)
    return np.array([c[0] + r * math.cos(am), c[1] + r * math.sin(am)])


def _apply_sphere_ball_ops(solid, spheres, Part, progress, *, bbox_guard,
                           config: ConversionConfig | None = None) -> tuple:
    """Apply the analytic-sphere cut/fuse ops with a cost budget + per-op revert.

    Shared by both tiers. Each op is a boolean against ``solid`` (cost
    ~O(base_faces)); on a DENSE base a single op — plus its deep BOP re-check —
    can grind for tens of seconds, so the whole M3 pass can appear to hang
    ("spheres cleaned 7/8" stalling). When ``spheres × base_faces`` exceeds
    ``config.sphere_op_budget`` the batch is SKIPPED wholesale (caps stay
    faceted) — a graceful per-feature degradation that never costs the watertight
    solid or the other analytic features. Otherwise every op goes through
    ``_try_boolean_step`` so a bad/slow-to-fail sphere reverts JUST itself and the
    rest of the reconstruction is kept analytic. Returns ``(solid, built)``.
    """
    if not spheres:
        return solid, 0
    budget = config.sphere_op_budget if config is not None else None
    if budget is not None:
        try:
            base_faces = len(solid.Faces)
        except Exception:  # noqa: BLE001
            base_faces = 0
        cost = len(spheres) * base_faces
        if base_faces and cost > budget:
            progress(f"  sphere ops skipped: cost {len(spheres)} spheres × "
                     f"{base_faces} faces = {cost:,} exceeds budget {budget:,} "
                     f"(domes/blends left faceted)")
            return solid, 0
    built = 0
    # RTAF-regression gate (task §2): a sphere op that makes the residual
    # tessellation WORSE — a mis-detected / bulging cap that the relaxed
    # shallow-cap volume guard now lets through, which fractures the surface into
    # more chains — is rolled back. Note this is a REGRESSION gate, not an
    # improvement gate: a legitimate low-coverage dome/dish (RMS-tight, tangent to
    # its flats) that trues up a small cap may leave the aggregate RTAF flat
    # (its area is tiny beside a part's dominant surface) yet is still correct
    # geometry to adopt. Requiring strict improvement wrongly reverted 4/5 real
    # port_cover caps; requiring only "no regression" keeps them. The per-op
    # validity + bbox guards remain the primary false-positive net.
    rtaf_gate = config is not None and getattr(config, "sphere_rtaf_gate", True)
    rtaf_before = None
    if rtaf_gate:
        try:
            rtaf_before = compute_rtaf(solid, config).get("rtaf")
        except Exception:  # noqa: BLE001
            rtaf_before = None
    for sph in spheres:
        candidate, ok = _try_boolean_step(
            solid, lambda s, sp=sph: _boolean_clean_sphere(s, sp, Part),
            max_bbox_growth=bbox_guard)
        if not ok:
            continue
        if rtaf_gate and rtaf_before is not None:
            try:
                rtaf_after = compute_rtaf(candidate, config).get("rtaf")
            except Exception:  # noqa: BLE001
                rtaf_after = None
            if rtaf_after is not None and rtaf_after > rtaf_before + 1e-3:
                progress(f"  sphere reverted: RTAF {rtaf_before:.3f} -> "
                         f"{rtaf_after:.3f} (regressed)")
                continue
            if rtaf_after is not None:
                rtaf_before = rtaf_after
        solid = candidate
        built += 1
    return solid, built


def _apply_swept_lens_ops(solid, profiles: list[SweptProfile], Part, progress,
                          max_ops: int = 200, config: ConversionConfig | None = None) -> tuple:
    """Apply one boolean lens op per fitted swept-arc segment to a valid solid.

    Shared by both tiers: the sew tier runs it on the closed reconstructed
    solid, the boolean tier on the faceted base. Cut for concave walls, guarded
    fuse for convex; every op goes through ``_try_boolean_step`` so a bad lens
    reverts and can never cost watertightness. Wide arcs are skipped (their
    chord<->arc lens grows fat enough to reach unrelated geometry), as is
    anything beyond the op budget on pathological meshes.

    Cost budget (M4 gear regression): each op is a boolean against the current
    solid, cost ~O(faces); the whole batch is O(distinct_arcs × faces). When that
    product exceeds ``config.swept_op_budget`` the batch is SKIPPED wholesale
    (walls stay faceted) rather than grinding for minutes — the repeated-arc
    guard already drops gear-tooth profiles, this catches any residual blow-up.

    Returns ``(solid, ops_attempted, ops_succeeded)``.
    """
    try:
        bb0 = solid.BoundBox
        bounds0 = (bb0.XMin, bb0.YMin, bb0.ZMin, bb0.XMax, bb0.YMax, bb0.ZMax)
    except Exception:  # noqa: BLE001
        bounds0 = None

    def guarded(s, p, g, deep):
        candidate = _boolean_clean_swept(s, p, g, Part)
        # Bounding-box guard: a lens op trues an existing wall, so it can never
        # legitimately grow the part beyond the chord-sagitta scale. A fuse
        # whose material-side classification was wrong bulges outward instead —
        # a valid solid the volume guard alone can miss on fat lenses.
        if bounds0 is not None:
            bb = candidate.BoundBox
            grow = max(bounds0[0] - bb.XMin, bounds0[1] - bb.YMin,
                       bounds0[2] - bb.ZMin, bb.XMax - bounds0[3],
                       bb.YMax - bounds0[4], bb.ZMax - bounds0[5])
            if grow > 0.2:
                raise ValueError(f"swept lens op rejected: bbox grew {grow:.2f} mm")
        if deep:
            _check_no_self_intersection(candidate)
        return candidate

    # Distinct arc ops (duplicate rails — the min/max copies of the same wall
    # profile — fit identical arcs; one op per distinct arc suffices).
    todo: list[tuple] = []
    seen: set[tuple] = set()
    for prof in profiles:
        for seg in prof.segments:
            if seg.kind != "arc" or seg.outward is None:
                continue
            c3 = prof.point3d(seg.center)
            key = (tuple(np.round(c3, 2)), tuple(np.round(prof.axis, 2)),
                   round(float(seg.radius), 2),
                   round(math.degrees(_arc_span_rad(seg)), 0), bool(seg.outward))
            if key in seen:
                continue
            seen.add(key)
            todo.append((prof, seg))
    todo = todo[:max_ops]
    if not todo:
        return solid, 0, 0

    # Cost budget: distinct_arcs × base_faces. Skip the whole batch when it would
    # grind (a gear's hundreds of tooth arcs on a dense base). The repeated-arc
    # guard in detect_swept_walls already drops the classic gear profile; this is
    # the belt-and-braces ceiling for any residual blow-up.
    budget = config.swept_op_budget if config is not None else None
    if budget is not None:
        try:
            base_faces = len(solid.Faces)
        except Exception:  # noqa: BLE001
            base_faces = 0
        cost = len(todo) * base_faces
        if base_faces and cost > budget:
            progress(f"  swept lens ops skipped: cost {len(todo)} arcs × "
                     f"{base_faces} faces = {cost:,} exceeds budget {budget:,} "
                     f"(walls left faceted)")
            return solid, len(todo), 0

    def run(start_solid, deep):
        s = start_solid
        n_ok = 0
        for prof, seg in todo:
            s, ok = _try_boolean_step(
                s, lambda cur, p=prof, g=seg: guarded(cur, p, g, deep))
            n_ok += ok
        return s, n_ok

    # Fast pass with cheap per-op guards, then ONE deep BOP self-intersection
    # check of the final result: OCC booleans occasionally produce a shape that
    # passes isValid() in memory yet re-reads invalid from the exported STEP
    # (seen on a convex lens fuse). The deep check catches those, but at ~1 s a
    # call it must not run per op in the common case — so it runs per-op only
    # in the retry pass, and only when the fast pass's result fails it.
    baseline = solid.copy()
    # The deep gate is only informative when the BASE passes it: some meshes
    # arrive with (benign, tolerated) self-intersections that every candidate
    # would inherit, and rejecting on those would veto all ops. For such parts
    # the per-op isValid/bbox/volume guards plus the pipeline's export
    # re-validation remain the safety net.
    try:
        _check_no_self_intersection(baseline)
        base_clean = True
    except Exception:  # noqa: BLE001
        base_clean = False
    result, ok_count = run(solid, deep=False)
    if ok_count and base_clean:
        try:
            _check_no_self_intersection(result)
        except Exception:  # noqa: BLE001 - retry, filtering the offending op(s)
            progress("  swept result failed deep check; retrying with per-op checks")
            result, ok_count = run(baseline, deep=True)
    progress(f"  swept-wall arcs cleaned {ok_count}/{len(todo)}")
    return result, len(todo), ok_count


def _check_no_self_intersection(shape) -> None:
    """Raise if OCC's BOP check reports a self-intersection.

    ``shape.check(True)`` reports several error classes; benign ones
    (TooSmallEdge, InvalidCurveOnSurface) occur even on solids that export and
    re-read perfectly, but a BOPAlgo SelfIntersect passes ``isValid()`` in
    memory and then re-reads INVALID from the exported STEP. Only the fatal
    class rejects.
    """
    try:
        shape.check(True)
    except Exception as exc:  # noqa: BLE001 - inspect the error classes
        if "SelfIntersect" in str(exc):
            raise ValueError("BOP check: self-intersection") from exc


def _arc_span_rad(seg) -> float:
    """Angular span (rad, positive) of a fitted arc segment."""
    c = seg.center
    a0 = math.atan2(seg.p0[1] - c[1], seg.p0[0] - c[0])
    a1 = math.atan2(seg.p1[1] - c[1], seg.p1[0] - c[0])
    if seg.ccw:
        while a1 <= a0:
            a1 += 2.0 * math.pi
        return a1 - a0
    while a1 >= a0:
        a1 -= 2.0 * math.pi
    return a0 - a1


def _annotate_profile(vertices, faces_sub, profile: SweptProfile, normals, areas) -> None:
    """Annotate each arc segment with its material side + facet coverage.

    ``outward``: the member facets' normals point away from the arc centre —
    a convex wall whose inscribed facets stop short of the true surface (fuse
    the sliver); ``False`` is a concave wall whose chords overshoot (cut).
    ``covered``: the segment's parametric rectangle (arc span x axial extent)
    is essentially fully covered by facet area; ``False`` means cutouts pierce
    the wall there and a fuse would bridge them (only cuts stay safe).
    """
    idx = np.asarray(profile.face_indices, dtype=int)
    if idx.size == 0:
        return
    cent = vertices[faces_sub[idx]].mean(axis=1)
    rel = cent - profile.origin
    c2 = np.column_stack((rel @ profile.e1, rel @ profile.e2))
    n2 = np.column_stack((normals[idx] @ profile.e1, normals[idx] @ profile.e2))
    n2n = np.linalg.norm(n2, axis=1)
    n2u = n2 / (n2n[:, None] + 1e-12)
    ar = areas[idx]
    extent = profile.axial_max - profile.axial_min
    for seg in profile.segments:
        if seg.kind != "arc" or seg.radius <= 1e-9:
            continue
        c = seg.center
        span = _arc_span_rad(seg)
        a0 = math.atan2(seg.p0[1] - c[1], seg.p0[0] - c[0])
        d = c2 - c
        rho = np.linalg.norm(d, axis=1)
        ang = np.arctan2(d[:, 1], d[:, 0])
        off = (ang - a0) if seg.ccw else (a0 - ang)
        off = np.mod(off, 2.0 * math.pi)
        # Radial band capped by half the radius: a thin panel's two opposite
        # surfaces both project near the profile curve, and a loose band would
        # mix their (opposite) normals into a coin-flip classification.
        band = min(1.5, max(0.3, 0.5 * seg.radius))
        in_zone = (off <= span) & (np.abs(rho - seg.radius) <= band)
        radial = d / (rho[:, None] + 1e-12)
        align = np.sum(n2u * radial, axis=1)
        # An arc-band facet's normal is radial about the arc centre; tangent
        # wall facets near the join and opposite-side panel facets are not.
        mask = in_zone & (np.abs(align) > 0.8) & (n2n > 0.3)
        if mask.sum() < 1:
            seg.outward = None
            continue
        mean_align = float(np.mean(align[mask]))
        if abs(mean_align) < 0.6:
            # Mixed population (both material sides in the band): unsafe.
            seg.outward = None
            continue
        seg.outward = bool(mean_align > 0)
        expected = span * seg.radius * extent
        seg.covered = bool(expected > 1e-9
                           and float(ar[in_zone].sum()) >= 0.8 * expected)


def _detect_gears(vertices, faces, config: ConversionConfig, progress):
    """Detect gear / whole-outline extrusion profiles on the FULL mesh (M5.3).

    Runs swept detection on the whole (un-claimed) mesh so the gear's outer tooth
    ring is found as ONE region — the ladder's claimed-subset swept pass fragments
    it (removing the coaxial hub cylinders/cones breaks the outline's smooth
    chain). Returns only ``whole_extrusion``-flagged profiles (a repeated-arc
    region wrapping the axis). Best-effort; [] on any failure."""
    if not (config.detect_swept_walls and getattr(config, "reconstruct_gears", True)):
        return []
    try:
        resolution = mesh_resolution(vertices, faces, config)
        regions = segment_planar(vertices, faces, config)
        sweeps = segment_swept_walls(vertices, faces, set(), regions, config)
        profiles = detect_swept_walls(vertices, faces, sweeps, config, resolution)
    except Exception as exc:  # noqa: BLE001 - gear detection is best-effort
        progress(f"Gear detection skipped ({exc})")
        return []
    gears = [p for p in profiles if getattr(p, "whole_extrusion", False)]
    if gears:
        progress(f"Found {len(gears)} gear/whole-outline profile(s) "
                 f"({', '.join(str(len(p.segments)) + ' segs' for p in gears)})")
    return gears


def _fit_swepts(vertices, faces_sub, regions, config: ConversionConfig, progress):
    """Detect + fit + annotate swept walls on an (unclaimed) face subset.

    Best-effort: any failure returns no profiles and leaves the strips faceted.
    """
    if not config.detect_swept_walls:
        return []
    try:
        resolution = mesh_resolution(vertices, faces_sub, config)
        sweeps = segment_swept_walls(vertices, faces_sub, set(), regions, config)
        profiles = detect_swept_walls(vertices, faces_sub, sweeps, config, resolution)
        normals, areas = face_normals_and_areas(vertices, faces_sub)
        for pr in profiles:
            _annotate_profile(vertices, faces_sub, pr, normals, areas)
    except Exception as exc:  # noqa: BLE001 - swept detection is best-effort
        progress(f"Swept-wall detection skipped ({exc})")
        return []
    if profiles:
        snaps = sum(p.tangency_snaps for p in profiles)
        arcs = sum(p.n_arcs for p in profiles)
        progress(f"Found {len(profiles)} swept wall(s) "
                 f"({arcs} arc segment(s), {snaps} tangency-snapped joins)")
    return profiles


def _swept_arc_lens_tool(profile: SweptProfile, seg, Part, pad: float):
    """Boolean tool for one swept arc segment: the chord<->arc lens, extruded.

    The lens is the 2D region between the arc (pushed out by a cut-eps so the
    surface clears the on-circle vertices instead of pinching them) and its full
    chord, lifted to 3D and extruded along the sweep axis over the wall's axial
    extent (+/- ``pad``). Material-side logic (design §4 boolean pattern):

    - Convex wall (facets inscribed): material stops at the chords; FUSING the
      lens adds the sliver up to the true curve. The part of the lens between
      the big chord and the faceted surface already overlaps material — a fuse
      is idempotent there.
    - Concave wall (chords overshoot): the sliver between the faceted surface
      and the true curve is excess material; CUTTING the lens removes exactly
      it. The rest of the lens is void, where a cut is a no-op.
    """
    R = float(seg.radius)
    eps = _clean_cut_eps(R)
    scale = (R + eps) / R
    c2 = seg.center
    p0s = c2 + (seg.p0 - c2) * scale
    p1s = c2 + (seg.p1 - c2) * scale
    pms = c2 + (_arc_mid_point2(seg) - c2) * scale
    axis = profile.axis
    shift = -pad * axis
    P0 = profile.point3d(p0s) + shift
    P1 = profile.point3d(p1s) + shift
    Pm = profile.point3d(pms) + shift
    chord = Part.makeLine(_vec(P0), _vec(P1))
    arc = Part.Arc(_vec(P1), _vec(Pm), _vec(P0)).toShape()
    wire = Part.Wire([chord, arc])
    face = Part.Face(wire)
    extent = profile.axial_max - profile.axial_min
    return face.extrude(_vec(axis * (extent + 2.0 * pad)))


def _guarded_cut(solid, tool, max_removed_frac: float = 0.5):
    """Cut a tool, but refuse if it would remove a large share of the tool.

    A correct swept-lens cut removes only the thin sagitta sliver between the
    faceted wall and the true curve — a small fraction of the lens volume. A
    mis-fitted lens that lands inside solid material would remove ~its whole
    volume; that would silently carve the part while staying a valid solid, so
    validity checks alone cannot catch it.
    """
    cut = solid.cut(tool)
    removed = solid.Volume - cut.Volume
    if removed > max_removed_frac * tool.Volume:
        raise ValueError(
            f"swept lens cut rejected: would remove {removed:.2f} of the tool's "
            f"{tool.Volume:.2f} volume (mis-fitted profile)")
    # Collapse guard: a local cut can only remove a bounded share of the tool, so
    # it must never carve away most of the WHOLE part. A degenerate boolean can
    # return a valid-but-tiny fragment (seen on gridfinity_base_lid: a cap op left
    # a 6mm cube from a 210mm plate) whose small `removed` slips past the check
    # above. Reject any cut that removes more than 30% of the solid's volume.
    if removed > 0.30 * solid.Volume:
        raise ValueError(
            f"cut rejected: would remove {removed:.2f} of the part's "
            f"{solid.Volume:.2f} volume (degenerate boolean / part collapse)")
    return cut


def _swept_arc_cylinder_tool(profile: SweptProfile, seg, Part, pad: float):
    """Boolean tool for a wide (>178 deg) swept arc: the full cylinder.

    A wall-end bead (rounded free edge of a thin wall) or a rounded groove
    wraps more than half the circle, so the chord<->arc lens degenerates; the
    full cylinder at the fitted centre is the natural tool — exactly the
    hole/boss treatment, with the axis being the sweep direction.
    """
    R = float(seg.radius) + _clean_cut_eps(float(seg.radius))
    axis = profile.axis
    base = profile.point3d(seg.center) - pad * axis
    extent = profile.axial_max - profile.axial_min
    return Part.makeCylinder(R, float(extent + 2.0 * pad), _vec(base), _vec(axis))


def _boolean_clean_swept(solid, profile: SweptProfile, seg, Part):
    """Replace one faceted swept-arc wall band with the analytic surface via a
    boolean op: a chord<->arc lens for partial arcs, the full cylinder for
    beads/grooves wrapping more than half the circle. Cut for concave walls,
    guarded fuse for convex."""
    wide = _arc_span_rad(seg) > math.radians(178.0)
    if seg.outward:
        if not seg.covered:
            raise ValueError("swept fuse skipped: wall pierced by cutouts")
        if wide:
            # Rounded wall-end bead: fuse the full cylinder over the exact
            # extent. Coarse beads add up to ~a third of the tool (polygon-to-
            # circle sliver), so the guard is looser than the lens's.
            tool = _swept_arc_cylinder_tool(profile, seg, Part, pad=0.0)
            return _guarded_fuse(solid, tool, max_added_frac=0.45)
        tool = _swept_arc_lens_tool(profile, seg, Part, pad=0.0)
        # A correct fuse adds only the sagitta sliver (~4-15% of the lens even
        # on coarse 3-chord arcs); a mis-classified convex wall would add most
        # of the lens, so the cap is deliberately tight.
        return _guarded_fuse(solid, tool, max_added_frac=0.25)
    pad = _cut_pad(profile.axial_max - profile.axial_min)
    if wide:
        # Rounded groove / slot end: cut the full cylinder, like a bore.
        tool = _swept_arc_cylinder_tool(profile, seg, Part, pad=pad)
        return _guarded_cut(solid, tool, max_removed_frac=0.6)
    tool = _swept_arc_lens_tool(profile, seg, Part, pad=pad)
    return _guarded_cut(solid, tool, max_removed_frac=0.5)


# --------------------------------------------------------------------------- #
# Gear / whole-outline extrusion (M5.3). A repeated-arc CLOSED profile centered
# on the axis (a gear cross-section, a splined shaft) is built as ONE closed
# Part.Wire from the whole outline -> Face -> extrude -> ONE guarded fuse. O(base)
# once, not O(arcs × base). The central bore is cut afterwards (ladder
# discipline). Same trusted primitive chain as the swept-wall lens ops.
# --------------------------------------------------------------------------- #


def _gear_profile_wire(profile: SweptProfile, Part, config: ConversionConfig):
    """Build the closed gear outline as one flat ``Part.Wire`` from its outline
    loop (design §2, M5.3).

    Uses ``profile.outline_loop`` — the region's outer mesh boundary as an ordered
    closed 3D polyline — flattened onto the min-axial plane (each loop vertex
    projected to a common axial height so the profile is planar) and built as one
    closed polygon wire. This is robust where the fitted 2D segments are
    fragmented by decimation: a mesh boundary loop is closed by construction.
    Returns the wire or ``None``."""
    loop = getattr(profile, "outline_loop", None)
    if loop is None or len(loop) < 8:
        return None
    axis = np.asarray(profile.axis, float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    loop = np.asarray(loop, float)
    # Flatten onto a common axial plane (the min-axial height) so the outline is
    # planar — the teeth outline weaves slightly in axial height on the mesh.
    z0 = float((loop @ axis).min())
    flat = loop - ((loop @ axis) - z0)[:, None] * axis
    # Drop consecutive near-duplicate points that break the wire builders, and
    # thin the loop: a decimated tooth boundary has ~1600 weaving vertices, which
    # (a) a B-spline over-fits into a SELF-INTERSECTING curve and (b) make the
    # boolean of the extruded polygon against the base expensive. Keep points
    # spaced by a minimum chord — enough to preserve the tooth shape while
    # bounding the vertex count (and thus the boolean cost).
    span = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    min_chord = max(0.15, 0.01 * span)
    pts = [flat[0]]
    for p in flat[1:]:
        if float(np.linalg.norm(p - pts[-1])) > min_chord:
            pts.append(p)
    if len(pts) < 8:
        return None
    verts = [_vec(p) for p in pts]
    closed_verts = verts + [verts[0]] if (verts[0] - verts[-1]).Length > 1e-6 else verts

    def _tool_is_sane(w):
        """A wire is usable only if its extruded tool is a single clean solid —
        a self-intersecting outline extrudes to a degenerate solid that collapses
        the fuse."""
        try:
            face = Part.Face(w)
            if not face.isValid() or face.Area <= 1e-6:
                return False
            axis = np.asarray(profile.axis, float)
            axis = axis / (np.linalg.norm(axis) or 1.0)
            tool = face.extrude(_vec(axis * (profile.axial_max - profile.axial_min)))
            sols = getattr(tool, "Solids", [])
            return len(sols) == 1 and sols[0].isValid() and tool.Volume > 1e-6
        except Exception:  # noqa: BLE001
            return False

    # Prefer a smooth closed B-spline through the THINNED outline (few enough
    # points not to self-intersect) — its lateral faces are non-planar, so the
    # gear teeth stop reading as residual tessellation. Fall back to the closed
    # polygon (always watertight) if the spline's extruded tool isn't a clean
    # solid. Either way the fuse recomputes the analytic intersection.
    if getattr(config, "gear_flanks_as_spline", True) and len(verts) >= 8:
        try:
            bs = Part.BSplineCurve()
            bs.interpolate(Points=verts, PeriodicFlag=True)
            w = Part.Wire(bs.toShape())
            if w.isValid() and w.isClosed() and _tool_is_sane(w):
                return w
        except Exception:  # noqa: BLE001 - fall back to polygon
            pass
    try:
        wire = Part.makePolygon(closed_verts)
    except Exception:  # noqa: BLE001
        return None
    if not wire.isValid() or not wire.isClosed() or not _tool_is_sane(wire):
        return None
    return wire


def _boolean_clean_gear(solid, profile: SweptProfile, Part, config: ConversionConfig):
    """Build the whole gear outline as one extruded solid and fuse it (design §2).

    Closed wire from the full cross-section -> Face -> extrude along the sweep
    axis over the profile's axial extent -> ONE guarded fuse. Wire/face isValid
    prechecks and the guarded fuse (with the bbox/collapse nets in _try_boolean_
    step) revert wholesale on any doubt. The central bore is a separately-detected
    cylinder cut AFTER this fuse (ladder discipline), so it is not filled here."""
    wire = _gear_profile_wire(profile, Part, config)
    if wire is None:
        raise ValueError("gear outline wire not closed/valid")
    face = Part.Face(wire)
    if not face.isValid() or face.Area <= 0.0:
        raise ValueError("gear outline face invalid")
    extent = profile.axial_max - profile.axial_min
    if extent <= 1e-6:
        raise ValueError("gear extrusion extent degenerate")
    tool = face.extrude(_vec(np.asarray(profile.axis, float) * extent))
    if not getattr(tool, "Solids", []):
        raise ValueError("gear extrusion is not a solid")
    # The gear outline is the part's own cross-section, so fusing it mostly
    # overlaps existing material and only trues up the faceted tooth flanks; a
    # generous added-volume ceiling admits the sliver between the inscribed facet
    # polygon and the true outline without letting a mis-fit outline bulge out.
    return _guarded_fuse(solid, tool, max_added_frac=0.5)


def _apply_gear_extrusions(solid, profiles, Part, progress, *, bbox_guard, config):
    """Fuse each whole-outline (gear) profile via one guarded boolean.

    Returns ``(solid, attempted, built)``. Each op reverts on invalidity /
    bbox growth / collapse (per-feature revert discipline) AND on RTAF regression
    (a gear fuse that doesn't reduce residual tessellation — e.g. a polygon
    outline no smoother than the mesh teeth — is rolled back so the op can never
    make the part worse). A profile whose segment count exceeds the ceiling is
    skipped (left faceted)."""
    todo = [p for p in profiles if getattr(p, "whole_extrusion", False)]
    if not todo:
        return solid, 0, 0
    # Cost budget: each whole-outline fuse is a boolean against the base solid,
    # cost ~O(base_faces); attempt the FEW largest-extent outlines only, capped so
    # a many-region gear can't grind past the time budget (a fuse that doesn't
    # improve RTAF reverts anyway).
    todo = sorted(todo, key=lambda p: -(p.axial_max - p.axial_min))[:config.gear_max_ops]
    built = 0
    try:
        rtaf_before = compute_rtaf(solid, config).get("rtaf")
    except Exception:  # noqa: BLE001
        rtaf_before = None
    for prof in todo:
        if len(prof.segments) > config.gear_max_profile_segments:
            progress(f"  gear outline skipped: {len(prof.segments)} segments "
                     f"> ceiling {config.gear_max_profile_segments}")
            continue
        # Cheap pre-check: only a SMOOTH (B-spline) outline can lower RTAF — a
        # polygon outline is itself faceted, so its (expensive) fuse would revert
        # on the RTAF gate anyway. Build the wire once; if it isn't a single
        # smooth edge, skip the fuse (saves an O(base_faces) boolean per gear).
        wire = _gear_profile_wire(prof, Part, config)
        if wire is None:
            continue
        smooth = (len(wire.Edges) <= 2
                  and all("BSpline" in e.Curve.TypeId or "Circle" in e.Curve.TypeId
                          for e in wire.Edges))
        if not smooth:
            progress("  gear outline skipped: only a faceted (polygon) wire builds "
                     "— a fuse would not lower RTAF")
            continue
        candidate, ok = _try_boolean_step(
            solid, lambda s, p=prof: _boolean_clean_gear(s, p, Part, config),
            max_bbox_growth=bbox_guard)
        if not ok:
            continue
        if rtaf_before is not None:
            try:
                rtaf_after = compute_rtaf(candidate, config).get("rtaf")
            except Exception:  # noqa: BLE001
                rtaf_after = None
            if rtaf_after is not None and rtaf_after > rtaf_before + 1e-3:
                progress(f"  gear reverted: RTAF {rtaf_before:.3f} -> "
                         f"{rtaf_after:.3f} (no improvement)")
                continue
            if rtaf_after is not None:
                rtaf_before = rtaf_after
        solid = candidate
        built += 1
    progress(f"  gear whole-outline extrusions built {built}/{len(todo)}")
    return solid, len(todo), built


def _freeform_sheet_surface(sheet, Part):
    """Approximate a freeform sheet's (u,v) grid to a ``Part.BSplineSurface``.

    C1 / degree-3 with centripetal parametrisation — the robust regime: C2 /
    degree-5 explodes catastrophically on mesh-sampled (noisy) grids (poles
    blow up, deviation → millions). Tightens the tolerance progressively and
    REJECTS a fit whose pole count saturates near the grid size (a signal the
    approximation couldn't hit tolerance and merely interpolated mesh noise).
    Also rejects a fit whose max deviation to the region facets exceeds the
    sheet's resolution-scaled tolerance. Returns the surface or ``None``.
    """
    import FreeCAD  # type: ignore

    ng = sheet.grid.shape[0]
    pts = [[FreeCAD.Vector(*sheet.grid[i, j]) for j in range(sheet.grid.shape[1])]
           for i in range(ng)]
    surf = None
    for tol in (sheet.dev_tol, 2.0 * sheet.dev_tol, 4.0 * sheet.dev_tol):
        cand = Part.BSplineSurface()
        try:
            cand.approximate(Points=pts, DegMin=3, DegMax=3, Tolerance=float(tol),
                             Continuity=1, ParamType="Centripetal")
        except Exception:  # noqa: BLE001 - OCC "Surface not done" etc.
            continue
        if cand.NbUPoles >= ng - 1 or cand.NbVPoles >= ng - 1:
            continue  # pole saturation — interpolated, not approximated
        surf = cand
        break
    if surf is None:
        return None
    return surf


def _freeform_sheet_deviation(surf, grid, stride: int = 2, covered=None,
                              sample_pts=None) -> float:
    """Max distance from the region's real surface to the fitted B-spline sheet.

    Projects each ground-truth point onto the surface (nearest-parameter search)
    and returns the worst residual — how far the analytic sheet strays from the
    mesh it replaces.

    Ground truth priority:
    - ``sample_pts`` (the region's dense facet centroids) when supplied: the
      honest metric. The (u,v) grid — especially with an inpainted skirt —
      samples only sparse nodes and can hide a large between-node error, so a
      corner-wrapping region that is NOT a true height field fits its own grid
      nodes yet misses the real facets; scoring the dense centroids catches it.
    - else the grid cells, restricted to ``covered`` (real) cells if a mask is
      given (an inpainted cell has no ground truth to match)."""
    import FreeCAD  # type: ignore

    worst = 0.0
    if sample_pts is not None:
        for p in sample_pts:
            vp = FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))
            try:
                u, v = surf.parameter(vp)
                q = surf.value(u, v)
                d = float((q - vp).Length)
            except Exception:  # noqa: BLE001
                continue
            if d > worst:
                worst = d
        return worst

    ng = grid.shape[0]
    for i in range(0, ng, stride):
        for j in range(0, ng, stride):
            if covered is not None and not covered[i, j]:
                continue
            p = FreeCAD.Vector(*grid[i, j])
            try:
                u, v = surf.parameter(p)
                q = surf.value(u, v)
                d = float((q - p).Length)
            except Exception:  # noqa: BLE001
                continue
            if d > worst:
                worst = d
    return worst


def _boolean_clean_freeform(solid, sheet, surf, Part):
    """Replace a faceted doubly-curved region with the analytic B-spline sheet
    via a guarded boolean, following the M4 cut/fuse pattern.

    The sheet is oversized (sampled slightly past the region footprint) so its
    boundary falls in empty space and the boolean itself clips it against the
    surrounding walls — no analytic-vs-mesh edge matching needed. The tool is
    the half-space on the +axis side of the sheet (extruded a bounding-diagonal
    along ``axis``); CUTTING it removes the faceted overshoot outside the true
    surface. A guarded fuse of the −axis half-space (clipped to the base
    footprint) would add the sliver in troughs, but the cut alone lands the
    analytic face and is the robust op; the caller adopts only if it validates
    and improves RTAF. Raises on any failure so ``_try_boolean_step`` reverts.
    """
    import FreeCAD  # type: ignore

    face = surf.toShape()
    if not face.isValid() or face.Area <= 0.0:
        raise ValueError("freeform sheet face invalid")
    bb = solid.BoundBox
    diag = math.sqrt(bb.XLength ** 2 + bb.YLength ** 2 + bb.ZLength ** 2) or 1.0
    axis = np.asarray(sheet.axis, float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    tool = face.extrude(FreeCAD.Vector(*(axis * diag)))
    if not getattr(tool, "Solids", []):
        raise ValueError("freeform tool is not a solid")
    result = solid.cut(tool)
    solids = getattr(result, "Solids", [])
    if len(solids) != 1 or not solids[0].isValid():
        raise ValueError("freeform cut did not yield one valid solid")
    # The cut must actually plant the analytic sheet as B-spline face(s).
    n_before = sum(1 for f in solid.Faces if "BSpline" in f.Surface.TypeId)
    n_after = sum(1 for f in result.Faces if "BSpline" in f.Surface.TypeId)
    if n_after <= n_before:
        raise ValueError("freeform cut planted no B-spline face")
    return result


def _refit_subsheet(vertices, faces, region, config, resolution):
    """Re-sample + wrap a split sub-region as a FreeformSheet (mirrors
    fit_freeform_sheets' per-region body). Returns the sheet or None."""
    ng = int(config.freeform_grid)
    sampled = sample_freeform_grid(vertices, faces, region, ng,
                                   inpaint=config.freeform_inpaint,
                                   return_mask=True)
    if sampled is None:
        return None
    grid, missing, covered = sampled
    if missing > config.freeform_max_missing:
        return None
    edge = resolution.edge_for(region.face_indices)
    dev_tol = max(config.freeform_dev_tol_abs, config.freeform_dev_tol_rel * edge)
    fa = np.array(region.face_indices, dtype=int)
    centroids = vertices[faces[fa]].mean(axis=1)
    if len(centroids) > 400:
        centroids = centroids[::int(np.ceil(len(centroids) / 400.0))]
    return FreeformSheet(
        grid=grid, axis=region.axis, face_indices=region.face_indices,
        area=region.area, curvature=region.curvature, foldover=region.foldover,
        missing=missing, dev_tol=dev_tol, covered=covered, sample_pts=centroids)


def _apply_freeform_sheets(solid, sheets, config, Part, progress,
                           vertices=None, faces=None):
    """Integrate freeform B-spline sheets into a valid solid, adopting each only
    when it validates, stays bbox-stable, and LOWERS the RTAF (never trades a
    faceted strip fan for something worse). Returns ``(solid, attempted, built)``.

    Every op goes through ``_try_boolean_step`` (reverts on invalidity), plus a
    per-op RTAF check: the whole point is de-faceting, so an op that doesn't
    reduce residual tessellation is rolled back. Bounded by a base-face cost
    ceiling (the boolean of a doubly-curved sheet is O(base_faces)).

    Build-time region splitting (task §1): when a sheet's fitted B-spline misses
    the real mesh (deviation gate fails), the region is bisected along its
    curvature ridge and each half re-sampled + re-fitted, up to
    ``freeform_max_split_depth`` levels. This lets a large cast surface that is
    locally — but not globally — a height field be represented by 2-4 sub-sheets
    instead of shipping faceted, using the true B-spline deviation as the trigger
    (a quadratic-residual trigger over-fires on gentle single bumps)."""
    if not sheets:
        return solid, 0, 0
    limit = config.freeform_max_base_faces
    try:
        base_faces = len(solid.Faces)
    except Exception:  # noqa: BLE001
        base_faces = 0
    if limit is not None and base_faces > limit:
        progress(f"  freeform sheets skipped: base {base_faces} faces > "
                 f"ceiling {limit} (left faceted)")
        return solid, len(sheets), 0
    can_split = (vertices is not None and faces is not None
                 and config.freeform_max_split_depth > 0)
    resolution = None
    if can_split:
        try:
            resolution = mesh_resolution(vertices, faces, config)
        except Exception:  # noqa: BLE001
            can_split = False
    bbox_guard = config.boolean_max_bbox_growth
    built = 0
    attempted = 0
    # RTAF only changes after a *successful* op, so cache it and refresh on adopt
    # (compute_rtaf is O(faces·edges) — do not pay it per rejected attempt).
    try:
        rtaf_before = compute_rtaf(solid, config).get("rtaf")
    except Exception:  # noqa: BLE001
        rtaf_before = None
    # Work queue of (sheet, depth); largest-area first. A split pushes its two
    # sub-sheets back on at depth+1. ``freeform_max_ops`` caps the number of
    # actual BOOLEAN attempts (each is O(base_faces)) so a doomed part can't
    # grind for minutes; a split (pure numpy re-fit) does NOT consume that budget,
    # only a bounded fan-out via ``freeform_max_split_depth`` and the queue.
    boolean_ops = 0
    max_queue = config.freeform_max_ops * (2 ** config.freeform_max_split_depth + 1)
    queue = [(s, 0) for s in sorted(sheets, key=lambda s: -s.area)]
    processed = 0
    while queue and boolean_ops < config.freeform_max_ops and processed < max_queue:
        sheet, depth = queue.pop(0)
        processed += 1
        surf = _freeform_sheet_surface(sheet, Part)
        dev = None
        if surf is not None:
            dev = _freeform_sheet_deviation(
                surf, sheet.grid, covered=getattr(sheet, "covered", None),
                sample_pts=getattr(sheet, "sample_pts", None))
        if surf is None or dev > sheet.dev_tol * 1.5:
            # The single fit misses the mesh. Split the region and re-fit each
            # half (deviation-triggered, task §1) rather than shipping faceted.
            if can_split and depth < config.freeform_max_split_depth:
                from .segmentation import FreeformRegion, split_freeform_region
                region = FreeformRegion(
                    face_indices=list(sheet.face_indices), axis=sheet.axis,
                    e1=_axis_basis(sheet.axis)[0], e2=_axis_basis(sheet.axis)[1],
                    origin=np.array(vertices[faces[np.array(sheet.face_indices)]]
                                    .reshape(-1, 3).mean(axis=0)),
                    area=sheet.area, curvature=sheet.curvature,
                    foldover=sheet.foldover)
                subs = split_freeform_region(vertices, faces, region, config)
                new_sheets = [_refit_subsheet(vertices, faces, r, config, resolution)
                              for r in subs]
                new_sheets = [s for s in new_sheets if s is not None]
                if new_sheets:
                    progress(f"  freeform sheet split ({len(sheet.face_indices)} "
                             f"facets -> {len(new_sheets)} sub-sheets, dev "
                             f"{dev if dev is not None else float('nan'):.2f} mm)")
                    for s in sorted(new_sheets, key=lambda s: -s.area):
                        queue.append((s, depth + 1))
                    continue
            if dev is not None:
                progress(f"  freeform sheet rejected: deviation {dev:.2f} mm > "
                         f"tol {sheet.dev_tol:.2f} ({len(sheet.face_indices)} facets)")
            continue
        boolean_ops += 1
        attempted += 1
        candidate, ok = _try_boolean_step(
            solid, lambda s, sh=sheet, sf=surf: _boolean_clean_freeform(s, sh, sf, Part),
            max_bbox_growth=bbox_guard)
        if not ok:
            continue
        # RTAF-improvement gate: adopt only if the sheet actually de-facets.
        rtaf_after = None
        if rtaf_before is not None:
            try:
                rtaf_after = compute_rtaf(candidate, config).get("rtaf")
            except Exception:  # noqa: BLE001
                rtaf_after = None
            if rtaf_after is not None and rtaf_after >= rtaf_before - 1e-4:
                progress(f"  freeform sheet reverted: RTAF {rtaf_before:.3f} -> "
                         f"{rtaf_after:.3f} (no improvement)")
                continue
        solid = candidate
        if rtaf_after is not None:
            rtaf_before = rtaf_after
        built += 1
        progress(f"  freeform sheet built ({len(sheet.face_indices)} facets, "
                 f"area {sheet.area:.0f}, dev {dev:.2f} mm)")
    return solid, attempted, built


def build_reconstructed_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig,
    on_progress=None,
):
    """Reconstruct planar + cylindrical faces, sew them, return ``(shape, stats)``.

    Raises if no valid geometry can be built — the caller falls back to faceted.
    """
    import Part  # type: ignore

    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    progress("Detecting cylinders/holes")
    cylinders = detect_cylinders(vertices, faces, config)
    holes = sum(1 for c in cylinders if not c.outward)
    progress(f"Found {len(cylinders)} cylinders ({holes} holes)")
    cones = detect_cones(vertices, faces, cylinders, config)
    if cones:
        progress(f"Found {len(cones)} countersink cone(s)")
    claimed: set[int] = set()
    for cyl in cylinders:
        claimed.update(cyl.face_indices)
    for cone in cones:
        claimed.update(cone.face_indices)

    # Straight-edge fillets: partial-arc cylinder sections between two planes.
    # Detected here for reporting, but a partial-arc cylinder face does NOT sew
    # to the faceted band it replaces (its straight side edges don't coincide
    # with the neighbour planes), which would OPEN this sew-based shell. So in the
    # sew tier the fillet band is left to the normal planar/gap-fill path (its
    # arc rows build as thin planar faces, which sew). The analytic fillet
    # surface is delivered by the boolean-clean tier via cut/fuse (design §4:
    # "do NOT rely on sewing"), where a bad step reverts and stays watertight.
    fillets: list[Cylinder] = _detect_fillets(vertices, faces, claimed, config, progress)

    # Spheres (M3): domes + corner blends, after fillets and BEFORE swept walls.
    # A dome's facets are claimed here so the swept detector never fits doomed
    # lens ops to its latitude rows. The analytic sphere surface is delivered by
    # the boolean-clean tier (design §4: "do NOT rely on sewing"); in the sew
    # tier its strips build as thin planar faces (which sew), and the boolean
    # lens/ball ops below true them up on the closed solid.
    # DOME pass before swept (see the boolean tier for the dome/blend split).
    spheres: list[Sphere] = _detect_spheres(
        vertices, faces, claimed, config, progress, mode="dome")

    # Exact end-circles of the analytic faces, used to replace matching faceted
    # boundary loops so their edges coincide and sew.
    circles = _analytic_circles(cylinders, cones)

    # Segment the *remaining* facets into planar regions. Removing the cylinder
    # walls turns each hole into a clean inner boundary loop on its end faces.
    keep = [i for i in range(len(faces)) if i not in claimed]
    faces_sub = faces[keep]

    progress("Segmenting planar regions")
    regions = segment_planar(vertices, faces_sub, config)

    # Freeform B-spline sheets (Candidate B): residual doubly-curved height-field
    # regions, detected BEFORE swept walls so a doubly-curved shell's rows are
    # attributed to a sheet rather than mis-fit as constant-cross-section swept
    # strips. The double-curvature gate keeps single-curvature walls out, so this
    # never steals a legitimate sweep. Detection uses a snapshot ``claimed`` so
    # the sew tier still BUILDS + sews the strips (that closes the shell); once
    # the solid is closed and valid, the boolean sheet ops true them into one
    # analytic face — exactly how swept/sphere ops already work here. The swept
    # ops and freeform ops both run post-close, guarded and RTAF-gated, so a
    # region claimed by both simply keeps whichever op improves it and reverts
    # the other; no explicit hand-off is needed in this tier.
    freeform_sheets = []
    if config.fit_freeform_sheets:
        try:
            freeform_sheets = fit_freeform_sheets(vertices, faces, claimed, config)
        except Exception as exc:  # noqa: BLE001 - detection is best-effort
            progress(f"Freeform sheet detection skipped ({exc})")
            freeform_sheets = []
        if freeform_sheets:
            progress(f"Found {len(freeform_sheets)} freeform sheet region(s) "
                     f"({sum(len(s.face_indices) for s in freeform_sheets)} facets)")

    # Swept/extruded curved walls (M4): fit profiles for chains of thin strips
    # perpendicular to a common extrusion direction. The strips are still built
    # and sewn as before (their chordal edges are what closes the shell); once
    # the solid is closed and valid, each fitted arc segment is replaced by the
    # analytic surface via a boolean lens op below — booleans recompute the
    # intersection geometry, so no analytic-vs-chordal edge matching is needed.
    swept_profiles = _fit_swepts(vertices, faces_sub, regions, config, progress)
    # Claim swept facets (mapped to original indices), then the BLEND pass finds
    # corner blends on what's left — without stealing swept-wall corners.
    for prof in swept_profiles:
        claimed.update(keep[i] for i in prof.face_indices)
    blend_spheres = _detect_spheres(vertices, faces, claimed, config, progress,
                                    mode="blend")
    spheres = spheres + blend_spheres
    for s in blend_spheres:
        claimed.update(s.face_indices)

    progress(f"Building {len(regions):,} planar faces")
    occ_faces = []
    reconstructed = 0
    skipped = 0
    gap_rows: list[int] = []  # faces_sub rows we couldn't merge — emit faceted
    for region in regions:
        rows = region.face_indices  # rows into faces_sub
        if region.size < config.min_region_facets:
            skipped += region.size
            gap_rows.extend(rows)
            continue
        loops = extract_face_loops(vertices, faces_sub, region, config)
        if loops is None:
            skipped += region.size
            gap_rows.extend(rows)
            continue
        try:
            occ_faces.append(_planar_face(loops, circles, Part))
            reconstructed += 1
        except Exception:  # noqa: BLE001 - OCC raises bare RuntimeErrors
            skipped += region.size
            gap_rows.extend(rows)

    cyl_faces_ok = 0
    for cyl in cylinders:
        try:
            occ_faces.append(_cylinder_face(cyl, Part))
            cyl_faces_ok += 1
        except Exception:  # noqa: BLE001
            pass

    cone_faces_ok = 0
    for cone in cones:
        try:
            occ_faces.append(_cone_face(cone, Part))
            cone_faces_ok += 1
        except Exception:  # noqa: BLE001
            pass

    # Fillet faces are NOT built in the sew tier (they don't sew — see above);
    # the boolean-clean tier delivers them. Reported as detected only.
    fillet_faces_ok = 0

    # Gap-fill: patch the facets that couldn't be reconstructed so the shell has
    # no holes and sews watertight (manifold). Facets are grouped into connected
    # local patches (genuinely separate pockets stay separate — sharp edges
    # between them are preserved) and each patch is merged with removeSplitter,
    # so flat sub-regions inside a pocket still collapse to one face and only
    # real curvature stays faceted. Keeps the face count far below a fully
    # faceted solid, which is what made the naive per-triangle version hang.
    gap_faces = 0
    gap_patches = 0
    if config.fill_faceted_gaps and gap_rows:
        progress(f"Gap-filling {len(gap_rows):,} facets to close the solid")
        adjacency = build_edge_adjacency(faces_sub)
        neighbors: list[list[int]] = [[] for _ in range(len(faces_sub))]
        for incident in adjacency.values():
            for i in incident:
                for j in incident:
                    if i != j:
                        neighbors[i].append(j)
        components = _connected_components(gap_rows, neighbors)
        components.sort(key=len, reverse=True)
        sizes = [len(c) for c in components]
        progress(f"  {len(components):,} local patch(es); largest={sizes[:5]}")
        all_points = [_vec(p) for p in vertices]
        for idx, comp in enumerate(components):
            if len(comp) > 500:
                progress(f"  merging large patch {idx + 1}/{len(components)} "
                         f"({len(comp):,} facets)...")
            patch_faces = _faceted_faces(all_points, faces_sub[comp], Part)
            occ_faces.extend(patch_faces)
            gap_faces += len(patch_faces)
        gap_patches = len(components)
        progress(f"  gap patches merged to {gap_faces:,} faces "
                 f"(from {len(gap_rows):,} facets)")

    if not occ_faces:
        raise RuntimeError("no faces could be reconstructed")

    progress(f"Sewing {len(occ_faces):,} faces into a solid")
    shape, is_solid = _faces_to_solid(occ_faces, Part, config.sew_tolerance, on_progress)

    # Snapshot the freshly-sewn solid BEFORE any boolean truing ops (P0-3): the
    # ops below (swept lenses, sphere balls, freeform sheets) can leave a solid
    # that passes isValid() in memory yet re-reads invalid from the exported STEP
    # (the tweezer's sew-tier ball-fuse). If the post-op export round-trip fails,
    # we revert to this known-good pre-op solid rather than shipping an invalid
    # file — bringing the reconstructed tier to export-revalidation parity with
    # the boolean tier's back-off ladder.
    pre_ops_shape = shape.copy() if is_solid else None

    # Swept-wall lens ops (M4): on a closed, valid solid, replace each fitted
    # arc segment's chordal strip fan with the analytic surface via a boolean
    # cut/fuse. Every op reverts on invalidity, so this can only improve the
    # surface, never cost watertightness. Swept lens ops FIRST (the dominant win),
    # then sphere ball ops: a sphere fuse reshapes the wall geometry a later swept
    # lens op keys off, so doing spheres first can make the swept ops miss (seen
    # on the tweezer: 8 swept walls -> 0). Each op reverts on invalidity.
    swept_ops = 0
    swept_built = 0
    if is_solid and swept_profiles:
        shape, swept_ops, swept_built = _apply_swept_lens_ops(
            shape, swept_profiles, Part, progress, config=config)
        if swept_built:
            # Simplify a snapshot: the OCC call can corrupt its input in place.
            backup = shape.copy()
            simplified = _safe_remove_splitter(shape, Part)
            shape = simplified if _is_valid_solid(simplified) else backup
            shape, _slivers = _defeature_sliver_chains(shape, config, Part, progress)
            solids = getattr(shape, "Solids", [])
            is_solid = bool(solids) and solids[0].isValid()

    # Sphere ball ops carry the same bbox-growth guard as the boolean tier (P0-2):
    # a mis-fit corner-blend ball that bulges past the silhouette (the tweezer's
    # 3.7mm over-grow) is reverted here instead of shipping an off-dimension solid.
    sphere_ok = 0
    if is_solid and spheres:
        bbox_guard = config.boolean_max_bbox_growth
        shape, sphere_ok = _apply_sphere_ball_ops(
            shape, spheres, Part, progress, bbox_guard=bbox_guard, config=config)
        if sphere_ok:
            progress(f"  spheres cleaned {sphere_ok}/{len(spheres)}")
            solids = getattr(shape, "Solids", [])
            is_solid = bool(solids) and solids[0].isValid()

    # Freeform B-spline sheets (Candidate B): last, on a valid closed solid.
    freeform_ops = 0
    freeform_ok = 0
    if is_solid and freeform_sheets:
        shape, freeform_ops, freeform_ok = _apply_freeform_sheets(
            shape, freeform_sheets, config, Part, progress,
            vertices=vertices, faces=faces)
        if freeform_ok:
            backup = shape.copy()
            simplified = _safe_remove_splitter(shape, Part)
            shape = simplified if _is_valid_solid(simplified) else backup
            progress(f"  freeform sheets built {freeform_ok}/{freeform_ops}")
            solids = getattr(shape, "Solids", [])
            is_solid = bool(solids) and solids[0].isValid()

    # Export round-trip re-validation with back-off (P0-3): if the truing ops
    # produced a solid that re-reads invalid from an exported STEP, revert to the
    # pre-op sewn solid (if it round-trips clean). This is the sew-tier analogue
    # of the boolean tier's decimation back-off ladder — a boolean op that passes
    # isValid() in memory but writes an invalid STEP (self-intersecting wires from
    # a ball fuse) must not ship. Only runs when ops actually modified the solid.
    export_revalidated = None
    ops_applied = bool(swept_built or sphere_ok or freeform_ok)
    if (is_solid and ops_applied and pre_ops_shape is not None
            and config.revalidate_export):
        if not _reread_valid(shape, Part, progress):
            progress("  reconstructed solid re-reads INVALID after truing ops; "
                     "reverting to pre-op sewn solid")
            if _reread_valid(pre_ops_shape, Part, progress):
                shape = pre_ops_shape
                swept_ops = swept_built = sphere_ok = 0
                freeform_ops = freeform_ok = 0
                export_revalidated = True
                solids = getattr(shape, "Solids", [])
                is_solid = bool(solids) and solids[0].isValid()
            else:
                # Even the pre-op solid doesn't round-trip; keep the truing result
                # (the pipeline's final revalidation will flag it) rather than
                # trading it for an equally-invalid solid with worse RTAF.
                export_revalidated = False
        else:
            export_revalidated = True

    stats = {
        "faces_in": int(len(faces)),
        "planar_faces": reconstructed,
        "cylinder_faces": cyl_faces_ok,
        "cone_faces": cone_faces_ok,
        "fillet_faces": fillet_faces_ok,
        "gap_faces": gap_faces,
        "gap_patches": gap_patches,
        "cylinders_detected": len(cylinders),
        "faces_out": reconstructed + cyl_faces_ok + cone_faces_ok
        + fillet_faces_ok + gap_faces,
        "cylinders": [c.as_dict() for c in cylinders],
        "cones_detected": len(cones),
        "cones": [c.as_dict() for c in cones],
        "fillets_detected": len(fillets),
        "fillets": [f.as_dict() for f in fillets],
        "fillet_radius_source": _radius_source_breakdown(fillets),
        "spheres_detected": len(spheres),
        "spheres_built": sphere_ok,
        "spheres": [s.as_dict() for s in spheres],
        "swept_walls_detected": len(swept_profiles),
        "swept_walls_built": swept_built,
        "swept_arc_ops": swept_ops,
        "swept_tangency_snaps": sum(p.tangency_snaps for p in swept_profiles),
        "swept_detail": [p.as_dict() for p in swept_profiles],
        "freeform_sheets_detected": len(freeform_sheets),
        "freeform_sheets_built": freeform_ok,
        "freeform_detail": [s.as_dict() for s in freeform_sheets],
        "skipped_facets": skipped,
        "is_solid": is_solid,
    }
    if export_revalidated is not None:
        stats["export_revalidated"] = export_revalidated
    return shape, stats


def _radius_source_breakdown(fillets) -> dict:
    """Count of fillet radii derived from tangency vs free fit (QA reporting)."""
    out = {"tangency": 0, "fit": 0}
    for f in fillets:
        out[f.radius_source] = out.get(f.radius_source, 0) + 1
    return out


def _detect_fillets(vertices, faces, claimed: set, config: ConversionConfig, progress):
    """Detect straight-edge fillets (design §2, §3). Returns a list of fillet
    ``Cylinder`` objects; empty when ``detect_fillets`` is off or none found.

    Runs planar segmentation + smooth-band grouping on the unclaimed facets, then
    ``detect_fillets_straight`` on the ``band``-classed regions. Best-effort: any
    failure leaves the bands faceted (never raises)."""
    if not config.detect_fillets:
        return []
    try:
        resolution = mesh_resolution(vertices, faces, config)
        regions = segment_planar(vertices, faces, config)
        bands = segment_smooth_bands(vertices, faces, claimed, regions, config)
        n_band = sum(1 for b in bands if b.class_hint == "band")
        fillets = detect_fillets_straight(
            vertices, faces, bands, regions, set(claimed), config, resolution)
    except Exception as exc:  # noqa: BLE001 - fillet detection is best-effort
        progress(f"Fillet detection skipped ({exc})")
        return []
    if fillets:
        tan = sum(1 for f in fillets if f.radius_source == "tangency")
        progress(f"Found {len(fillets)} straight-edge fillet(s) "
                 f"({tan} tangency-snapped) from {n_band} band(s)")
    return fillets


def _detect_knurling(vertices, faces, cylinders, claimed: set,
                     config: ConversionConfig, progress):
    """Detect knurled bands (design §3, M5.1). Best-effort; never raises."""
    if not getattr(config, "detect_knurling", True):
        return []
    try:
        knurls = detect_knurling(vertices, faces, cylinders, claimed, config)
    except Exception as exc:  # noqa: BLE001 - knurl detection is best-effort
        progress(f"Knurl detection skipped ({exc})")
        return []
    if knurls:
        progress(f"Found {len(knurls)} knurled band(s) "
                 f"({', '.join(k.pattern for k in knurls)})")
    return knurls


def _detect_threads(vertices, faces, cylinders, claimed: set,
                    config: ConversionConfig, progress):
    """Detect threaded bands (design §1, M5.2). Best-effort; never raises."""
    if not getattr(config, "detect_threads", True):
        return []
    try:
        threads = detect_threads(vertices, faces, cylinders, claimed, config)
    except Exception as exc:  # noqa: BLE001 - thread detection is best-effort
        progress(f"Thread detection skipped ({exc})")
        return []
    if threads:
        kinds = ", ".join(("internal" if t.is_internal else "external")
                          for t in threads)
        progress(f"Found {len(threads)} thread(s) ({kinds}); "
                 f"pitch {', '.join(f'{t.pitch:.2f}' for t in threads)} mm")
    return threads


def _boolean_clean_thread(solid, thread: Thread, Part, **_):
    """Suppress a threaded band to its pitch-diameter cylinder (design §1).

    External thread (a bolt/cap-neck, material inside): FUSE a cylinder at the
    pitch radius over the band's extent to true up the crest micro-facets.
    Internal thread (a nut/cap bore, material outside): CUT a cylinder at the
    pitch radius to trim the crest ridges back to a clean bore wall. Same proven
    primitive chain as _boolean_clean_cylinder; metadata carries pitch/starts."""
    cyl = Cylinder(
        axis_point=np.asarray(thread.axis_point, float),
        axis_dir=np.asarray(thread.axis_dir, float),
        radius=float(thread.suppress_radius),
        axial_min=float(thread.axial_min),
        axial_max=float(thread.axial_max),
        rms=0.0, face_indices=list(thread.face_indices),
        outward=bool(thread.outward),
    )
    return _boolean_clean_cylinder(solid, cyl, Part)


def _boolean_clean_knurl(solid, knurl: KnurlBand, Part, **_):
    """Suppress a knurled band to its median-radius mid-surface cylinder.

    Reuses the proven cylinder cut/fuse: a grip knurl is a boss (material inside)
    so we FUSE a cylinder of the band's nominal (mid-surface) radius over its
    exact axial extent to true up the crest micro-facets into one clean wall; a
    knurled bore is a hole so we CUT. Same primitive chain as _boolean_clean_
    cylinder — the metadata (pattern/diameter) is what distinguishes it."""
    cyl = Cylinder(
        axis_point=np.asarray(knurl.axis_point, float),
        axis_dir=np.asarray(knurl.axis_dir, float),
        radius=float(knurl.suppress_radius),
        axial_min=float(knurl.axial_min),
        axial_max=float(knurl.axial_max),
        rms=0.0, face_indices=list(knurl.face_indices),
        outward=bool(knurl.outward),
    )
    return _boolean_clean_cylinder(solid, cyl, Part)


def _dedupe_spheres(spheres):
    """Merge spheres sharing a (centre, radius) — a fragmented cap found twice.

    Two spheres are the same when their centres coincide within a radius-scaled
    tolerance and their radii agree; the merged sphere keeps the union of their
    facets so its metadata (facet count / coverage) reflects the whole cap.
    """
    kept: list = []
    for s in spheres:
        merged = False
        for k in kept:
            rtol = 0.05 * max(s.radius, k.radius) + 0.2
            if (abs(s.radius - k.radius) <= rtol
                    and float(np.linalg.norm(np.asarray(s.center) - np.asarray(k.center)))
                    <= rtol):
                k.face_indices = sorted(set(k.face_indices) | set(s.face_indices))
                merged = True
                break
        if not merged:
            kept.append(s)
    return kept


def _detect_spheres(vertices, faces, claimed: set, config: ConversionConfig,
                    progress, mode: str = "all"):
    """Detect spherical caps/domes + corner blends (design §3, task §3).

    Two paths, run on the facets no cylinder/cone/fillet claimed:

    1. **Tessellated domes** (``mode`` in ``all``/``dome``) — a dome (a grille
       cap) shatters into many strips, none compact enough to read as a sphere
       alone. The cross-region sphere *consensus* clusters those strips by shared
       (centre, R) and fits one sphere to the union. Run BEFORE swept-wall fitting
       so the dome's facets are removed from the pool (otherwise M4 fits doomed
       lens ops to the latitude rows).
    2. **Compact caps/blends** (``mode`` in ``all``/``blend``) —
       ``segment_smooth_bands`` groups the residual curved strips; ``cap``/
       ``blend``-classed regions whose normals fan out fit one sphere each (corner
       blends get the tangency prior from their flats). Run AFTER swept-wall
       detection so a corner blend that is really part of a swept curved wall is
       claimed by the sweep first — otherwise the blend cannibalises the sweep's
       facets and drops swept walls (seen on drive_bay: 35 -> 28 built).

    Returns a list of ``Sphere`` objects; updates ``claimed`` in place. Best-
    effort: any failure leaves the regions faceted (never raises).
    """
    if not config.detect_spheres:
        return []
    dome_spheres: list[Sphere] = []
    try:
        resolution = mesh_resolution(vertices, faces, config)
        keep = [i for i in range(len(faces)) if i not in claimed]
        if not keep:
            return []
        faces_sub = faces[keep]
        regions = segment_planar(vertices, faces_sub, config)
        # Map sub-region facet rows back to original indices for claiming.
        def to_orig(sub_ids):
            return [keep[i] for i in sub_ids]

        spheres: list[Sphere] = []
        if mode in ("all", "dome"):
            # Dome consensus over ALL planar strips (candidate regions).
            candidate_regions = [r.face_indices for r in regions]
            dome_spheres, _dc = sphere_consensus_regions(
                vertices, faces_sub, candidate_regions, config, resolution)
            for sph in dome_spheres:
                sph.face_indices = to_orig(sph.face_indices)
                spheres.append(sph)
                claimed.update(sph.face_indices)

        if mode in ("all", "blend"):
            # Compact caps/blends over the residual strips.
            bands = segment_smooth_bands(vertices, faces_sub, set(), regions, config)
            cap_spheres = detect_spheres(
                vertices, faces_sub, bands, regions, set(), config, resolution)
            for sph in cap_spheres:
                sph.face_indices = to_orig(sph.face_indices)
                spheres.append(sph)
                claimed.update(sph.face_indices)

        # A single cap can fragment into >1 smooth band (an apex/seam split), so
        # two spheres of the same (centre, R) get built as identical redundant
        # balls. Merge them — the boolean op is idempotent but the extra op is
        # wasted time, and one analytic face reads cleaner than two.
        spheres = _dedupe_spheres(spheres)
    except Exception as exc:  # noqa: BLE001 - sphere detection is best-effort
        progress(f"Sphere detection skipped ({exc})")
        return []
    if spheres:
        domes = len(dome_spheres)
        tan = sum(1 for s in spheres if s.radius_source == "tangency")
        progress(f"Found {len(spheres)} sphere(s) "
                 f"({domes} dome(s) via consensus, {tan} tangency-snapped)")
    return spheres


def _safe_remove_splitter(shape, Part):
    """removeSplitter is an optimization (merge coplanar faces); never let a
    malformed edge/curve in a huge shell crash the whole reconstruction."""
    try:
        return shape.removeSplitter()
    except Exception:  # noqa: BLE001
        return shape


def _shell_from_faces(occ_faces, Part, progress):
    """Build a ``Part.Shell`` from a face list, flattening non-``Face`` entries.

    The fast path is a single ``Part.Shell(occ_faces)``. But if any entry is a
    ``Compound`` or ``Shell`` rather than a bare ``Face`` — one can leak from a
    gap-fill patch whose ``removeSplitter`` merged its facets into a compound —
    the bulk call throws ``TopoDS_UnCompatibleShapes`` (``TopoDS_Builder::Add``),
    which previously aborted reconstruction entirely and dropped the whole part
    to a fully-faceted fallback (double_4u: 3,262 faces, 0% analytic surfaces,
    43% RTAF — P1-1). ``TopoDS_Builder::Add`` only accepts ``Face`` shapes, so
    explode every entry into its constituent faces (a bare ``Face`` yields
    itself); that keeps ALL the geometry — no drop-and-gap-fill needed — and the
    shell builds cleanly. Genuinely unusable entries (null shapes, or ones with
    no faces) are dropped and logged.
    """
    flat = []
    exploded = 0
    dropped = 0
    for f in occ_faces:
        try:
            if f is None or f.isNull():
                dropped += 1
                continue
            if f.ShapeType == "Face":
                flat.append(f)
                continue
            member_faces = f.Faces  # Compound / Shell / Solid -> its faces
            if member_faces:
                flat.extend(member_faces)
                exploded += 1
            else:
                dropped += 1
        except Exception:  # noqa: BLE001 - a bad entry must not abort the shell
            dropped += 1
    if exploded or dropped:
        progress(f"  shell: flattened {exploded} compound/shell entr(ies), "
                 f"dropped {dropped} unusable; {len(flat)} faces")
    try:
        return Part.Shell(flat)
    except Exception as exc:  # noqa: BLE001 - last-resort per-face isolation
        progress(f"  shell build still hit {exc}; isolating face-by-face")
        accepted = []
        skipped = 0
        for f in flat:
            try:
                Part.Shell(accepted + [f])
                accepted.append(f)
            except Exception:  # noqa: BLE001
                skipped += 1
        progress(f"  shell built from {len(accepted)}/{len(flat)} faces "
                 f"({skipped} incompatible dropped)")
        return Part.Shell(accepted)


def _faces_to_solid(occ_faces, Part, tolerance: float = 1e-3, on_progress=None):
    """Sew faces into a (hopefully) closed solid. Returns ``(shape, is_solid)``.

    ``tolerance`` (mm) lets sewing bridge faces whose shared edges are
    coordinate-identical in theory (same source vertices) but differ by FP
    noise in practice — e.g. an analytic circle vs. the mesh-derived patch
    boundary it replaced, or two independently-built local patches.
    """
    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    shell = _shell_from_faces(occ_faces, Part, progress)
    sewn = shell.copy()
    try:
        progress("  sewShape...")
        sewn.sewShape(tolerance)
        progress("  sewShape done")
    except Exception:  # noqa: BLE001
        progress("  sewShape failed; using un-sewn shell")
        sewn = shell

    for candidate in (sewn, shell):
        try:
            solid = Part.Solid(candidate)
        except Exception:  # noqa: BLE001
            continue
        if solid.isValid():
            progress("  solid valid; simplifying")
            return _safe_remove_splitter(solid, Part), True

    # No closed solid; hand back the sewn shell so the caller can still export.
    progress("  no valid solid; returning open shell")
    return _safe_remove_splitter(sewn, Part), False


def build_faceted_solid(vertices: np.ndarray, faces: np.ndarray):
    """Classic fallback: wrap the raw mesh as a faceted solid.

    Mirrors what existing tools do — one STEP face per triangle — but is only
    used when reconstruction fails or ``--faceted`` is requested.
    """
    import Part  # type: ignore

    # Part.Shape.makeShapeFromMesh takes a (points, facet-index) topology tuple.
    points = [_vec(p) for p in vertices]
    topo = [(int(a), int(b), int(c)) for a, b, c in faces]
    shape = Part.Shape()
    shape.makeShapeFromMesh((points, topo), 0.1)
    try:
        # Coalesce coplanar facets where possible (fails on some large/degenerate
        # meshes — it's an optimization, so skip it if OCC objects).
        shape = shape.removeSplitter()
    except Exception:  # noqa: BLE001
        pass
    try:
        solid = Part.Solid(shape)
        if solid.isValid():
            return solid
    except Exception:  # noqa: BLE001
        pass
    return shape


def _boolean_cut_tool_cylinder(center, axis, radius: float, z0: float, z1: float, Part):
    """A solid cylinder for use as a boolean cut/fuse tool.

    ``z0``/``z1`` are axial offsets from ``center`` along ``axis``; the cylinder
    spans exactly that range (``center`` need not be the cylinder's own base).
    """
    base = center + z0 * axis
    height = z1 - z0
    return Part.makeCylinder(float(radius), float(height), _vec(base), _vec(axis))


def _clean_cut_eps(radius: float) -> float:
    """Radial clearance for an exact-radius boolean cut.

    A faceted feature's vertices sit *on* the fitted radius, so a cut at exactly
    that radius runs the cut surface through them — OCC then pinches the abutting
    flat face to a point at every vertex, fragmenting it into hundreds of zero-
    width slivers. Nudging the cut out by a hair past the vertices (and any small
    fit noise) lets the boolean clear the whole faceted rim and leave one clean
    face. The clearance is a micron-scale fraction of the radius, capped so a big
    feature never grows meaningfully — 1..50 µm, far below any print tolerance,
    so diameter/axis/centre are preserved for all practical purposes.
    """
    return min(max(1e-3, 2.5e-3 * radius), 0.05)


def _cut_pad(extent: float) -> float:
    """Axial over-run for a bore cut past the feature's own extent.

    Just enough for the cut to poke through the end faces and separate cleanly,
    but deliberately small: a large pad (the old ``max(extent, 1)`` — a full
    feature length) reaches down the axis and swallows anything coaxial, so a
    wide counterbore would erase the narrow through-hole beneath it (rack-mount
    screw holes are exactly this). 0.25..0.75 mm clears a face without spanning
    the millimetre-plus gap to the next coaxial feature.
    """
    return min(max(0.25, 0.05 * extent), 0.75)


def _wall_vertex_radii(vertices, faces, axis_dir, axis_point, face_indices):
    """Radial distances of a feature's wall-facet vertices from its axis."""
    idx = np.unique(np.asarray(faces)[list(face_indices)].ravel())
    d = vertices[idx] - np.asarray(axis_point, dtype=float)
    axis = np.asarray(axis_dir, dtype=float)
    radial = d - np.outer(d @ axis, axis)
    return np.linalg.norm(radial, axis=1)


def _design_radius(vertices, faces, axis_dir, axis_point, face_indices, fit_radius):
    """The feature's *design* radius — the circle its wall vertices lie on.

    A tessellated hole/boss is a polygon whose vertices sit on the design circle
    and whose chords bulge off it, so the algebraic (Kasa) fit — which balances
    residuals across the whole wall — can land *inside* the polygon's inradius
    (seen on real parts: fit 2.10 for a wall whose vertices are all at 2.125).
    Cutting at that under-fit radius misses the material entirely. The wall
    vertices are the reliable signal: a high quantile of their radial distances
    recovers the true radius while ignoring the odd misclassified facet. Falls
    back to the fit radius if there are too few vertices to be meaningful.
    """
    r = _wall_vertex_radii(vertices, faces, axis_dir, axis_point, face_indices)
    if r.size < 3:
        return float(fit_radius)
    return float(np.quantile(r, 0.95))


def _boolean_clean_cylinder(solid, cyl: Cylinder, Part, radius: float | None = None, **_):
    """Replace a faceted hole/boss with an exact analytic cylinder via a boolean
    op, instead of trying to sew mismatched topology.

    A faceted hole is *inscribed*: its vertices sit on the true cylinder and the
    chordal facets bulge **inward**, so the solid material only ever reaches the
    true radius R (never past it). Cutting a cylinder of *exactly* R therefore
    removes every scrap of faceted rim and leaves an analytic wall at exactly R,
    with the bore's end circles landing on the part's own faces — no fill-back,
    and crucially no oversized cut ring left behind (the old oversize+shrink
    scheme left a partial wall at R+margin whenever the shrink fuse-back failed,
    which is exactly the "partial cylinder artifact" we must avoid). Diameter,
    axis and centre are all preserved because the cut *is* the fitted cylinder.

    A boss is the mirror image (material inside R), so we fuse a solid cylinder
    of exactly R over the feature's exact axial extent to true-up its wall.

    Booleans recompute the intersection geometry, so the analytic tool and the
    faceted mesh need not share matching topology — unlike sewing.
    """
    R = float(cyl.radius if radius is None else radius)
    eps = _clean_cut_eps(R)
    axis = np.asarray(cyl.axis_dir, dtype=float)
    center = np.asarray(cyl.axis_point, dtype=float)
    zmin, zmax = cyl.axial_min, cyl.axial_max
    if cyl.outward:
        # Plain (unguarded) fuse: a boss legitimately swallows any nested bore
        # (flanged_pipe's flange fills its own bore, which the bore's later cut
        # re-opens), so the added-volume guard used for cones would reject it.
        # Cylinder detection has tight guards (coverage/centroid/rms), so
        # mis-detected bosses are rare here — unlike cone fits.
        fill = _boolean_cut_tool_cylinder(center, axis, R + eps, zmin, zmax, Part)
        return solid.fuse(fill)
    pad = _cut_pad(zmax - zmin)
    cut = _boolean_cut_tool_cylinder(center, axis, R + eps, zmin - pad, zmax + pad, Part)
    return solid.cut(cut)


def _fillet_wedge_tool(cyl: Cylinder, Part, eps: float, pad: float):
    """The exact rounded-corner solid for a *convex* straight-edge fillet.

    A convex fillet rounds an outer edge: the rounded corner solid is the
    cylinder (radius R, on the interior axis) clipped to the material wedge
    between the two tangent planes — i.e. only the sector the fillet actually
    occupies. Fusing this adds just the thin sliver between the inscribed faceted
    arc and the true arc, never the full disk (which would bulge into open air).
    The sector is taken from the fitted ``[u_start, u_span]`` plus a small angular
    pad so the tool fully clears the faceted rim.

    The tool spans EXACTLY the fillet's axial extent (no ``pad``): a convex fuse
    ADDS material, so an axial over-run would stick nubs of solid into open air
    past the part's ends. The band's axial extent already equals the edge length.
    """
    axis = np.asarray(cyl.axis_dir, dtype=float)
    center = np.asarray(cyl.axis_point, dtype=float)
    zmin, zmax = cyl.axial_min, cyl.axial_max
    R = float(cyl.radius) + eps
    height = (zmax - zmin)
    pad = 0.0
    base = center + zmin * axis
    full = Part.makeCylinder(R, height, _vec(base), _vec(axis))
    # Clip to the angular sector [u_start-dpad, u_start+u_span+dpad]. A cylinder
    # sector is the common of the solid cylinder and a prism spanning that wedge.
    u, v = _plane_basis(axis)
    dpad = math.radians(6.0)
    a0 = cyl.u_start - dpad
    a1 = cyl.u_start + cyl.u_span + dpad
    span = a1 - a0
    if span >= 2.0 * math.pi:
        return full
    # Build a wedge prism (fan of the sector) large enough to cover radius R.
    big = 2.5 * R
    steps = max(2, int(math.degrees(span) / 15) + 1)
    pts2d = [(0.0, 0.0)]
    for k in range(steps + 1):
        a = a0 + span * k / steps
        pts2d.append((big * math.cos(a), big * math.sin(a)))
    poly3d = [center + (zmin - pad) * axis + p2 * u + p2v * v
              for (p2, p2v) in pts2d]
    vfrom = [_vec(p) for p in poly3d]
    vfrom = vfrom + [vfrom[0]]
    wire = Part.makePolygon(vfrom)
    face = Part.Face(wire)
    prism = face.extrude(_vec(axis * height))
    try:
        return full.common(prism)
    except Exception:  # noqa: BLE001 - fall back to the full cylinder sector
        return full


def _plane_basis(axis):
    """Two orthonormal in-plane axes for a unit axis (builder-local copy)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(axis, u)
    return u, v


def _boolean_clean_fillet(solid, cyl: Cylinder, Part, **_):
    """Replace a faceted straight-edge fillet with an exact analytic cylinder
    sector via a boolean op.

    Concave (inner corner, ``outward=False``): the faceted arc bulges toward the
    axis (into open space), so the material overshoots the true arc — CUT a solid
    cylinder of radius R on the interior axis to trim it back to the analytic
    concave wall.

    Convex (outer edge, ``outward=True``): the faceted arc is inscribed, so
    material stops short of the true arc — FUSE the exact rounded-corner sector
    (the cylinder clipped to the material wedge) to true it up. Guarded so a
    mis-fit fillet that would bulge into open air is rejected.
    """
    R = float(cyl.radius)
    eps = _clean_cut_eps(R)
    axis = np.asarray(cyl.axis_dir, dtype=float)
    center = np.asarray(cyl.axis_point, dtype=float)
    zmin, zmax = cyl.axial_min, cyl.axial_max
    pad = _cut_pad(zmax - zmin)
    if cyl.outward:
        tool = _fillet_wedge_tool(cyl, Part, eps, pad)
        return _guarded_fuse(solid, tool, max_added_frac=0.5)
    cut = _boolean_cut_tool_cylinder(center, axis, R + eps, zmin - pad, zmax + pad, Part)
    return solid.cut(cut)


def _guarded_fuse(solid, tool, max_added_frac: float = 0.30):
    """Fuse a boss tool, but refuse if it would ADD real material.

    A correct boss fuse mostly overlaps existing material — the only volume it
    adds is the sliver between the inscribed faceted polygon and the true
    circle (a few percent even for coarse tessellations). A mis-detected boss
    (wrong radius/axis/extent) sticks out into open air instead, adding a large
    share of the tool's volume; that would silently distort the part while
    remaining a perfectly valid solid, so validity checks alone can't catch it.
    """
    fused = solid.fuse(tool)
    added = fused.Volume - solid.Volume
    if added > max_added_frac * tool.Volume:
        raise ValueError(
            f"boss fuse rejected: would add {added:.2f} of the tool's "
            f"{tool.Volume:.2f} volume (mis-detected boss)")
    # Collapse guard: a fuse ADDS material, so the result can never be smaller
    # than the input. A degenerate OCC fuse can return a valid-but-tiny solid
    # (seen on gridfinity_base_lid: a cap fuse turned a 210mm plate into a 6mm
    # cube, Vol 173576 -> 119, yet `added` was hugely NEGATIVE so the check above
    # passed it). Reject any fuse that shrinks the part at all beyond FP noise.
    if fused.Volume < solid.Volume * 0.999 - 1e-6:
        raise ValueError(
            f"boss fuse rejected: shrank the part from {solid.Volume:.2f} to "
            f"{fused.Volume:.2f} (degenerate boolean / part collapse)")
    return fused


def _boolean_clean_cone(solid, cone, Part, **_):
    """Exact-cut analogue of :func:`_boolean_clean_cylinder` for a countersink
    cone. The faceted cone is inscribed the same way, so cutting the exact cone
    (extended a hair past each end along its own taper so it passes cleanly
    through the surfaces) clears the facets and leaves an analytic conical wall
    at the fitted radii — no oversize ring. A boss cone (tapered neck/chamfer,
    material inside) is the mirror image: fuse the exact solid cone over the
    feature's exact extent instead — cutting it would carve the boss off."""
    r_base = float(cone.r_base)
    r_top = float(cone.r_top)
    axis = np.asarray(cone.axis_dir, dtype=float)
    center = np.asarray(cone.axis_point, dtype=float)
    zmin, zmax = float(cone.axial_min), float(cone.axial_max)
    height = zmax - zmin
    if getattr(cone, "outward", False):
        eps = _clean_cut_eps(max(r_base, r_top))
        fill = Part.makeCone(max(r_base + eps, 1e-4), max(r_top + eps, 1e-4), height,
                             _vec(center + zmin * axis), _vec(axis))
        return _guarded_fuse(solid, fill)
    pad = _cut_pad(height)
    # Nudge the radii out a hair (as for cylinders) so the cut clears the faceted
    # vertices instead of pinching the abutting faces into slivers, then extend
    # along the taper so radii stay ~exact at the real surfaces while the tool
    # over-runs into open air at both ends (slope = dR/dz along the axis).
    eps = _clean_cut_eps(max(r_base, r_top))
    r_base += eps
    r_top += eps
    slope = (r_top - r_base) / height if height > 1e-9 else 0.0
    r0 = max(r_base - slope * pad, 1e-4)
    r1 = max(r_top + slope * pad, 1e-4)
    cut = Part.makeCone(r0, r1, height + 2 * pad,
                        _vec(center + (zmin - pad) * axis), _vec(axis))
    return solid.cut(cut)


def _boolean_clean_sphere(solid, sph, Part, **_):
    """Replace a faceted dome / corner blend with an exact analytic sphere via a
    boolean op (design §4).

    Convex cap (dome / rounded boss top, ``outward=True``): the faceted cap is
    inscribed — material stops short of the true surface — so FUSE a solid ball
    of radius R+eps to true it up. Guarded so a mis-fit sphere that would bulge
    into open air is rejected. Concave dish (``outward=False``): material
    overshoots the true sphere, so CUT the ball to trim it back.

    A ball fuse/cut is far more localised than a lens — the tool is exactly the
    fitted sphere, so booleans recompute the trim against whatever flats/fillets
    bound the cap. Uses ``Part.makeSphere`` (design §4 boolean tool).
    """
    R = float(sph.radius)
    eps = _clean_cut_eps(R)
    center = np.asarray(sph.center, dtype=float)
    # Build only the CAP (a partial-sphere solid), not the whole ball — a full
    # ball fuse would add the far hemisphere sticking out of the part, and a full
    # ball cut would gouge past the dish. ``Part.makeSphere(r, pnt, dir, a1, a2,
    # a3)`` builds the solid wedge between latitudes a1..a2 (from -90 at the south
    # pole along ``dir`` to +90 at the north). The cap of angular radius ``alpha``
    # about ``cap_axis`` is latitudes [90-alpha, 90]; pad alpha generously so the
    # detected facets (whose near-tangent rim rows are absorbed into the flat) and
    # their sagitta slivers are fully covered, capped so it stays a cap not a ball.
    tool = None
    if sph.cap_axis is not None:
        cap_axis = np.asarray(sph.cap_axis, dtype=float)
        cap_axis /= np.linalg.norm(cap_axis) or 1.0
        cov = min(0.98, max(1e-3, float(sph.coverage)))
        alpha = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * cov))))
        alpha = min(150.0, alpha + 40.0)  # generous pad, still a cap
        lat0 = max(-89.0, 90.0 - alpha)
        try:
            tool = Part.makeSphere(R + eps, _vec(center), _vec(cap_axis),
                                   lat0, 90.0, 360.0)
        except Exception:  # noqa: BLE001 - fall back to the full ball below
            tool = None
    if tool is None:
        tool = Part.makeSphere(R + eps, _vec(center))
    if sph.outward:
        # A correct dome fuse adds only the sliver between the inscribed facets
        # and the true cap. For a DEEP cap (high solid-angle coverage) that
        # sliver is a small fraction of the tool, so the guard is tight. But a
        # legitimate SHALLOW dome/dish (low coverage — a gentle grille cap) is a
        # thin lens whose padded cap tool is mostly the sliver itself, so the
        # added fraction is naturally large; a fixed tight guard wrongly rejects
        # it (port_cover: 4/5 shallow caps failed to build). Scale the ceiling up
        # as coverage drops. The false-positive protection is preserved by the
        # tight fit gate at detection (RMS <= tol, tangency prior) plus the
        # caller's RTAF-improvement + bbox guards, not by this volume ratio alone.
        cov = max(1e-3, float(getattr(sph, "coverage", 1.0)))
        max_added = min(0.92, 0.5 + 0.45 * (1.0 - min(cov, 1.0)))
        return _guarded_fuse(solid, tool, max_added_frac=max_added)
    return _guarded_cut(solid, tool, max_removed_frac=0.6)


def _is_valid_solid(shape) -> bool:
    solids = getattr(shape, "Solids", [])
    return bool(solids) and solids[0].isValid()


def _heal_solid(shape, Part):
    """Try OCC ShapeFix (``Shape.fix``) at a couple of tolerances to turn a
    geometrically-invalid-but-closed shell into a valid solid. Returns the
    healed solid, or None if it can't be made valid. Only ever adopted when the
    result validates, so it can't make things worse."""
    for tol in (1e-3, 1e-2, 1e-1):
        try:
            s = shape.copy()
            s.fix(tol, tol, tol)
            sol = s if getattr(s, "Solids", []) else Part.Solid(s)
            if _is_valid_solid(sol):
                return sol
        except Exception:  # noqa: BLE001 - healing is best-effort
            pass
    return None


def _repair_nonmanifold(vertices: np.ndarray, faces: np.ndarray):
    """Repair non-manifold/duplicate mesh defects via FreeCAD's Mesh kernel.

    Uses only watertightness-safe ops (NOT fixSelfIntersections, which can open
    the mesh). Returns repaired ``(vertices, faces)``.
    """
    import Mesh  # type: ignore

    m = Mesh.Mesh()
    m.addFacets([(tuple(vertices[a]), tuple(vertices[b]), tuple(vertices[c]))
                 for a, b, c in faces])
    for op in ("removeDuplicatedPoints", "removeDuplicatedFacets",
               "removeNonManifolds", "removeNonManifoldPoints",
               "harmonizeNormals", "fixIndices"):
        try:
            getattr(m, op)()
        except Exception:  # noqa: BLE001
            pass
    points, facets = m.Topology
    verts = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    tris = np.array(facets, dtype=np.int64) if facets else faces
    return verts, tris


def _bbox_dims(shape):
    """Sorted (desc) bounding-box side lengths of a shape, as a tuple."""
    b = shape.BoundBox
    return tuple(sorted((b.XLength, b.YLength, b.ZLength), reverse=True))


def _bbox_grew(before, after, rel_tol: float, abs_tol: float = 0.05) -> bool:
    """True if ``after``'s bounding box is materially larger than ``before``'s.

    A hole cut can only remove material and a fuse-back trues up a boss over its
    own extent — neither should enlarge the part's overall silhouette. A mis-fit
    feature (a spurious giant tilted cylinder, an over-radius fillet) *does* grow
    the box, so any op that expands a side beyond ``rel_tol`` (relative) and
    ``abs_tol`` (mm, to ignore FP noise on tiny parts) is rejected.
    """
    for a, b in zip(after, before):
        if a - b > abs_tol and a - b > rel_tol * b:
            return True
    return False


# Fraction below which a boolean op's result bounding box (largest side) is
# treated as a catastrophic collapse. A legitimate cut/fuse trues up a local
# feature and can trim a boundary sliver, but never shrinks the part's overall
# silhouette by a third. A degenerate OCC boolean can return a valid-but-tiny
# fragment (gridfinity_base_lid: a cap fuse left a 6mm cube from a 210mm plate);
# this catches that regardless of which op produced it or how the per-op
# volume/added guards were configured. Deliberately loose so it only ever fires
# on true collapse, never on a legitimate edge trim.
_BBOX_COLLAPSE_FRAC = 0.5


def _bbox_collapsed(before, after) -> bool:
    """True if ``after``'s largest bbox side collapsed far below ``before``'s.

    Complements ``_bbox_grew``: growth is a mis-fit feature, collapse is a
    degenerate boolean returning a tiny valid fragment. Compares the LARGEST side
    only, so a thin part whose small sides legitimately change is unaffected —
    only a wholesale collapse of the dominant dimension trips it.
    """
    if not before or not after:
        return False
    b0, a0 = before[0], after[0]
    return b0 > 1e-6 and a0 < _BBOX_COLLAPSE_FRAC * b0


def _try_boolean_step(current_solid, fn, *, max_bbox_growth: float | None = None):
    """Apply one boolean cleanup step; revert if it breaks solid validity.

    Never lets a single bad feature corrupt or abort the whole result — later
    steps always see a known-good solid. When ``max_bbox_growth`` is given, also
    revert any op that enlarges the solid's bounding box beyond that relative
    fraction: a cut/fuse-back that grows the silhouette is a mis-detected feature
    (see _bbox_grew) and must not silently distort the part's dimensions.

    Independently of ``max_bbox_growth``, ALWAYS revert an op whose result
    collapses the bounding box (a degenerate OCC boolean returning a tiny valid
    fragment — gridfinity_base_lid's 210mm→6mm sphere fuse). Collapse is never a
    legitimate outcome, so this net applies to every op in both tiers.
    """
    try:
        candidate = fn(current_solid)
    except Exception:  # noqa: BLE001
        return current_solid, False
    solids = getattr(candidate, "Solids", [])
    if len(solids) != 1 or not solids[0].isValid():
        return current_solid, False
    try:
        if _bbox_collapsed(_bbox_dims(current_solid), _bbox_dims(candidate)):
            return current_solid, False
    except Exception:  # noqa: BLE001 - bbox read must not break the step
        pass
    if max_bbox_growth is not None:
        try:
            if _bbox_grew(_bbox_dims(current_solid), _bbox_dims(candidate),
                          max_bbox_growth):
                return current_solid, False
        except Exception:  # noqa: BLE001 - bbox read must not break the step
            pass
    return candidate, True


def build_boolean_clean_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig,
    on_progress=None,
):
    """Watertight faceted solid + boolean cut/fuse to make holes/cones analytic.

    More robust than sewing analytic and mesh-derived faces together (which
    needs matching topology): booleans recompute intersection geometry, so they
    tolerate the faceted mesh and the analytic tool disagreeing about exactly
    where their surfaces are. Always watertight if the base faceted solid is
    (each step reverts on failure), with far fewer faces than a fully faceted
    solid, and true round holes/bosses/countersinks.
    """
    import Part  # type: ignore

    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    # Each boolean cut costs O(base faces); on very dense meshes that is minutes
    # per hole, so bail early and let the caller fall through to a plain faceted
    # solid rather than spending many minutes. (Mesh decimation is the real fix.)
    limit = config.boolean_max_base_faces
    if limit is not None and len(faces) > limit:
        raise RuntimeError(
            f"mesh too dense for boolean clean-up ({len(faces):,} > {limit:,} "
            f"triangles); decimate the mesh or raise boolean_max_base_faces")

    progress("Detecting cylinders/holes")
    cylinders = detect_cylinders(vertices, faces, config)
    cones = detect_cones(vertices, faces, cylinders, config)
    progress(f"Found {len(cylinders)} cylinders, {len(cones)} cones")

    # Detector ladder (design §ladder placement): cylinders -> cones -> threads
    # -> knurling -> fillets -> ... . Threads and knurling claim their bands early
    # so they never mis-fit as swept walls or domes; both suppress to a cylinder
    # (metadata differs). Threads first (a single-family helix), then knurling
    # (bimodal crossing families) on whatever a thread didn't claim.
    claimed: set[int] = set()
    for c in cylinders:
        claimed.update(c.face_indices)
    for c in cones:
        claimed.update(c.face_indices)

    threads = _detect_threads(vertices, faces, cylinders, claimed, config, progress)
    for t in threads:
        claimed.update(t.face_indices)

    knurls = _detect_knurling(vertices, faces, cylinders, claimed, config, progress)
    for k in knurls:
        claimed.update(k.face_indices)

    # Gear / whole-outline profiles (M5.3): detected on the FULL mesh (the
    # claimed-subset swept pass fragments the tooth ring). Claim their facets so
    # the swept pass leaves them to the single whole-outline fuse.
    gear_profiles = _detect_gears(vertices, faces, config, progress)
    for g in gear_profiles:
        claimed.update(g.face_indices)

    fillets = _detect_fillets(vertices, faces, claimed, config, progress)
    for f in fillets:
        claimed.update(f.face_indices)

    # Spheres — DOME pass (M3): tessellated domes via cross-region consensus,
    # BEFORE swept walls, so a dome's latitude rows are claimed and M4 never fits
    # doomed lens ops to them. Corner blends are deferred to the BLEND pass below
    # (after swept) so a blend that is really part of a swept curved wall doesn't
    # cannibalise the sweep's facets and drop swept walls (drive_bay: 35 -> 28).
    spheres = _detect_spheres(vertices, faces, claimed, config, progress, mode="dome")
    for s in spheres:
        claimed.update(s.face_indices)

    # Freeform B-spline sheets (Candidate B): genuinely doubly-curved
    # height-field regions. Detected BEFORE swept walls (which would otherwise
    # mis-claim a doubly-curved shell's rows as constant-cross-section strips —
    # the double-curvature gate keeps single-curvature walls out, so this never
    # steals a legitimate sweep). Facets claimed here are removed from the swept
    # pool. Integrated by a guarded boolean below.
    freeform_sheets = []
    if config.fit_freeform_sheets:
        try:
            freeform_sheets = fit_freeform_sheets(vertices, faces, claimed, config)
        except Exception as exc:  # noqa: BLE001 - detection is best-effort
            progress(f"Freeform sheet detection skipped ({exc})")
            freeform_sheets = []
        # Only CONFIDENT sheets pre-empt the swept pool. A well-covered region
        # (low ``missing``) is a genuine height field the sweep would mis-fit, so
        # claiming it protects the sheet. A marginal / heavily-extrapolated sheet
        # (a split sub-region with a large inpainted skirt) must NOT steal facets
        # from swept walls that de-facet more reliably — it is still ATTEMPTED for
        # building below, but swept gets first crack and, since both ops are
        # guarded + RTAF-gated and run in order, whichever improves the surface
        # wins and the other reverts (task §1: never trade a working swept wall
        # for a doomed sheet). This is what stops an aggressive region split from
        # collapsing swept coverage (port_cover: 44 swept walls -> 0).
        claim_thr = config.freeform_claim_max_missing
        claimed_sheets = 0
        for sh in freeform_sheets:
            if sh.missing <= claim_thr:
                claimed.update(sh.face_indices)
                claimed_sheets += 1
        if freeform_sheets:
            progress(f"Found {len(freeform_sheets)} freeform sheet region(s) "
                     f"({sum(len(s.face_indices) for s in freeform_sheets)} facets; "
                     f"{claimed_sheets} confident, claimed from swept pool)")

    # Swept/extruded curved walls (M4), after fillets in the detector ladder.
    # Fitted on the facets no other detector claimed; each arc segment becomes a
    # boolean lens op (cut/fuse) against the faceted base below.
    swept_profiles: list[SweptProfile] = []
    if config.detect_swept_walls:
        keep_sw = [i for i in range(len(faces)) if i not in claimed]
        faces_sw = faces[keep_sw]
        try:
            regions_sw = segment_planar(vertices, faces_sw, config)
        except Exception:  # noqa: BLE001 - detection is best-effort
            regions_sw = []
        if regions_sw:
            swept_profiles = _fit_swepts(vertices, faces_sw, regions_sw, config, progress)
            # Claim the swept walls' facets (mapped back to original indices) so
            # the blend pass below leaves swept-wall corners to the sweep.
            for prof in swept_profiles:
                claimed.update(keep_sw[i] for i in prof.face_indices)

    # Spheres — BLEND pass (M3): compact corner blends / end-cap domes on the
    # facets no cylinder/cone/fillet/dome/sweep claimed.
    blend_spheres = _detect_spheres(vertices, faces, claimed, config, progress,
                                    mode="blend")
    spheres = spheres + blend_spheres
    for s in blend_spheres:
        claimed.update(s.face_indices)

    progress("Building faceted watertight solid (base)")
    solid = build_faceted_solid(vertices, faces)
    if not _is_valid_solid(solid):
        # Decimation can leave non-manifold edges; repair them (FreeCAD's Mesh
        # kernel, safe ops only — no fixSelfIntersections, which breaks
        # watertightness) and rebuild before giving up.
        progress("  base not watertight; repairing non-manifold edges")
        rv, rf = _repair_nonmanifold(vertices, faces)
        solid = build_faceted_solid(rv, rf)
    if not _is_valid_solid(solid):
        # Self-intersecting / overlapping-body mesh (exported without a final
        # boolean union): re-solve the true outer surface with manifold3d's
        # winding-number boolean. Detection already ran on the original mesh and
        # the cut tools are purely geometric, so they apply to the resolved base.
        progress("  still invalid; resolving self-intersections (boolean union)")
        from .meshprep import resolve_self_intersections

        resolved = resolve_self_intersections(vertices, faces, on_progress=progress)
        if resolved is not None:
            rv, rf, rep = resolved
            candidate = build_faceted_solid(rv, rf)
            if _is_valid_solid(candidate):
                solid = candidate
                progress(f"  resolved: {rep['bodies']} overlapping bodies unioned "
                         f"({rep['faces_in']:,} -> {rep['faces_out']:,} facets)")
    if not _is_valid_solid(solid):
        # Last resort: OCC shape healing (ShapeFix) can close a shell that is
        # topologically closed but geometrically invalid (small self-touches).
        progress("  still invalid; attempting OCC shape healing")
        healed = _heal_solid(solid, Part)
        if healed is not None:
            solid = healed
    if not _is_valid_solid(solid):
        raise RuntimeError("base faceted solid is not watertight; cannot boolean-clean")

    # Per-feature cut + fuse-back, each reverting if it breaks validity. Applied
    # sequentially (not batched) because coaxial/nested features — e.g. a bore
    # inside a boss — must be processed in order: a batched fuse-back of the boss
    # would fill the bore. One bad feature never corrupts the rest.
    # Bounding-box growth guard: revert any cut/fuse-back that enlarges the
    # part's silhouette (a mis-detected feature — e.g. a spurious giant tilted
    # cylinder — otherwise distorts the exported dimensions by 10-30%).
    bbox_guard = config.boolean_max_bbox_growth

    cyl_ok = 0
    for i, cyl in enumerate(cylinders):
        r_cut = _design_radius(vertices, faces, cyl.axis_dir, cyl.axis_point,
                               cyl.face_indices, cyl.radius)
        solid, ok = _try_boolean_step(
            solid, lambda s, c=cyl, rc=r_cut: _boolean_clean_cylinder(s, c, Part, radius=rc),
            max_bbox_growth=bbox_guard)
        cyl_ok += ok
        if (i + 1) % 10 == 0 or i + 1 == len(cylinders):
            progress(f"  cylinders cleaned {cyl_ok}/{i + 1} of {len(cylinders)}")

    cone_ok = 0
    for cone in cones:
        solid, ok = _try_boolean_step(
            solid, lambda s, c=cone: _boolean_clean_cone(s, c, Part),
            max_bbox_growth=bbox_guard)
        cone_ok += ok

    # Threads (M5.2) then knurls (M5.1): suppress each band to its nominal
    # cylinder (pitch diameter for threads, mid-surface for knurls). Boss-> fuse,
    # bore/internal-> cut, each reverting on invalidity. Metadata is captured
    # separately in stats; here we only true up the geometry.
    thread_ok = 0
    for th in threads:
        solid, ok = _try_boolean_step(
            solid, lambda s, t=th: _boolean_clean_thread(s, t, Part),
            max_bbox_growth=bbox_guard)
        thread_ok += ok
    if threads:
        progress(f"  threads suppressed {thread_ok}/{len(threads)}")

    knurl_ok = 0
    for kn in knurls:
        solid, ok = _try_boolean_step(
            solid, lambda s, k=kn: _boolean_clean_knurl(s, k, Part),
            max_bbox_growth=bbox_guard)
        knurl_ok += ok
    if knurls:
        progress(f"  knurls suppressed {knurl_ok}/{len(knurls)}")

    # Fillets last: a concave fillet cuts, a convex one fuses its rounded-corner
    # sector, each reverting on invalidity so a bad fillet never breaks the solid.
    fillet_ok = 0
    for fl in fillets:
        solid, ok = _try_boolean_step(
            solid, lambda s, f=fl: _boolean_clean_fillet(s, f, Part),
            max_bbox_growth=bbox_guard)
        fillet_ok += ok
    if fillets:
        progress(f"  fillets cleaned {fillet_ok}/{len(fillets)}")

    # Swept walls last: one lens op (cut for concave, guarded fuse for convex)
    # per fitted arc segment, each reverting on invalidity. If the whole batch
    # introduces artifact radii the pre-swept solid didn't have (seen when lens
    # ops nibble at a spherical dome's latitude rows — M3 geometry), the batch
    # is rolled back wholesale: swept reconstruction must never turn an
    # artifact-free part into a dual-output one.
    detected_r = sorted({round(c.radius, 3) for c in cylinders}
                        | {round(f.radius, 3) for f in fillets}
                        | {round(t.nominal_radius, 3) for t in threads}
                        | {round(t.suppress_radius, 3) for t in threads}
                        | {round(k.nominal_radius, 3) for k in knurls}
                        | {round(k.suppress_radius, 3) for k in knurls}
                        | {round(seg.radius, 3) for p in swept_profiles
                           for seg in p.segments if seg.kind == "arc"})
    # Gear whole-outline extrusions (M5.3): a repeated-arc region wrapping the
    # axis (detected up-front on the full mesh) is fused as ONE extruded solid,
    # BEFORE the per-arc lens ops. The fuse fills any central bore (the outline is
    # a solid disk with teeth), so the bore/hole cylinders are RE-CUT afterwards
    # (design: bore cut after gear fuse, ladder discipline). Guarded throughout.
    gear_ops, gear_ok = 0, 0
    if gear_profiles:
        solid, gear_ops, gear_ok = _apply_gear_extrusions(
            solid, gear_profiles, Part, progress, bbox_guard=bbox_guard, config=config)
        if gear_ok:
            # Re-cut the internal (hole) cylinders the gear fuse filled.
            for cyl in cylinders:
                if cyl.outward:
                    continue
                r_cut = _design_radius(vertices, faces, cyl.axis_dir, cyl.axis_point,
                                       cyl.face_indices, cyl.radius)
                solid, _ok = _try_boolean_step(
                    solid, lambda s, c=cyl, rc=r_cut: _boolean_clean_cylinder(
                        s, c, Part, radius=rc),
                    max_bbox_growth=bbox_guard)

    pre_swept = solid.copy()
    pre_rogue = set(_rogue_radii(solid, detected_r))
    solid, swept_ops, swept_ok = _apply_swept_lens_ops(
        solid, swept_profiles, Part, progress, config=config)

    # removeSplitter is an optimization; on shells dense with fresh boolean
    # seams it can occasionally produce an invalid solid — AND the underlying
    # OCC call can corrupt the *input* shape in place (shared internal shape
    # data), so simplify a snapshot and keep whichever is a valid solid.
    backup = solid.copy()
    simplified = _safe_remove_splitter(solid, Part)
    solid = simplified if _is_valid_solid(simplified) else backup
    solid, slivers_removed = _defeature_sliver_chains(solid, config, Part, progress)

    if swept_ok and set(_rogue_radii(solid, detected_r)) - pre_rogue:
        progress("  swept ops introduced artifact radii; rolling back swept ops")
        swept_ok = 0
        backup = pre_swept.copy()
        simplified = _safe_remove_splitter(pre_swept, Part)
        solid = simplified if _is_valid_solid(simplified) else backup
        solid, slivers_removed = _defeature_sliver_chains(solid, config, Part, progress)

    # Spheres (M3) after the swept block: a sphere fuse reshapes wall geometry a
    # swept lens op keys off, so doing spheres first can make the swept ops miss
    # (seen in the sew tier on the tweezer). A convex dome fuses its cap ball, a
    # concave dish cuts, each reverting on invalidity so a bad sphere never breaks
    # the solid.
    sphere_ok = 0
    if spheres and _is_valid_solid(solid):
        solid, sphere_ok = _apply_sphere_ball_ops(
            solid, spheres, Part, progress, bbox_guard=bbox_guard, config=config)
        progress(f"  spheres cleaned {sphere_ok}/{len(spheres)}")

    # Freeform B-spline sheets (Candidate B): last, on a valid solid, each
    # adopted only if it validates, stays bbox-stable, and lowers RTAF.
    freeform_ops = 0
    freeform_ok = 0
    if freeform_sheets and _is_valid_solid(solid):
        solid, freeform_ops, freeform_ok = _apply_freeform_sheets(
            solid, freeform_sheets, config, Part, progress,
            vertices=vertices, faces=faces)
        if freeform_ok:
            backup = solid.copy()
            simplified = _safe_remove_splitter(solid, Part)
            solid = simplified if _is_valid_solid(simplified) else backup
            progress(f"  freeform sheets built {freeform_ok}/{freeform_ops}")

    total = (len(cylinders) + len(cones) + len(threads) + len(knurls)
             + len(fillets) + len(spheres) + swept_ops + gear_ops + freeform_ops)
    cleaned = (cyl_ok + cone_ok + thread_ok + knurl_ok + fillet_ok
               + sphere_ok + swept_ok + gear_ok + freeform_ok)
    progress(f"Boolean clean-up: {cleaned}/{total} features replaced with analytic geometry "
             f"({total - cleaned} left faceted)")

    solids = getattr(solid, "Solids", [])
    is_solid = bool(solids) and solids[0].isValid()

    # Artifact check: every cylindrical face in the result should sit at (near)
    # a detected hole radius. Oversize-cut rings or partial-radius slivers left
    # by booleans on intersecting holes show up as faces at *other* radii — those
    # cause downstream CAD issues, so we flag them and the caller rejects the
    # result rather than shipping artifacts.
    rogue = _rogue_radii(solid, detected_r)
    artifact_free = len(rogue) == 0

    stats = {
        "faces_in": int(len(faces)),
        "faces_out": len(solid.Faces),
        "cylinders_detected": len(cylinders),
        "cylinder_faces": cyl_ok,
        "cones_detected": len(cones),
        "cone_faces": cone_ok,
        "threads_detected": len(threads),
        "threads_built": thread_ok,
        "threads": [t.as_dict() for t in threads],
        "knurling_detected": len(knurls),
        "knurling_built": knurl_ok,
        "knurling": [k.as_dict() for k in knurls],
        "fillets_detected": len(fillets),
        "fillet_faces": fillet_ok,
        "fillet_radius_source": _radius_source_breakdown(fillets),
        "spheres_detected": len(spheres),
        "spheres_built": sphere_ok,
        "spheres": [s.as_dict() for s in spheres],
        "cylinders": [c.as_dict() for c in cylinders],
        "cones": [c.as_dict() for c in cones],
        "fillets": [f.as_dict() for f in fillets],
        "swept_walls_detected": len(swept_profiles),
        "swept_walls_built": swept_ok,
        "swept_arc_ops": swept_ops,
        "gears_detected": len(gear_profiles),
        "gears_built": gear_ok,
        "gears": [{"segments": len(p.segments), "arcs": p.n_arcs,
                   "lines": p.n_lines, "splines": p.n_splines,
                   "extent": round(p.axial_max - p.axial_min, 3),
                   "closed": p.closed} for p in gear_profiles],
        "swept_tangency_snaps": sum(p.tangency_snaps for p in swept_profiles),
        "swept_slivers_removed": slivers_removed,
        "swept_detail": [p.as_dict() for p in swept_profiles],
        "freeform_sheets_detected": len(freeform_sheets),
        "freeform_sheets_built": freeform_ok,
        "freeform_detail": [s.as_dict() for s in freeform_sheets],
        "boolean_cleaned": cleaned,
        "boolean_failed": total - cleaned,
        "artifact_free": artifact_free,
        "rogue_radii": sorted(set(rogue)),
        "is_solid": is_solid,
    }
    return solid, stats


def _rogue_radii(solid, detected_r) -> list:
    """Cylinder-face radii in ``solid`` that match no detected feature radius."""
    rogue = []
    try:
        for fc in solid.Faces:
            if fc.Surface.TypeId == "Part::GeomCylinder":
                r = fc.Surface.Radius
                if not any(abs(r - d) <= 0.05 * d + 0.05 for d in detected_r):
                    rogue.append(round(r, 3))
    except Exception:  # noqa: BLE001 - a scan failure must not break the build
        pass
    return sorted(set(rogue))


def _face_plane_normal(face) -> np.ndarray:
    """Unit normal of a planar face's surface (its plane Axis)."""
    n = face.Surface.Axis
    v = np.array([n.x, n.y, n.z], float)
    return v / (np.linalg.norm(v) or 1.0)


def _defeature_sliver_chains(solid, config: ConversionConfig, Part, progress):
    """Remove micro-sliver planar faces that drag large flats into smooth
    chains (M4 cleanup; see docs/CURVED_FEATURES.md §6a).

    Decimation and boolean seams leave near-zero-area planar wedges tilted a
    few degrees off an adjacent big flat. Geometrically they are noise, but
    the RTAF smooth-chain construction (correctly) links them to the flat —
    one 0.05 mm^2 sliver can mark a 20,000 mm^2 wall as residual tessellation.
    OCC defeaturing removes the sliver faces and heals the gap by extending
    their neighbours; since only chains consisting of ONE dominant face plus a
    handful of sub-``swept_sliver_max_area`` slivers are touched, the healed
    geometry differs by a sliver-sized amount (volume-guarded, reverts
    wholesale on any doubt).

    Returns ``(solid, n_removed)``.
    """
    if not (config.detect_swept_walls and config.swept_defeature_slivers):
        return solid, 0
    try:
        faces = list(solid.Faces)
    except Exception:  # noqa: BLE001
        return solid, 0
    n = len(faces)
    if n == 0 or n > 20000:
        return solid, 0
    try:
        is_plane = [f.Surface.TypeId == "Part::GeomPlane" for f in faces]
        normals = [_face_plane_normal(faces[i]) if is_plane[i] else None
                   for i in range(n)]
        edge_map: dict = {}
        for fi, f in enumerate(faces):
            for e in f.Edges:
                k = _rtaf_edge_key(e)
                if k is not None:
                    edge_map.setdefault(k, set()).add(fi)
        lo, hi = float(config.rtaf_angle_lo), float(config.rtaf_angle_hi)
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for fs in edge_map.values():
            fl = sorted(fs)
            for a_i in range(len(fl)):
                for b_i in range(a_i + 1, len(fl)):
                    a, b = fl[a_i], fl[b_i]
                    if not (is_plane[a] and is_plane[b]):
                        continue
                    d = min(abs(float(normals[a] @ normals[b])), 1.0)
                    ang = math.degrees(math.acos(d))
                    if lo < ang < hi:
                        adj[a].add(b)
                        adj[b].add(a)
        seen: set[int] = set()
        slivers: list[int] = []
        max_area = float(config.swept_sliver_max_area)
        for i in range(n):
            if i in seen or not adj[i]:
                continue
            comp, stack = [], [i]
            seen.add(i)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if len(comp) < config.rtaf_min_chain:
                continue
            small = [x for x in comp if faces[x].Area < max_area]
            # Only the unambiguous case: one dominant face + a few slivers.
            if 0 < len(small) <= 4 and len(small) == len(comp) - 1:
                slivers.extend(small)
        if not slivers or len(slivers) > 80:
            return solid, 0
        candidate = solid.defeaturing([faces[i] for i in slivers])
        solids = getattr(candidate, "Solids", [])
        if len(solids) != 1 or not solids[0].isValid():
            return solid, 0
        if abs(candidate.Volume - solid.Volume) > 5.0:
            return solid, 0
        _check_no_self_intersection(candidate)
        progress(f"  defeatured {len(slivers)} sliver face(s) off smooth chains")
        return candidate, len(slivers)
    except Exception:  # noqa: BLE001 - cleanup is best-effort; keep the solid
        return solid, 0


def _rtaf_edge_key(edge, ndig: int = 4):
    """A tolerance-rounded key identifying an edge by its endpoints + midpoint.

    Two faces share an edge when they contribute an edge with the same key; the
    rounding bridges the FP noise between a face's copy of a shared edge and its
    neighbour's. Mirrors the prototype (diagnosis/step_strips.py)."""
    try:
        vs = edge.Vertexes
        if len(vs) >= 2:
            a = (round(vs[0].X, ndig), round(vs[0].Y, ndig), round(vs[0].Z, ndig))
            b = (round(vs[1].X, ndig), round(vs[1].Y, ndig), round(vs[1].Z, ndig))
            lo, hi = (a, b) if a <= b else (b, a)
        else:  # closed edge (full circle) — no endpoints
            c = edge.CenterOfMass
            lo = hi = (round(c.x, ndig), round(c.y, ndig), round(c.z, ndig))
        m = edge.CenterOfMass
        return (lo, hi, (round(m.x, ndig), round(m.y, ndig), round(m.z, ndig)))
    except Exception:  # noqa: BLE001
        return None


def compute_rtaf(shape, config: ConversionConfig) -> dict:
    """Residual Tessellation Area Fraction of an output solid (design §6a).

    RTAF = area(planar faces in smooth chains of length >= rtaf_min_chain) /
           total face area, where a *smooth chain* is a connected component over
           shared edges of planar faces whose pairwise normal angle is in
           (rtaf_angle_lo, rtaf_angle_hi) degrees. This captures the area a human
           reads as "faceted" — a tessellated curve arriving as a fan of
           near-tangent flat panels — regardless of whether the pipeline counted
           the underlying facets as skipped (gap-filled patches and built-as-thin
           -strips both produce the signature, so one number catches both).

    Returns a dict with ``rtaf`` (float in [0,1]) plus supporting counts, or
    ``{"rtaf": None, "skipped": <reason>}`` when disabled/too large/failed. Never
    raises — quality metrics must not break a conversion.
    """
    out: dict = {"rtaf": None}
    if not config.compute_rtaf:
        out["skipped"] = "disabled"
        return out
    try:
        faces = list(shape.Faces)
    except Exception as exc:  # noqa: BLE001
        out["skipped"] = f"no faces ({exc})"
        return out
    n = len(faces)
    if n == 0:
        out["skipped"] = "no faces"
        return out
    cap = config.rtaf_max_faces
    if cap is not None and n > cap:
        out.update(skipped=f"face count {n} > rtaf_max_faces {cap}", faces=n)
        return out

    try:
        is_plane = [f.Surface.TypeId == "Part::GeomPlane" for f in faces]
        areas = [float(f.Area) for f in faces]
        total_area = float(sum(areas))
        normals = [_face_plane_normal(faces[i]) if is_plane[i] else None
                   for i in range(n)]

        # Build face adjacency by shared-edge key.
        edge_map: dict = {}
        for fi, f in enumerate(faces):
            for e in f.Edges:
                k = _rtaf_edge_key(e)
                if k is not None:
                    edge_map.setdefault(k, set()).add(fi)

        lo = float(config.rtaf_angle_lo)
        hi = float(config.rtaf_angle_hi)
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for fs in edge_map.values():
            fl = sorted(fs)
            for a_i in range(len(fl)):
                for b_i in range(a_i + 1, len(fl)):
                    a, b = fl[a_i], fl[b_i]
                    if not (is_plane[a] and is_plane[b]):
                        continue
                    d = min(abs(float(normals[a] @ normals[b])), 1.0)
                    ang = math.degrees(math.acos(d))
                    if lo < ang < hi:
                        adj[a].add(b)
                        adj[b].add(a)

        # Connected components of smooth-linked planar faces; keep those >= min.
        seen: set[int] = set()
        chain_faces: set[int] = set()
        n_chains = 0
        largest = 0
        for i in range(n):
            if i in seen or not adj[i]:
                continue
            comp, stack = [], [i]
            seen.add(i)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if len(comp) >= config.rtaf_min_chain:
                n_chains += 1
                largest = max(largest, len(comp))
                chain_faces.update(comp)

        chain_area = float(sum(areas[i] for i in chain_faces))
        rtaf = chain_area / total_area if total_area > 1e-12 else 0.0
        out.update(
            rtaf=round(rtaf, 4),
            faces=n,
            planar_faces=int(sum(is_plane)),
            chain_faces=len(chain_faces),
            smooth_chains=n_chains,
            largest_chain=largest,
            total_area_mm2=round(total_area, 1),
        )
    except Exception as exc:  # noqa: BLE001 - metric is best-effort
        out.update(rtaf=None, skipped=f"error ({exc})")
    return out


def export_step(shape, out_path: str | Path) -> None:
    """Write ``shape`` to a STEP file."""
    shape.exportStep(str(out_path))


def revalidate_step(out_path: str | Path, expected_solids: int | None = None) -> dict:
    """Re-read an exported STEP and check the solid(s) survived the round-trip.

    Some defects — most notably self-intersecting wires produced by sliver
    triangles after aggressive decimation — pass ``isValid()`` in memory but
    only manifest when OCC writes the shape to STEP and reads it back (the
    write/read re-derives wire geometry). This re-reads the file and confirms:
    it loads, has at least one solid, the first solid ``isValid()``, and — when
    ``expected_solids`` is given — the solid count matches what we wrote.

    Returns a dict with ``valid`` (bool) and, on failure, a ``reason`` string.
    Never raises for geometry problems (they are reported via the dict); only a
    genuinely unreadable file surfaces as ``valid=False`` with the read error.
    """
    import Part  # type: ignore

    info: dict = {"path": str(out_path)}
    try:
        shape = Part.Shape()
        shape.read(str(out_path))
    except Exception as exc:  # noqa: BLE001
        info.update(valid=False, reason=f"STEP unreadable: {exc}")
        return info

    solids = getattr(shape, "Solids", [])
    info["num_solids"] = len(solids)
    if not solids:
        info.update(valid=False, reason="no solids in re-read STEP")
        return info
    if not solids[0].isValid():
        info.update(valid=False, reason="re-read solid isValid() is False")
        return info
    if expected_solids is not None and len(solids) != expected_solids:
        info.update(valid=False,
                    reason=f"solid count changed ({expected_solids} -> {len(solids)})")
        return info
    info["valid"] = True
    return info


def _reread_valid(shape, Part, progress=None) -> bool:
    """Write ``shape`` to a temp STEP and confirm it re-reads as a valid solid.

    Used by the reconstructed tier's export back-off (P0-3): some boolean-op
    defects pass ``isValid()`` in memory but re-read invalid from a written STEP.
    Best-effort — a write/read error is treated as "not valid" so the caller
    backs off conservatively. Never raises.
    """
    import tempfile

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as fh:
            tmp = fh.name
        shape.exportStep(tmp)
        reread = Part.Shape()
        reread.read(tmp)
        solids = getattr(reread, "Solids", [])
        return bool(solids) and solids[0].isValid()
    except Exception as exc:  # noqa: BLE001
        if progress is not None:
            progress(f"  re-read check failed ({exc}); treating as invalid")
        return False
    finally:
        if tmp is not None:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
