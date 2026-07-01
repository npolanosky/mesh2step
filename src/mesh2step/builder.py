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
from .fitting import Cylinder, _connected_components, detect_cones, detect_cylinders
from .segmentation import build_edge_adjacency, segment_planar


def _vec(p):
    import FreeCAD  # type: ignore  # local import; module only runs under FreeCAD

    return FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))


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
    """
    normal = loops.normal

    def wire_for(loop, is_hole):
        match = _match_loop_to_circle(loop, normal, circles)
        if match is not None:
            center, n, radius = match
            return _circle_wire(center, -n if is_hole else n, radius, Part)
        return _wire_from_points(loop, Part)

    wires = [wire_for(loops.outer, is_hole=False)]
    for hole in loops.holes:
        if len(hole) >= 3:
            wires.append(wire_for(hole, is_hole=True))
    return Part.Face(wires)


def _cylinder_face(cyl: Cylinder, Part):
    """Build an analytic cylindrical face trimmed to the region's axial span.

    A boss keeps the surface's natural outward normal; a hole is reversed so the
    solid's outward normal points into the material (out of the bore).
    """
    surf = Part.Cylinder()
    surf.Center = _vec(cyl.axis_point)
    surf.Axis = _vec(cyl.axis_dir)
    surf.Radius = float(cyl.radius)
    face = surf.toShape(0.0, 2.0 * math.pi, cyl.axial_min, cyl.axial_max)
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

    # Exact end-circles of the analytic faces, used to replace matching faceted
    # boundary loops so their edges coincide and sew.
    circles = _analytic_circles(cylinders, cones)

    # Segment the *remaining* facets into planar regions. Removing the cylinder
    # walls turns each hole into a clean inner boundary loop on its end faces.
    keep = [i for i in range(len(faces)) if i not in claimed]
    faces_sub = faces[keep]

    progress("Segmenting planar regions")
    regions = segment_planar(vertices, faces_sub, config)
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

    stats = {
        "faces_in": int(len(faces)),
        "planar_faces": reconstructed,
        "cylinder_faces": cyl_faces_ok,
        "cone_faces": cone_faces_ok,
        "gap_faces": gap_faces,
        "gap_patches": gap_patches,
        "cylinders_detected": len(cylinders),
        "faces_out": reconstructed + cyl_faces_ok + cone_faces_ok + gap_faces,
        "cylinders": [c.as_dict() for c in cylinders],
        "cones_detected": len(cones),
        "cones": [c.as_dict() for c in cones],
        "skipped_facets": skipped,
        "is_solid": is_solid,
    }
    return shape, stats


def _safe_remove_splitter(shape, Part):
    """removeSplitter is an optimization (merge coplanar faces); never let a
    malformed edge/curve in a huge shell crash the whole reconstruction."""
    try:
        return shape.removeSplitter()
    except Exception:  # noqa: BLE001
        return shape


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

    shell = Part.Shell(occ_faces)
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
        fill = _boolean_cut_tool_cylinder(center, axis, R + eps, zmin, zmax, Part)
        return solid.fuse(fill)
    pad = max(zmax - zmin, 1.0)
    cut = _boolean_cut_tool_cylinder(center, axis, R + eps, zmin - pad, zmax + pad, Part)
    return solid.cut(cut)


def _boolean_clean_cone(solid, cone, Part, **_):
    """Exact-cut analogue of :func:`_boolean_clean_cylinder` for a countersink
    cone. The faceted cone is inscribed the same way, so cutting the exact cone
    (extended a hair past each end along its own taper so it passes cleanly
    through the surfaces) clears the facets and leaves an analytic conical wall
    at the fitted radii — no oversize ring."""
    r_base = float(cone.r_base)
    r_top = float(cone.r_top)
    axis = np.asarray(cone.axis_dir, dtype=float)
    center = np.asarray(cone.axis_point, dtype=float)
    zmin, zmax = float(cone.axial_min), float(cone.axial_max)
    height = zmax - zmin
    pad = max(height, 1.0)
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


def _is_valid_solid(shape) -> bool:
    solids = getattr(shape, "Solids", [])
    return bool(solids) and solids[0].isValid()


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


def _try_boolean_step(current_solid, fn):
    """Apply one boolean cleanup step; revert if it breaks solid validity.

    Never lets a single bad feature corrupt or abort the whole result — later
    steps always see a known-good solid.
    """
    try:
        candidate = fn(current_solid)
    except Exception:  # noqa: BLE001
        return current_solid, False
    solids = getattr(candidate, "Solids", [])
    if len(solids) == 1 and solids[0].isValid():
        return candidate, True
    return current_solid, False


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
        raise RuntimeError("base faceted solid is not watertight; cannot boolean-clean")

    # Per-feature cut + fuse-back, each reverting if it breaks validity. Applied
    # sequentially (not batched) because coaxial/nested features — e.g. a bore
    # inside a boss — must be processed in order: a batched fuse-back of the boss
    # would fill the bore. One bad feature never corrupts the rest.
    cyl_ok = 0
    for i, cyl in enumerate(cylinders):
        r_cut = _design_radius(vertices, faces, cyl.axis_dir, cyl.axis_point,
                               cyl.face_indices, cyl.radius)
        solid, ok = _try_boolean_step(
            solid, lambda s, c=cyl, rc=r_cut: _boolean_clean_cylinder(s, c, Part, radius=rc))
        cyl_ok += ok
        if (i + 1) % 10 == 0 or i + 1 == len(cylinders):
            progress(f"  cylinders cleaned {cyl_ok}/{i + 1} of {len(cylinders)}")

    cone_ok = 0
    for cone in cones:
        solid, ok = _try_boolean_step(solid, lambda s, c=cone: _boolean_clean_cone(s, c, Part))
        cone_ok += ok

    total = len(cylinders) + len(cones)
    cleaned = cyl_ok + cone_ok
    progress(f"Boolean clean-up: {cleaned}/{total} features replaced with analytic geometry "
             f"({total - cleaned} left faceted)")

    solid = _safe_remove_splitter(solid, Part)
    solids = getattr(solid, "Solids", [])
    is_solid = bool(solids) and solids[0].isValid()

    # Artifact check: every cylindrical face in the result should sit at (near)
    # a detected hole radius. Oversize-cut rings or partial-radius slivers left
    # by booleans on intersecting holes show up as faces at *other* radii — those
    # cause downstream CAD issues, so we flag them and the caller rejects the
    # result rather than shipping artifacts.
    detected_r = sorted({round(c.radius, 3) for c in cylinders})
    rogue = []
    for fc in solid.Faces:
        if fc.Surface.TypeId == "Part::GeomCylinder":
            r = fc.Surface.Radius
            if not any(abs(r - d) <= 0.05 * d + 0.05 for d in detected_r):
                rogue.append(round(r, 3))
    artifact_free = len(rogue) == 0

    stats = {
        "faces_in": int(len(faces)),
        "faces_out": len(solid.Faces),
        "cylinders_detected": len(cylinders),
        "cylinder_faces": cyl_ok,
        "cones_detected": len(cones),
        "cone_faces": cone_ok,
        "cylinders": [c.as_dict() for c in cylinders],
        "cones": [c.as_dict() for c in cones],
        "boolean_cleaned": cleaned,
        "boolean_failed": total - cleaned,
        "artifact_free": artifact_free,
        "rogue_radii": sorted(set(rogue)),
        "is_solid": is_solid,
    }
    return solid, stats


def export_step(shape, out_path: str | Path) -> None:
    """Write ``shape`` to a STEP file."""
    shape.exportStep(str(out_path))
