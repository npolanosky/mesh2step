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


def export_step(shape, out_path: str | Path) -> None:
    """Write ``shape`` to a STEP file."""
    shape.exportStep(str(out_path))
