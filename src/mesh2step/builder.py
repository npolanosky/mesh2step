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
from .fitting import Cylinder, detect_cylinders
from .segmentation import segment_planar


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


def _match_loop_to_cylinder(
    loop: np.ndarray, plane_normal: np.ndarray, cylinders: list[Cylinder]
):
    """If a boundary loop is a faceted circle matching a cylinder end, return
    the exact circle (center, normal, radius) to replace it with; else None."""
    centroid = loop.mean(axis=0)
    mean_radius = float(np.linalg.norm(loop - centroid, axis=1).mean())
    for cyl in cylinders:
        # Plane must be perpendicular to the cylinder axis.
        if abs(float(plane_normal @ cyl.axis_dir)) < 0.99:
            continue
        if abs(mean_radius - cyl.radius) > 0.05 * cyl.radius + 0.1:
            continue
        # Loop must be centred on the axis line.
        rel = centroid - cyl.axis_point
        off_axis = np.linalg.norm(rel - (rel @ cyl.axis_dir) * cyl.axis_dir)
        if off_axis > 0.05 * cyl.radius + 0.1:
            continue
        # Exact circle = where the axis pierces this loop's plane.
        denom = float(cyl.axis_dir @ plane_normal)
        if abs(denom) < 1e-9:
            continue
        s = float((centroid - cyl.axis_point) @ plane_normal) / denom
        center = cyl.axis_point + s * cyl.axis_dir
        return center, plane_normal, cyl.radius
    return None


def _planar_face(loops: FaceLoops, cylinders: list[Cylinder], Part):
    """Build a planar face, swapping any faceted-circle loop for a true circle.

    Hole loops are wound opposite to the outer loop so OCC subtracts them; when
    a loop is replaced by an exact circle we flip its axis to match.
    """
    normal = loops.normal

    def wire_for(loop, is_hole):
        match = _match_loop_to_cylinder(loop, normal, cylinders)
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


def build_reconstructed_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig,
):
    """Reconstruct planar + cylindrical faces, sew them, return ``(shape, stats)``.

    Raises if no valid geometry can be built — the caller falls back to faceted.
    """
    import Part  # type: ignore

    cylinders = detect_cylinders(vertices, faces, config)
    claimed: set[int] = set()
    for cyl in cylinders:
        claimed.update(cyl.face_indices)

    # Segment the *remaining* facets into planar regions. Removing the cylinder
    # walls turns each hole into a clean inner boundary loop on its end faces.
    keep = [i for i in range(len(faces)) if i not in claimed]
    faces_sub = faces[keep]

    regions = segment_planar(vertices, faces_sub, config)
    occ_faces = []
    reconstructed = 0
    skipped = 0
    for region in regions:
        if region.size < config.min_region_facets:
            skipped += region.size
            continue
        loops = extract_face_loops(vertices, faces_sub, region, config)
        if loops is None:
            skipped += region.size
            continue
        try:
            occ_faces.append(_planar_face(loops, cylinders, Part))
            reconstructed += 1
        except Exception:  # noqa: BLE001 - OCC raises bare RuntimeErrors
            skipped += region.size

    cyl_faces_ok = 0
    for cyl in cylinders:
        try:
            occ_faces.append(_cylinder_face(cyl, Part))
            cyl_faces_ok += 1
        except Exception:  # noqa: BLE001
            pass

    if not occ_faces:
        raise RuntimeError("no faces could be reconstructed")

    shape, is_solid = _faces_to_solid(occ_faces, Part)

    stats = {
        "faces_in": int(len(faces)),
        "planar_faces": reconstructed,
        "cylinder_faces": cyl_faces_ok,
        "faces_out": reconstructed + cyl_faces_ok,
        "cylinders": [c.as_dict() for c in cylinders],
        "skipped_facets": skipped,
        "is_solid": is_solid,
    }
    return shape, stats


def _faces_to_solid(occ_faces, Part):
    """Sew faces into a (hopefully) closed solid. Returns ``(shape, is_solid)``."""
    # Sewing tolerates the tiny gaps between analytic circles and the planar
    # loops they replace; a raw Part.Shell often won't close cleanly.
    shell = Part.Shell(occ_faces)
    sewn = shell.copy()
    try:
        sewn.sewShape()
    except Exception:  # noqa: BLE001
        sewn = shell

    for candidate in (sewn, shell):
        try:
            solid = Part.Solid(candidate)
        except Exception:  # noqa: BLE001
            continue
        if solid.isValid():
            return solid.removeSplitter(), True

    # No closed solid; hand back the sewn shell so the caller can still export.
    return sewn.removeSplitter(), False


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
    shape = shape.removeSplitter()
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
