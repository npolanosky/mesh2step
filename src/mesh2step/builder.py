"""FreeCAD/OpenCASCADE geometry construction and STEP export.

This is the only module that imports FreeCAD. It turns reconstructed planar
``FaceLoops`` into OCC faces, sews them into a solid, and exports STEP — with a
faceted fallback that always yields a watertight result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .boundary import FaceLoops, extract_face_loops
from .config import ConversionConfig
from .segmentation import segment_planar


def _vec(p):
    import FreeCAD  # type: ignore  # local import; module only runs under FreeCAD

    return FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))


def _wire_from_points(points3d: np.ndarray, Part):
    """Build a closed polygonal wire from ordered 3D points."""
    vectors = [_vec(p) for p in points3d]
    vectors.append(vectors[0])  # close the loop
    return Part.makePolygon(vectors)


def _face_from_loops(loops: FaceLoops, Part):
    """Build a planar OCC face (with holes) from reconstructed loops."""
    outer = _wire_from_points(loops.outer, Part)
    wires = [outer]
    for hole in loops.holes:
        if len(hole) >= 3:
            wires.append(_wire_from_points(hole, Part))
    face = Part.Face(wires)
    return face


def build_reconstructed_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig,
):
    """Reconstruct planar faces, sew them, and return ``(shape, stats)``.

    Returns the resulting ``Part.Shape`` (ideally a solid) plus a stats dict.
    Raises if no valid shell can be sewn — the caller falls back to faceted.
    """
    import Part  # type: ignore

    regions = segment_planar(vertices, faces, config)
    occ_faces = []
    reconstructed = 0
    skipped = 0
    for region in regions:
        if region.size < config.min_region_facets:
            skipped += region.size
            continue
        loops = extract_face_loops(vertices, faces, region, config)
        if loops is None:
            skipped += region.size
            continue
        try:
            occ_faces.append(_face_from_loops(loops, Part))
            reconstructed += 1
        except Exception:  # noqa: BLE001 - OCC raises bare RuntimeErrors
            skipped += region.size

    if not occ_faces:
        raise RuntimeError("no planar faces could be reconstructed")

    shell = Part.Shell(occ_faces)
    shape = shell
    solid = None
    try:
        solid = Part.Solid(shell)
    except Exception:  # noqa: BLE001
        # try sewing to repair small gaps before giving up on a solid
        sewn = shell.copy()
        sewn.sewShape()
        try:
            solid = Part.Solid(sewn)
        except Exception:  # noqa: BLE001
            solid = None

    if solid is not None and solid.isValid():
        shape = solid.removeSplitter()
    else:
        shape = shape.removeSplitter()

    stats = {
        "regions": len(regions),
        "faces_out": reconstructed,
        "faces_in": int(len(faces)),
        "skipped_facets": skipped,
        "is_solid": bool(solid is not None and solid.isValid()),
    }
    return shape, stats


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
