"""FreeCAD-backed mesh health checks, repair and decimation.

Imports FreeCAD's ``Mesh`` module, so it only runs under FreeCAD's Python. Used
by the worker's inspect mode (health read before converting) and by the pipeline
when repair/decimation is requested.
"""

from __future__ import annotations

import numpy as np

from .config import ConversionConfig


def _to_numpy(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return welded (vertices, faces) from a FreeCAD Mesh via its Topology."""
    points, facets = mesh.Topology
    verts = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    faces = (
        np.array(facets, dtype=np.int64) if facets else np.zeros((0, 3), dtype=np.int64)
    )
    return verts, faces


def mesh_health(path: str) -> dict:
    """Cheap input-quality read: counts + manifold/self-intersection flags."""
    import FreeCAD  # type: ignore  # noqa: F401  (must precede `import Mesh`)
    import Mesh  # type: ignore

    m = Mesh.Mesh(str(path))
    info = {
        "facets": int(m.CountFacets),
        "points": int(m.CountPoints),
        "non_manifold": bool(m.hasNonManifolds()),
        "self_intersections": bool(m.hasSelfIntersections()),
    }
    # Watertightness where the kernel exposes it (method name varies by version).
    for attr in ("isSolid",):
        fn = getattr(m, attr, None)
        if callable(fn):
            try:
                info["watertight"] = bool(fn())
            except Exception:  # noqa: BLE001
                pass
    return info


def has_self_intersections(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """True if the mesh's facets geometrically intersect each other."""
    import FreeCAD  # type: ignore  # noqa: F401  (must precede `import Mesh`)
    import Mesh  # type: ignore

    m = Mesh.Mesh()
    m.addFacets([(tuple(vertices[a]), tuple(vertices[b]), tuple(vertices[c]))
                 for a, b, c in faces])
    try:
        return bool(m.hasSelfIntersections())
    except Exception:  # noqa: BLE001 - if the check itself fails, assume clean
        return False


def resolve_self_intersections(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Rebuild a self-intersecting / overlapping-body mesh as one clean solid.

    Meshes exported without a final boolean union — a clip modelled through the
    panel it snaps onto, mounting tabs interpenetrating a base — are closed but
    geometrically self-intersecting, which OpenCASCADE rejects outright
    ("Self-intersecting wire"), and no surface-level repair can fix: deleting
    the offending facets and re-closing the hole just re-creates the overlap.

    manifold3d (the winding-number boolean engine OpenSCAD uses) solves exactly
    this: construct a Manifold from the mesh, split it into its disjoint bodies
    (``decompose``), and boolean-union them back together. The union re-computes
    the true outer surface, eliminating the interpenetrations while leaving all
    other geometry bit-identical — no voxelising, no rounded edges.

    Returns ``(vertices, faces, report)`` on success, or None if manifold3d is
    unavailable or the result looks degenerate (callers keep the original mesh).
    """
    try:
        import manifold3d as m3
    except ImportError:
        return None
    try:
        mesh = m3.Mesh(vert_properties=vertices.astype(np.float32),
                       tri_verts=faces.astype(np.uint32))
        mesh.merge()  # weld near-duplicate vertices so construction succeeds
        man = m3.Manifold(mesh)
        if man.is_empty():
            return None
        parts = man.decompose()
        if len(parts) > 1:
            man = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
        out = man.to_mesh()
        v2 = np.array(out.vert_properties, dtype=np.float64)[:, :3]
        f2 = np.array(out.tri_verts, dtype=np.int64)
        if len(f2) == 0:
            return None
        # Sanity: the union must not have moved the part (bbox within 1%).
        ext_in = vertices.max(axis=0) - vertices.min(axis=0)
        ext_out = v2.max(axis=0) - v2.min(axis=0)
        if np.any(np.abs(ext_out - ext_in) > 0.01 * np.maximum(ext_in, 1.0)):
            return None
        report = {
            "bodies": int(len(parts)),
            "faces_in": int(len(faces)),
            "faces_out": int(len(f2)),
        }
        return v2, f2, report
    except Exception:  # noqa: BLE001 - resolution is best-effort
        return None


def problem_points(path: str, cap: int = 4000) -> list[list[float]]:
    """3D points marking mesh defects, for highlighting in the preview.

    Currently the self-intersection segments (the defect that most often blocks
    a watertight solid). Each self-intersection yields its two segment
    endpoints. Returns up to ``cap`` points; empty if the mesh is clean or the
    kernel can't report them.
    """
    import FreeCAD  # type: ignore  # noqa: F401  (must precede `import Mesh`)
    import Mesh  # type: ignore

    m = Mesh.Mesh(str(path))
    pts: list[list[float]] = []
    try:
        if m.hasSelfIntersections():
            for entry in m.getSelfIntersections():
                # entry = (facetA, facetB, Vector p1, Vector p2)
                for v in entry[2:4]:
                    pts.append([float(v.x), float(v.y), float(v.z)])
                    if len(pts) >= cap:
                        return pts
    except Exception:  # noqa: BLE001 - best-effort; never break inspect over this
        pass
    return pts


def load_and_prepare(
    path: str, config: ConversionConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load via FreeCAD Mesh, optionally repair/decimate, return numpy + report."""
    import FreeCAD  # type: ignore  # noqa: F401  (must precede `import Mesh`)
    import Mesh  # type: ignore

    m = Mesh.Mesh(str(path))
    report: dict = {
        "before_facets": int(m.CountFacets),
        "non_manifold_before": bool(m.hasNonManifolds()),
        "actions": [],
    }

    if config.repair_mesh:
        for op in ("removeDuplicatedPoints", "removeDuplicatedFacets",
                   "harmonizeNormals", "removeNonManifolds", "fixIndices",
                   "fixSelfIntersections", "removeFoldsOnSurface", "fixCaps"):
            fn = getattr(m, op, None)
            if fn is None:
                continue
            try:
                fn()
                report["actions"].append(op)
            except Exception:  # noqa: BLE001 - repairs are best-effort
                pass
        try:
            m.fixDegenerations(1e-6)
            report["actions"].append("fixDegenerations")
        except Exception:  # noqa: BLE001
            pass

    report["after_facets"] = int(m.CountFacets)
    verts, faces = _to_numpy(m)
    return verts, faces, report


def decimate_planar(
    verts: np.ndarray, faces: np.ndarray, target_faces: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Planar-preserving quadric decimation via pymeshlab.

    Collapses over-tessellated flat regions toward ``target_faces`` while keeping
    holes/curves dense (``planarquadric`` favours coplanar collapses) and edges
    sharp (``preserveboundary``). This both shrinks the output and makes the
    boolean clean-up tractable — and it can even *improve* hole detection by
    cleaning up noisy tessellation. Returns ``(verts, faces, report)``; on any
    failure (e.g. pymeshlab missing) it returns the input unchanged.
    """
    report = {"before_facets": int(len(faces)), "target": int(target_faces)}
    if len(faces) <= target_faces:
        report["after_facets"] = int(len(faces))
        report["skipped"] = "already below target"
        return verts, faces, report
    try:
        import pymeshlab as ml

        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(verts, dtype=np.float64),
                            face_matrix=np.asarray(faces, dtype=np.int32)))
        # Cleanup that pymeshlab's file loader would do implicitly but add_mesh
        # skips — without it the decimated mesh can be left non-watertight,
        # which breaks the boolean base solid.
        for filt in ("meshing_remove_duplicate_vertices",
                     "meshing_remove_duplicate_faces",
                     "meshing_remove_unreferenced_vertices",
                     "meshing_remove_null_faces"):
            try:
                ms.apply_filter(filt)
            except Exception:  # noqa: BLE001 - filter names vary by version
                pass
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            qualitythr=0.4,
            preserveboundary=True,
            boundaryweight=2.0,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
            autoclean=True,
        )
        for filt in ("meshing_repair_non_manifold_edges",
                     "meshing_repair_non_manifold_vertices",
                     "meshing_re_orient_faces_coherently"):
            try:
                ms.apply_filter(filt)
            except Exception:  # noqa: BLE001
                pass
        m = ms.current_mesh()
        out_v = np.asarray(m.vertex_matrix(), dtype=np.float64)
        out_f = np.asarray(m.face_matrix(), dtype=np.int64)
        report["after_facets"] = int(len(out_f))
        return out_v, out_f, report
    except Exception as exc:  # noqa: BLE001 - decimation is best-effort
        report["error"] = str(exc)
        report["after_facets"] = int(len(faces))
        return verts, faces, report
