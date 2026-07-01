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
