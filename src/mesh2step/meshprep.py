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

    if config.decimate:
        reduction = min(max(float(config.decimate), 0.0), 0.99)
        try:
            m.decimate(float(config.decimate_tol), reduction)
            report["actions"].append(f"decimate({reduction:g})")
        except Exception:  # noqa: BLE001
            pass

    report["after_facets"] = int(m.CountFacets)
    verts, faces = _to_numpy(m)
    return verts, faces, report
