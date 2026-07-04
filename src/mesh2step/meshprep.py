"""FreeCAD-backed mesh health checks, repair and decimation.

Imports FreeCAD's ``Mesh`` module, so it only runs under FreeCAD's Python. Used
by the worker's inspect mode (health read before converting) and by the pipeline
when repair/decimation is requested.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

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
    vertices: np.ndarray, faces: np.ndarray, on_progress=None
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

    Returns ``(vertices, faces, report)`` on success, or ``None`` on failure.
    Failure is ambiguous by return value alone, so (P1-2) the *reason* is logged
    via ``on_progress`` — an environment problem (manifold3d not installed / a
    kernel crash) reads very differently from a geometric rejection (the union
    moved the part, or produced an empty result), and diagnosing which no longer
    requires re-running with instrumentation.
    """
    def log(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    try:
        import manifold3d as m3
    except ImportError:
        log("  self-intersection resolve: SKIPPED — manifold3d not installed "
            "(environment/provisioning issue, not a geometry problem)")
        return None
    try:
        mesh = m3.Mesh(vert_properties=vertices.astype(np.float32),
                       tri_verts=faces.astype(np.uint32))
        mesh.merge()  # weld near-duplicate vertices so construction succeeds
        man = m3.Manifold(mesh)
        if man.is_empty():
            log("  self-intersection resolve: REJECTED — manifold construction "
                "produced an empty solid (mesh not a valid closed volume)")
            return None
        parts = man.decompose()
        if len(parts) > 1:
            man = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
        out = man.to_mesh()
        v2 = np.array(out.vert_properties, dtype=np.float64)[:, :3]
        f2 = np.array(out.tri_verts, dtype=np.int64)
        if len(f2) == 0:
            log("  self-intersection resolve: REJECTED — union produced an empty "
                "mesh (0 faces)")
            return None
        # Sanity: the union must not have moved the part (bbox within 1%).
        ext_in = vertices.max(axis=0) - vertices.min(axis=0)
        ext_out = v2.max(axis=0) - v2.min(axis=0)
        if np.any(np.abs(ext_out - ext_in) > 0.01 * np.maximum(ext_in, 1.0)):
            log(f"  self-intersection resolve: REJECTED — union moved the part's "
                f"bounding box beyond 1% (in {ext_in.tolist()} vs "
                f"out {ext_out.tolist()}); keeping original mesh")
            return None
        report = {
            "bodies": int(len(parts)),
            "faces_in": int(len(faces)),
            "faces_out": int(len(f2)),
        }
        return v2, f2, report
    except Exception as exc:  # noqa: BLE001 - resolution is best-effort
        log(f"  self-intersection resolve: FAILED — manifold3d raised "
            f"{type(exc).__name__}: {exc} (kernel/runtime issue, not a geometry "
            f"rejection)")
        return None


def combine_bodies(
    vertices: np.ndarray, faces: np.ndarray, *, weld: float = 1e-3, on_progress=None
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Union a multi-shell mesh into ONE solid via the manifold3d winding boolean.

    The "combine" multi-body mode: several disjoint (or coincident/near-coincident)
    shells that are really one part are fused into a single watertight surface.
    This is the same winding-number engine ``resolve_self_intersections`` uses, but
    driven unconditionally (the caller has decided the bodies belong together) and
    with a small vertex *weld* first so shells that meet across a sub-tolerance gap
    still merge rather than staying two solids.

    Bit-exact / quantised coincidence is already handled by manifold3d's own
    ``Mesh.merge``; ``weld`` (mm) snaps near-coincident vertices onto a shared grid
    beforehand so a tiny modelled seam (faces that should touch but differ by FP
    noise) collapses to a shared boundary and the union welds through it.

    Unlike ``resolve_self_intersections`` there is deliberately NO bbox-stability
    guard: fusing two shells can legitimately change the combined silhouette (two
    touching cubes become one bar). Returns ``(vertices, faces, report)`` on
    success, or ``None`` (logged with a reason) on any failure, so the caller can
    fall back to the per-body "separate" path.
    """
    def log(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    try:
        import manifold3d as m3
    except ImportError:
        log("  multi-body combine: SKIPPED — manifold3d not installed "
            "(environment/provisioning issue, not a geometry problem)")
        return None

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    # Optional pre-weld: snap near-coincident vertices onto a shared grid so a
    # tiny modelled seam between two shells collapses to a shared boundary the
    # union can fuse through. weld<=0 leaves the mesh untouched (manifold3d's own
    # merge still handles bit-exact coincidence).
    welded = 0
    if weld and weld > 0:
        scale = 1.0 / weld
        keys = np.round(v * scale).astype(np.int64)
        uniq, first_idx, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        if len(first_idx) < len(v):
            welded = int(len(v) - len(first_idx))
            # ``inverse`` maps each ORIGINAL vertex to its merged index; remap the
            # face indices through it, then keep the surviving unique vertices.
            # (np.unique may return inverse with a trailing 1-dim; flatten it.)
            inverse = np.asarray(inverse).reshape(-1)
            v = v[first_idx]
            f = inverse[f].astype(np.int64)
            good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
            f = f[good]

    try:
        mesh = m3.Mesh(vert_properties=v.astype(np.float32),
                       tri_verts=f.astype(np.uint32))
        mesh.merge()
        man = m3.Manifold(mesh)
        if man.is_empty():
            log("  multi-body combine: REJECTED — manifold construction produced "
                "an empty solid (bodies not valid closed volumes)")
            return None
        parts = man.decompose()
        if len(parts) > 1:
            man = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
        out = man.to_mesh()
        v2 = np.array(out.vert_properties, dtype=np.float64)[:, :3]
        f2 = np.array(out.tri_verts, dtype=np.int64)
        if len(f2) == 0:
            log("  multi-body combine: REJECTED — union produced an empty mesh "
                "(0 faces)")
            return None
        report = {
            "bodies_in": int(len(parts)),
            "faces_in": int(len(faces)),
            "faces_out": int(len(f2)),
            "welded_vertices": welded,
        }
        return v2, f2, report
    except Exception as exc:  # noqa: BLE001 - combine is best-effort
        log(f"  multi-body combine: FAILED — manifold3d raised "
            f"{type(exc).__name__}: {exc} (kernel/runtime issue)")
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


# Decimation runs pymeshlab in a SEPARATE subprocess (see meshprep_runner):
# pymeshlab's bundled Qt 5 collides with FreeCAD's Qt 6 in one process on macOS
# (Objective-C duplicate-class SIGTRAP). The runner is launched with only the
# pydeps dir + our package root on PYTHONPATH — never FreeCAD's lib dir — so the
# two Qts never share a process.

_PYMESHLAB_OK: bool | None = None
_DEC_TIMEOUT = 900  # seconds; a single decimation should take a few seconds


def _pydeps_dir() -> str | None:
    """The dir holding the provisioned pymeshlab, inferred without FreeCAD.

    The worker's ``PYTHONPATH`` already contains it (prep_env injects it), so we
    find it as the ``sys.path`` entry that actually holds a ``pymeshlab``
    package. Falls back to provision.pydeps_dir() for the current interpreter.
    """
    for entry in sys.path:
        try:
            if entry and (Path(entry) / "pymeshlab").is_dir():
                return entry
        except OSError:
            continue
    try:
        from . import provision

        target = provision.pydeps_dir(sys.executable)
        if (target / "pymeshlab").is_dir():
            return str(target)
    except Exception:  # noqa: BLE001
        pass
    return None


def _runner_env() -> dict | None:
    """A subprocess env with ONLY pydeps + our package root on PYTHONPATH.

    Crucially excludes FreeCAD's lib dir: importing FreeCAD/Part before pymeshlab
    is what aborts the process (Qt5/Qt6 class collision). Returns None if the
    pydeps dir (hence pymeshlab) can't be located.
    """
    pydeps = _pydeps_dir()
    if pydeps is None:
        return None
    # Directory that makes ``import mesh2step`` resolve (the parent of this pkg).
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([pydeps, pkg_root])
    return env


def _pymeshlab_importable() -> bool:
    """Probe whether pymeshlab imports without crashing, in the clean runner env.

    Runs ``<freecad python> -c "import numpy, pymeshlab"`` with the *runner* env
    (only pydeps + our package root on PYTHONPATH — never FreeCAD's lib dir), the
    same environment the real decimation subprocess uses. A Qt/objc abort
    (uncatchable SIGTRAP, exit 133) or a numpy-ABI crash surfaces as a nonzero
    return code here, and we skip decimation gracefully. Cached per run.
    """
    global _PYMESHLAB_OK
    if _PYMESHLAB_OK is not None:
        return _PYMESHLAB_OK
    env = _runner_env()
    if env is None:
        _PYMESHLAB_OK = False
        return _PYMESHLAB_OK
    try:
        proc = subprocess.run(
            [sys.executable, "-c",
             "import numpy, pymeshlab; print('ok')"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        _PYMESHLAB_OK = proc.returncode == 0 and "ok" in proc.stdout
    except Exception:  # noqa: BLE001
        _PYMESHLAB_OK = False
    return _PYMESHLAB_OK


def decimate_planar(
    verts: np.ndarray, faces: np.ndarray, target_faces: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Planar-preserving quadric decimation via the out-of-process pymeshlab runner.

    Collapses over-tessellated flat regions toward ``target_faces`` while keeping
    holes/curves dense (``planarquadric`` favours coplanar collapses) and edges
    sharp (``preserveboundary``). This both shrinks the output and makes the
    boolean clean-up tractable — and it can even *improve* hole detection by
    cleaning up noisy tessellation.

    pymeshlab is run in a separate subprocess (meshprep_runner) so its bundled
    Qt 5 never collides with FreeCAD's Qt 6 in this (FreeCAD-loaded) process.
    Returns ``(verts, faces, report)``; on any failure (pymeshlab missing,
    subprocess crash) it returns the input unchanged.
    """
    report = {"before_facets": int(len(faces)), "target": int(target_faces)}
    if len(faces) <= target_faces:
        report["after_facets"] = int(len(faces))
        report["skipped"] = "already below target"
        return verts, faces, report
    if not _pymeshlab_importable():
        report["after_facets"] = int(len(faces))
        report["skipped"] = "pymeshlab unavailable (import failed/crashed)"
        return verts, faces, report

    env = _runner_env()
    if env is None:  # defensive: _pymeshlab_importable already gated this
        report["after_facets"] = int(len(faces))
        report["skipped"] = "pymeshlab unavailable (import failed/crashed)"
        return verts, faces, report

    with tempfile.TemporaryDirectory(prefix="m2s_decimate_") as tmp:
        in_path = os.path.join(tmp, "in.npz")
        out_path = os.path.join(tmp, "out.npz")
        params_path = os.path.join(tmp, "params.json")
        np.savez(in_path,
                 vertices=np.asarray(verts, dtype=np.float64),
                 faces=np.asarray(faces, dtype=np.int64))
        with open(params_path, "w", encoding="utf-8") as fh:
            json.dump({"target_faces": int(target_faces)}, fh)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "mesh2step.meshprep_runner",
                 in_path, out_path, params_path],
                capture_output=True, text=True, timeout=_DEC_TIMEOUT, env=env,
            )
        except Exception as exc:  # noqa: BLE001 - decimation is best-effort
            report["error"] = f"runner launch failed: {exc}"
            report["after_facets"] = int(len(faces))
            return verts, faces, report
        if proc.returncode != 0 or not os.path.exists(out_path):
            report["error"] = (proc.stderr or f"runner exited {proc.returncode}").strip()[:500]
            report["after_facets"] = int(len(faces))
            return verts, faces, report
        try:
            with np.load(out_path) as data:
                out_v = np.asarray(data["vertices"], dtype=np.float64)
                out_f = np.asarray(data["faces"], dtype=np.int64)
            stats_path = out_path + ".json"
            if os.path.exists(stats_path):
                with open(stats_path, encoding="utf-8") as fh:
                    report.update(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            report["error"] = f"runner output unreadable: {exc}"
            report["after_facets"] = int(len(faces))
            return verts, faces, report

    if "error" in report:  # runner reported an internal decimation failure
        report["after_facets"] = int(len(faces))
        return verts, faces, report
    report["after_facets"] = int(len(out_f))
    return out_v, out_f, report
