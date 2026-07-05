"""Typed STEP tessellation for the viewer: per-face surface provenance.

Run under **FreeCAD's Python** (spawned by :func:`webapp.conversion.
tessellate_typed`, same env wiring as the conversion worker)::

    python -m mesh2step.webapp.stepmesh --step in.step \
        --out-blob types.m2sm --out-meta types.json [--deflection 0.1]

For every Part face of the STEP it records a *category* from the surface
class (``Face.Surface.TypeId``), tessellates the face, and emits one M2SM
blob (see :mod:`.meshdata`) whose per-vertex colours encode the category —
so the browser can show at a glance what was truly converted to analytic
geometry vs left as residual tessellation:

* ``plane``    — analytic planar faces (light gray)
* ``cylinder`` — cylinders + cones (blue)
* ``sphere``   — spheres + tori (teal)
* ``freeform`` — B-spline / Bezier / swept / revolved surfaces (green)
* ``residual`` — planar faces in RTAF smooth chains (orange): tessellation
  strips that arrived as fans of near-tangent flat panels. Chain detection
  reuses the SAME helpers + knobs as ``builder.compute_rtaf`` (imported,
  not copied), so the highlighted area matches the reported RTAF number.

A JSON sidecar carries the legend (colour, face/triangle counts, area) the
UI renders next to the model. This module lives in webapp/ (not the
pipeline): it is a *viewer* concern and must not disturb conversion code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# Category -> (RGB, human label). Order = legend display order.
CATEGORIES: dict[str, tuple[tuple[int, int, int], str]] = {
    "plane": ((203, 208, 215), "Planar (analytic)"),
    "cylinder": ((59, 130, 246), "Cylinder / cone"),
    "sphere": ((20, 184, 166), "Sphere / torus"),
    "freeform": ((34, 197, 94), "B-spline / swept"),
    "residual": ((249, 115, 22), "Residual tessellation"),
}

_TYPE_TO_CAT = {
    "Part::GeomPlane": "plane",
    "Part::GeomCylinder": "cylinder",
    "Part::GeomCone": "cylinder",
    "Part::GeomSphere": "sphere",
    "Part::GeomToroid": "sphere",
    # Everything else (BSpline/Bezier/Extrusion/Revolution/Offset...) => freeform
}


def _classify(face) -> str:
    try:
        return _TYPE_TO_CAT.get(face.Surface.TypeId, "freeform")
    except Exception:  # noqa: BLE001 - unclassifiable face: treat as freeform
        return "freeform"


def _residual_faces(faces, is_plane) -> set:
    """Indices of planar faces in RTAF smooth chains (>= min chain length).

    Same construction as ``builder.compute_rtaf`` — shared-edge adjacency of
    planar faces whose normal step lies in (rtaf_angle_lo, rtaf_angle_hi) —
    using the builder's own edge-key/normal helpers and config defaults.
    """
    from mesh2step.builder import _face_plane_normal, _rtaf_edge_key
    from mesh2step.config import ConversionConfig

    cfg = ConversionConfig()
    n = len(faces)
    if cfg.rtaf_max_faces is not None and n > cfg.rtaf_max_faces:
        return set()  # same cap as compute_rtaf: too big to classify
    normals = [_face_plane_normal(faces[i]) if is_plane[i] else None
               for i in range(n)]
    edge_map: dict = {}
    for fi, f in enumerate(faces):
        for e in f.Edges:
            k = _rtaf_edge_key(e)
            if k is not None:
                edge_map.setdefault(k, set()).add(fi)
    lo, hi = float(cfg.rtaf_angle_lo), float(cfg.rtaf_angle_hi)
    adj: dict[int, set] = {i: set() for i in range(n)}
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
    seen: set = set()
    chain: set = set()
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
        if len(comp) >= cfg.rtaf_min_chain:
            chain.update(comp)
    return chain


def build_typed_mesh(step_path: str, deflection: float) -> tuple[bytes, dict]:
    """Tessellate ``step_path`` per-face and return (M2SM blob, legend meta)."""
    import numpy as np

    import Part  # FreeCAD's Part module (env prepared by the caller)

    from .meshdata import _face_normals, _pack

    shape = Part.Shape()
    shape.read(str(step_path))
    faces = list(shape.Faces)
    is_plane = []
    cats = []
    for f in faces:
        cat = _classify(f)
        is_plane.append(cat == "plane")
        cats.append(cat)

    try:
        residual = _residual_faces(faces, is_plane)
    except Exception:  # noqa: BLE001 - provenance must not break the preview
        residual = set()
    for i in residual:
        cats[i] = "residual"

    tri_chunks: list = []
    color_chunks: list = []
    legend: dict = {c: {"color": list(CATEGORIES[c][0]), "label": CATEGORIES[c][1],
                        "faces": 0, "tris": 0, "area_mm2": 0.0}
                    for c in CATEGORIES}
    for f, cat in zip(faces, cats):
        try:
            verts, tris = f.tessellate(deflection)
        except Exception:  # noqa: BLE001 - skip an untessellatable face
            continue
        if not tris:
            continue
        v = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
        t = np.array(tris, dtype=np.int64)
        tri_pts = v[t]                       # (F,3,3)
        tri_chunks.append(tri_pts)
        rgb = np.array(CATEGORIES[cat][0], dtype=np.uint8)
        color_chunks.append(np.tile(rgb, (tri_pts.shape[0] * 3, 1)))
        entry = legend[cat]
        entry["faces"] += 1
        entry["tris"] += int(tri_pts.shape[0])
        try:
            entry["area_mm2"] += float(f.Area)
        except Exception:  # noqa: BLE001
            pass

    if not tri_chunks:
        raise RuntimeError("no tessellatable faces in STEP")
    tri = np.concatenate(tri_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    fn = _face_normals(tri)
    normals = np.repeat(fn, 3, axis=0)
    blob = _pack(tri.reshape(-1, 3), normals, colors)

    total_area = sum(e["area_mm2"] for e in legend.values()) or 1.0
    for e in legend.values():
        e["area_mm2"] = round(e["area_mm2"], 2)
        e["area_frac"] = round(e["area_mm2"] / total_area, 4)
    meta = {
        "legend": {c: legend[c] for c in CATEGORIES if legend[c]["faces"]},
        "faces": len(faces),
        "tris": int(tri.shape[0]),
        "residual_faces": len(residual),
    }
    return blob, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Typed STEP tessellation (viewer)")
    ap.add_argument("--step", required=True)
    ap.add_argument("--out-blob", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--deflection", type=float, default=0.1)
    args = ap.parse_args(argv)
    try:
        import FreeCAD  # noqa: F401  (import first so `import Part` is safe)
    except ImportError as exc:
        print(f"stepmesh: FreeCAD import failed: {exc}", file=sys.stderr)
        return 3
    try:
        blob, meta = build_typed_mesh(args.step, args.deflection)
    except Exception as exc:  # noqa: BLE001 - report, nonzero exit
        print(f"stepmesh: {exc}", file=sys.stderr)
        return 4
    with open(args.out_blob, "wb") as fh:
        fh.write(blob)
    with open(args.out_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
