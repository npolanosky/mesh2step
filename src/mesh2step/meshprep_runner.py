"""Out-of-process pymeshlab decimation runner (never imports FreeCAD).

pymeshlab bundles its own **Qt 5.15**; FreeCAD 1.1 loads **Qt 6.8**. Importing
both into one process on macOS triggers an Objective-C duplicate-class abort
(SIGTRAP, exit 133) — see meshprep. So decimation must run in a *separate*
process that has pymeshlab but never touches FreeCAD.

This module is that process. It is invoked as::

    <freecad-python> -m mesh2step.meshprep_runner <in.npz> <out.npz> <params.json>

with **only the pydeps dir on PYTHONPATH** (NOT FreeCAD's lib dir), so the Qt
collision can never happen. It reads welded ``(vertices, faces)`` from the input
``.npz``, runs the planar-preserving quadric edge-collapse, and writes the
decimated ``(vertices, faces)`` plus a JSON stats blob back out.

It deliberately imports nothing from mesh2step that pulls in FreeCAD (only numpy
+ pymeshlab), so ``python -m mesh2step.meshprep_runner`` stays FreeCAD-free even
though it lives inside the package.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def decimate(
    verts: np.ndarray, faces: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Planar-preserving quadric decimation via pymeshlab (in-process here).

    Mirrors the filter chain the pipeline relied on when pymeshlab ran in-process
    (cleanup → quadric edge-collapse with planarquadric → non-manifold repair).
    Returns ``(verts, faces, stats)``.
    """
    import pymeshlab as ml

    target_faces = int(params["target_faces"])
    stats = {"before_facets": int(len(faces)), "target": target_faces}

    ms = ml.MeshSet()
    ms.add_mesh(
        ml.Mesh(
            vertex_matrix=np.asarray(verts, dtype=np.float64),
            face_matrix=np.asarray(faces, dtype=np.int32),
        )
    )
    # Cleanup that pymeshlab's file loader would do implicitly but add_mesh
    # skips — without it the decimated mesh can be left non-watertight, which
    # breaks the boolean base solid.
    for filt in (
        "meshing_remove_duplicate_vertices",
        "meshing_remove_duplicate_faces",
        "meshing_remove_unreferenced_vertices",
        "meshing_remove_null_faces",
    ):
        try:
            ms.apply_filter(filt)
        except Exception:  # noqa: BLE001 - filter names vary by version
            pass
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=float(params.get("qualitythr", 0.4)),
        preserveboundary=True,
        boundaryweight=float(params.get("boundaryweight", 2.0)),
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
        autoclean=True,
    )
    for filt in (
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
        "meshing_re_orient_faces_coherently",
    ):
        try:
            ms.apply_filter(filt)
        except Exception:  # noqa: BLE001
            pass
    m = ms.current_mesh()
    out_v = np.asarray(m.vertex_matrix(), dtype=np.float64)
    out_f = np.asarray(m.face_matrix(), dtype=np.int64)
    stats["after_facets"] = int(len(out_f))
    return out_v, out_f, stats


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: meshprep_runner <in.npz> <out.npz> <params.json>",
              file=sys.stderr)
        return 2
    in_path, out_path, params_path = args

    with np.load(in_path) as data:
        verts = data["vertices"]
        faces = data["faces"]
    with open(params_path, encoding="utf-8") as fh:
        params = json.load(fh)

    try:
        out_v, out_f, stats = decimate(verts, faces, params)
    except Exception as exc:  # noqa: BLE001 - report as stats, exit nonzero
        np.savez(out_path, vertices=verts, faces=faces)
        stats = {"before_facets": int(len(faces)),
                 "after_facets": int(len(faces)), "error": str(exc)}
        with open(out_path + ".json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh)
        return 1

    np.savez(out_path, vertices=out_v.astype(np.float64),
             faces=out_f.astype(np.int64))
    with open(out_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
