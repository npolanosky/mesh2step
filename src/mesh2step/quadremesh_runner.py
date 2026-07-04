"""Out-of-process quad-remesh runner (native pynanoinstantmeshes, isolated).

The native remesher (``pynanoinstantmeshes``, a compiled C-extension) can hang
or run away on memory for some meshes — measured on a 894-facet prepared
gridfinity_bin_1x1x3, where a single ``remesh`` call climbed past 20 GB of RSS
and never returned (SIGKILL). A wall-clock check in Python cannot interrupt a
C-extension call, so the only robust guard is process isolation: run the remesh
in a *separate* process with a hard timeout and an address-space (RLIMIT_AS)
memory ceiling, so a hang is killed by the parent's ``timeout`` and a blow-up is
killed by the OS — either way the parent survives and the organic pass declines.

This module is that process. It is invoked as::

    <python> -m mesh2step.quadremesh_runner <in.npz> <out.npz> <params.json>

It imports ONLY numpy + pynanoinstantmeshes (never FreeCAD, never the rest of
mesh2step's FreeCAD-touching modules), reads ``(vertices, faces)`` + params from
the input files, runs one ``quad_remesh`` equivalent, and writes ``(qv, quads)``
back out. On the child side it sets RLIMIT_AS to the requested memory ceiling so
a runaway allocation aborts the child cleanly instead of the machine.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def _apply_memory_limit(max_mb: int | None) -> None:
    """Cap the child's address space (RLIMIT_AS) so a runaway remesh allocation
    is killed by the OS rather than exhausting the machine. Best-effort: a
    platform without ``resource`` or that refuses the limit just runs uncapped."""
    if not max_mb or max_mb <= 0:
        return
    try:
        import resource

        nbytes = int(max_mb) * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Never raise an already-lower hard limit; only tighten.
        new_hard = nbytes if hard in (resource.RLIM_INFINITY, -1) else min(nbytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (min(nbytes, new_hard), new_hard))
    except Exception:  # noqa: BLE001 - the timeout is the primary guard anyway
        pass


def remesh(verts: np.ndarray, faces: np.ndarray, params: dict):
    """Run the native quad remesh in this (isolated) process.

    Mirrors :func:`mesh2step.quadremesh.quad_remesh` exactly so the isolated call
    is byte-for-byte equivalent to the in-process one. Returns ``(qv, quads)``."""
    import pynanoinstantmeshes as pnim

    v = np.ascontiguousarray(verts, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    out = pnim.remesh(
        v, f, int(params["target_quads"]), posy=4, rosy=4,
        deterministic=bool(params.get("deterministic", True)),
        smooth_iter=int(params.get("smooth_iter", 2)),
    )
    qv = np.asarray(out[0], dtype=np.float64)
    quads = np.asarray(out[1], dtype=np.int64)
    return qv, quads


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: quadremesh_runner <in.npz> <out.npz> <params.json>",
              file=sys.stderr)
        return 2
    in_path, out_path, params_path = argv[1], argv[2], argv[3]
    with open(params_path, encoding="utf-8") as fh:
        params = json.load(fh)
    _apply_memory_limit(params.get("max_memory_mb"))
    with np.load(in_path) as data:
        verts = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    qv, quads = remesh(verts, faces, params)
    np.savez(out_path, qv=qv, quads=quads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
