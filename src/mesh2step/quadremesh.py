"""Quad remeshing via pynanoinstantmeshes (Candidate A step 2).

Wraps the BSD-licensed ``pynanoinstantmeshes`` (a nano re-implementation of
Instant Meshes, Jakob et al. SIGGRAPH Asia 2015) to turn a triangle mesh into a
coarse ALL-QUAD control cage for the Catmull-Clark pipeline
(docs/ORGANIC_CONVERSION_RESEARCH.md §4). Pure numpy in/out; the dependency is
optional and imported lazily, so a runtime without it degrades gracefully (the
organic-multipatch tier simply declines and the pipeline falls back).

Validation is load-bearing: Candidate A needs a *closed, manifold, all-quad*
cage. The remesher can emit boundary edges on some inputs (an open quad mesh);
those cages are rejected here so the builder never tries to close an un-closeable
shell (never regress).
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np


@contextlib.contextmanager
def _suppress_c_stdout():
    """Silence C-extension output written straight to the OS stdout fd.

    ``pynanoinstantmeshes`` is a compiled extension that logs progress to file
    descriptor 1 directly, so a Python-level ``redirect_stdout`` does not catch
    it. We dup2 /dev/null over fd 1 for the duration, then restore. Falls back to
    a no-op if the platform can't dup (never blocks the remesh)."""
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    saved = None
    devnull = None
    try:
        saved = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        yield
    except Exception:  # noqa: BLE001 - if dup isn't available, just run uncaptured
        yield
    finally:
        try:
            if saved is not None:
                os.dup2(saved, 1)
                os.close(saved)
            if devnull is not None:
                os.close(devnull)
        except Exception:  # noqa: BLE001
            pass


def available() -> bool:
    """True if pynanoinstantmeshes can be imported in this interpreter."""
    try:
        import pynanoinstantmeshes  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def quad_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_quads: int,
    *,
    deterministic: bool = True,
    smooth_iter: int = 2,
):
    """Remesh a triangle mesh into an all-quad cage of ~``target_quads`` faces.

    Returns ``(quad_verts (N,3) float64, quads (M,4) int64)``. The remesher's
    ``vertex_count`` argument is an edge-length target rather than a hard cap, so
    the actual face count can differ from ``target_quads`` (it subdivides a mesh
    that is too coarse for the requested edge length). ``posy=4`` forces pure-quad
    extraction. ``deterministic`` makes the field optimisation reproducible so
    byte-stability holds across runs. Raises if the dependency is unavailable."""
    import pynanoinstantmeshes as pnim

    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    # The remesher is chatty on the OS stdout fd; swallow it (progress goes via
    # the caller's on_progress).
    with _suppress_c_stdout():
        out = pnim.remesh(
            v, f, int(target_quads), posy=4, rosy=4,
            deterministic=bool(deterministic), smooth_iter=int(smooth_iter),
        )
    qv = np.asarray(out[0], dtype=np.float64)
    quads = np.asarray(out[1], dtype=np.int64)
    return qv, quads


def validate_quad_cage(quads: np.ndarray) -> tuple[bool, dict]:
    """Gate a quad cage for the Catmull-Clark pipeline.

    Requires: all faces are quads (4 corners), the mesh is CLOSED (no boundary
    edge) and MANIFOLD (no edge shared by >2 faces), and there is at least a
    minimal face count. Returns ``(ok, detail)``; ``detail`` carries the counts
    for the stats/report so a decline is explainable."""
    from .catmull_clark import is_closed_manifold

    detail: dict = {"quads": int(len(quads))}
    if quads.ndim != 2 or quads.shape[1] != 4:
        detail["reason"] = "not all-quad"
        return False, detail
    if len(quads) < 8:
        detail["reason"] = "too few quads"
        return False, detail
    # Degenerate quads (repeated corner index).
    degenerate = int(sum(len(set(int(x) for x in q)) < 4 for q in quads))
    detail["degenerate_quads"] = degenerate
    if degenerate > 0:
        detail["reason"] = "degenerate quads"
        return False, detail
    ok, cm = is_closed_manifold(quads)
    detail.update(cm)
    if not ok:
        detail["reason"] = "not closed-manifold"
    return ok, detail
