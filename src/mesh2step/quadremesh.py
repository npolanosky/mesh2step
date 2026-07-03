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


def _weld_vertices(qv: np.ndarray, quads: np.ndarray, tol: float):
    """Merge cage vertices closer than ``tol`` (union-find). The remesher emits a
    few coincident-but-distinct vertices that read as seams; welding them recovers
    the shared topology. Returns ``(verts, quads)`` re-indexed."""
    try:
        from scipy.spatial import cKDTree
    except Exception:  # noqa: BLE001
        return qv, quads
    parent = list(range(len(qv)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in cKDTree(qv).query_pairs(tol):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    remap: dict[int, int] = {}
    newv = []
    for i in range(len(qv)):
        r = find(i)
        if r not in remap:
            remap[r] = len(newv)
            newv.append(qv[r])
    newq = np.array([[remap[find(int(x))] for x in q] for q in quads], dtype=int)
    return np.array(newv, dtype=float), newq


def _boundary_loops(quads: np.ndarray) -> list[list[int]]:
    """Ordered vertex loops of the cage's boundary (edges on exactly one face).

    The remesher occasionally leaves small holes; each is a simple closed loop of
    boundary edges. Returns one vertex list per loop (open loops are dropped)."""
    from collections import defaultdict

    from .catmull_clark import quad_edge_faces

    ef = quad_edge_faces(quads)
    adj: dict[int, list[int]] = defaultdict(list)
    for (a, b), fs in ef.items():
        if len(fs) == 1:
            adj[a].append(b)
            adj[b].append(a)
    seen: set[int] = set()
    loops: list[list[int]] = []
    for start in list(adj):
        if start in seen or len(adj[start]) != 2:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        closed = False
        while True:
            nxts = [x for x in adj[cur] if x != prev]
            if not nxts:
                break
            nb = nxts[0]
            if nb == loop[0]:
                closed = True
                break
            if nb in seen:
                break
            loop.append(nb)
            seen.add(nb)
            prev, cur = cur, nb
        if closed and len(loop) >= 3:
            loops.append(loop)
    return loops


def repair_quad_cage(qv: np.ndarray, quads: np.ndarray, *, max_hole: int = 64):
    """Best-effort repair of a remesher cage toward closed-manifold-all-quad.

    Steps: (1) weld near-coincident vertices; (2) drop degenerate quads; (3) drop
    the excess faces on any non-manifold edge; (4) fill each small boundary hole
    with a quad fan around a new centroid vertex (even loops split into quads
    pairwise; odd loops are left, so a cage with an odd hole stays open and is
    declined upstream). The added centroids become extraordinary vertices, which
    the EV-cap machinery already handles. Returns ``(verts, quads)`` — the caller
    re-validates; this never raises."""
    qv = np.asarray(qv, dtype=float)
    quads = np.asarray(quads, dtype=int)
    if quads.ndim != 2 or quads.shape[1] != 4 or len(quads) == 0:
        return qv, quads
    # Median edge length sets the weld tolerance (a small fraction of it).
    el = [float(np.linalg.norm(qv[int(q[k])] - qv[int(q[(k + 1) % 4])]))
          for q in quads for k in range(4)]
    med = float(np.median(el)) if el else 1.0
    qv, quads = _weld_vertices(qv, quads, 0.05 * med)

    # Drop degenerate quads.
    quads = np.array([q for q in quads if len({int(x) for x in q}) == 4], dtype=int)
    if len(quads) == 0:
        return qv, quads

    # Drop excess faces on non-manifold edges (keep the first two per edge).
    from .catmull_clark import quad_edge_faces
    ef = quad_edge_faces(quads)
    bad = set()
    for fs in ef.values():
        if len(fs) > 2:
            bad.update(fs[2:])
    if bad:
        quads = np.array([q for i, q in enumerate(quads) if i not in bad], dtype=int)

    # Fill boundary holes with a centroid fan of quads.
    loops = _boundary_loops(quads)
    if loops:
        newv = list(qv)
        newq = list(quads)
        for loop in loops:
            n = len(loop)
            if n < 4 or n % 2 != 0 or n > max_hole:
                continue  # odd/large holes left open -> caller declines
            centroid = np.mean([qv[i] for i in loop], axis=0)
            ci = len(newv)
            newv.append(centroid)
            for i in range(0, n, 2):
                a, b, c = loop[i], loop[(i + 1) % n], loop[(i + 2) % n]
                newq.append([ci, a, b, c])
        qv = np.array(newv, dtype=float)
        quads = np.array([q for q in newq if len({int(x) for x in q}) == 4], dtype=int)
    return qv, quads


def build_quad_cage(vertices, faces, target_quads, on_progress=None):
    """Remesh + repair + target back-off -> a closed-manifold-all-quad cage.

    The remesher (pynanoinstantmeshes) is more robust at coarser targets: a fine
    target can leave small holes / non-manifold quads while a coarser one on the
    same input is clean. We try the requested target, repair the cage, and if it
    still doesn't validate, back off to progressively coarser targets (repairing
    each). Returns ``(qv, quads, detail)`` with ``detail['ok']`` set; on failure
    ``detail`` explains why so the caller can decline (never regress)."""
    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    target = int(target_quads)
    ladder = [target]
    for frac in (0.5, 0.3, 0.18):
        t = max(40, int(target * frac))
        if t < ladder[-1]:
            ladder.append(t)
    last_detail: dict = {}
    repaired_fallback = None   # (qv, quads, detail) — used only if no clean cage
    for t in ladder:
        try:
            qv, quads = quad_remesh(vertices, faces, t)
        except Exception as exc:  # noqa: BLE001
            last_detail = {"reason": f"remesh failed: {exc}", "target": t}
            continue
        ok, detail = validate_quad_cage(quads)
        if ok:
            # A clean, unrepaired cage is best (no artificial hole-fill EVs) — take
            # it immediately. The remesher is cleaner at coarser targets, so the
            # ladder trends this way anyway.
            detail["target"] = t
            detail["repaired"] = False
            return qv, quads, {**detail, "ok": True}
        # Remember the first repairable cage as a fallback, but keep trying coarser
        # targets for a clean one (repair adds a high-valence EV per filled hole).
        if repaired_fallback is None:
            rv, rq = repair_quad_cage(qv, quads)
            ok2, detail2 = validate_quad_cage(rq)
            if ok2:
                detail2["target"] = t
                detail2["repaired"] = True
                repaired_fallback = (rv, rq, {**detail2, "ok": True})
        last_detail = {**detail, "target": t}
        progress(f"Organic: cage at target {t} not clean "
                 f"({detail.get('reason')}); trying coarser")
    if repaired_fallback is not None:
        rv, rq, rdetail = repaired_fallback
        progress(f"Organic: no clean cage; using repaired cage at target "
                 f"{rdetail.get('target')} ({rdetail.get('quads')} quads)")
        return rv, rq, rdetail
    return None, None, {**last_detail, "ok": False}
