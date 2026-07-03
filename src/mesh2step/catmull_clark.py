"""Catmull-Clark subdivision + exact bicubic B-spline patch extraction (Stam 1998).

Pure numpy; no FreeCAD. This is the geometry core of Candidate A (whole-body
organic reconstruction, see docs/ORGANIC_CONVERSION_RESEARCH.md §5): given an
all-quad control cage (from ``quadremesh``), it

  * subdivides the cage 1-2x (Catmull-Clark) so extraordinary vertices (EVs,
    valence != 4) are isolated — every subdivision step makes new vertices
    valence-4, so after one step almost every face touches at most one EV;
  * least-squares shrink-wraps the cage so its limit surface approximates the
    *original* mesh (the limit surface otherwise shrinks inside the cage — this
    is the step that makes the output dimensionally honest); and
  * classifies each face: a *regular* quad (all four corners valence-4,
    interior) is an EXACT uniform bicubic B-spline patch given by its 4x4
    control-point neighbourhood (Stam) — the builder turns each into one
    ``Part.BSplineSurface`` face; remaining faces touch an EV and are capped.

Only the numpy topology/geometry lives here; patch *emission* (OCC faces, sew,
solid) is the builder's job, keeping the pure-numpy rule the rest of the
pipeline follows.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def quad_edge_faces(quads: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge (min,max vertex) to the quad faces on it."""
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, q in enumerate(quads):
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            edge_faces[(min(a, b), max(a, b))].append(fi)
    return edge_faces


def vertex_valence(verts: np.ndarray, quads: np.ndarray):
    """Return ``(valence[nV], neighbour_sets)`` for an all-quad mesh."""
    nV = len(verts)
    v_nbr: list[set[int]] = [set() for _ in range(nV)]
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            v_nbr[a].add(b)
            v_nbr[b].add(a)
    val = np.array([len(v_nbr[v]) for v in range(nV)], dtype=int)
    return val, v_nbr


def is_closed_manifold(quads: np.ndarray) -> tuple[bool, dict]:
    """Check an all-quad mesh is closed (no boundary edge) and manifold (no edge
    shared by >2 faces). Returns ``(ok, detail)`` for reporting/gating."""
    edge_faces = quad_edge_faces(quads)
    boundary = sum(1 for fs in edge_faces.values() if len(fs) == 1)
    nonmanifold = sum(1 for fs in edge_faces.values() if len(fs) > 2)
    ok = boundary == 0 and nonmanifold == 0
    return ok, {"boundary_edges": boundary, "nonmanifold_edges": nonmanifold,
                "edges": len(edge_faces)}


def catmull_clark_subdivide(verts: np.ndarray, quads: np.ndarray):
    """One Catmull-Clark subdivision step on an all-quad mesh.

    Returns ``(new_verts, new_quads)``. Standard masks (Catmull & Clark 1978):
    face point = mean of the face's 4 vertices; edge point = mean of the 2 edge
    endpoints and the 2 adjacent face points (boundary edge: endpoint midpoint);
    updated vertex = (F + 2R + (n-3)V)/n for interior valence n (F = mean of
    adjacent face points, R = mean of adjacent edge midpoints), boundary vertices
    use the crease rule 3/4 V + 1/8 of each of the two boundary neighbours. Each
    quad splits into four (corner - edge - face - edge)."""
    verts = np.asarray(verts, dtype=float)
    quads = np.asarray(quads, dtype=int)
    nV = len(verts)

    face_pt = verts[quads].mean(axis=1)                      # (nQ,3)
    edge_faces = quad_edge_faces(quads)

    edge_pt: dict[tuple[int, int], np.ndarray] = {}
    v_face_sum = np.zeros((nV, 3))
    v_face_cnt = np.zeros(nV)
    for fi, q in enumerate(quads):
        for vid in q:
            v_face_sum[vid] += face_pt[fi]
            v_face_cnt[vid] += 1

    v_edge_sum = np.zeros((nV, 3))
    v_edge_cnt = np.zeros(nV)
    is_boundary = np.zeros(nV, dtype=bool)
    b_sum = np.zeros((nV, 3))
    b_cnt = np.zeros(nV)
    for (a, b), fs in edge_faces.items():
        mid = 0.5 * (verts[a] + verts[b])
        if len(fs) == 2:
            edge_pt[(a, b)] = (verts[a] + verts[b] + face_pt[fs[0]] + face_pt[fs[1]]) / 4.0
        else:                                               # boundary edge
            edge_pt[(a, b)] = mid
            is_boundary[a] = is_boundary[b] = True
            b_sum[a] += verts[b]
            b_sum[b] += verts[a]
            b_cnt[a] += 1
            b_cnt[b] += 1
        for vid in (a, b):
            v_edge_sum[vid] += mid
            v_edge_cnt[vid] += 1

    new_v = np.zeros((nV, 3))
    for vid in range(nV):
        if is_boundary[vid] and b_cnt[vid] >= 2:
            new_v[vid] = 0.75 * verts[vid] + 0.125 * b_sum[vid]
        else:
            n = v_face_cnt[vid]
            if n < 3:
                new_v[vid] = verts[vid]
                continue
            F = v_face_sum[vid] / n
            R = v_edge_sum[vid] / v_edge_cnt[vid]
            new_v[vid] = (F + 2.0 * R + (n - 3.0) * verts[vid]) / n

    fp_base = nV
    ep_base = fp_base + len(quads)
    edge_list = list(edge_pt.keys())
    edge_index = {e: ep_base + i for i, e in enumerate(edge_list)}
    all_verts = np.vstack([new_v, face_pt,
                           np.array([edge_pt[e] for e in edge_list])])

    new_quads = []
    for fi, q in enumerate(quads):
        fpi = fp_base + fi
        for k in range(4):
            a = int(q[k])
            b = int(q[(k + 1) % 4])
            d = int(q[(k - 1) % 4])
            e_ab = edge_index[(min(a, b), max(a, b))]
            e_da = edge_index[(min(d, a), max(d, a))]
            new_quads.append([a, e_ab, fpi, e_da])
    return all_verts, np.array(new_quads, dtype=int)


def limit_positions(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Catmull-Clark LIMIT positions of the cage vertices (numpy).

    Interior valence-n vertex limit (Halstead et al. 1993):
    ``V_inf = (n V + 4 R + F) / (n + 5)`` where R = mean of adjacent edge
    midpoints, F = mean of adjacent face points — this is the exact limit of the
    C2 Catmull-Clark surface at the vertex. Boundary vertices use the cubic
    B-spline limit (V + 4*mid_nbrs)/6 form: ``(4V + nbr_a + nbr_b)/6``.
    """
    verts = np.asarray(verts, dtype=float)
    quads = np.asarray(quads, dtype=int)
    nV = len(verts)
    face_pt = verts[quads].mean(axis=1)
    edge_faces = quad_edge_faces(quads)

    vf_sum = np.zeros((nV, 3))
    vf_cnt = np.zeros(nV)
    for fi, q in enumerate(quads):
        for vid in q:
            vf_sum[vid] += face_pt[fi]
            vf_cnt[vid] += 1

    ve_sum = np.zeros((nV, 3))
    ve_cnt = np.zeros(nV)
    is_boundary = np.zeros(nV, dtype=bool)
    b_sum = np.zeros((nV, 3))
    b_cnt = np.zeros(nV)
    for (a, b), fs in edge_faces.items():
        mid = 0.5 * (verts[a] + verts[b])
        for vid in (a, b):
            ve_sum[vid] += mid
            ve_cnt[vid] += 1
        if len(fs) == 1:
            is_boundary[a] = is_boundary[b] = True
            b_sum[a] += verts[b]
            b_sum[b] += verts[a]
            b_cnt[a] += 1
            b_cnt[b] += 1

    out = np.zeros((nV, 3))
    for vid in range(nV):
        if is_boundary[vid] and b_cnt[vid] >= 2:
            out[vid] = (4.0 * verts[vid] + b_sum[vid]) / 6.0
        else:
            n = vf_cnt[vid]
            if n < 3:
                out[vid] = verts[vid]
                continue
            R = ve_sum[vid] / ve_cnt[vid]
            F = vf_sum[vid] / n
            out[vid] = (n * verts[vid] + 4.0 * R + F) / (n + 5.0)
    return out


def _build_kdtree(points: np.ndarray):
    try:
        from scipy.spatial import cKDTree

        return cKDTree(points)
    except Exception:  # noqa: BLE001 - scipy always present under FreeCAD, but be safe
        return None


def fit_cage_to_mesh(
    verts: np.ndarray,
    quads: np.ndarray,
    target_points: np.ndarray,
    iterations: int = 2,
    step: float = 1.0,
):
    """Shrink-wrap the control cage so its Catmull-Clark limit surface approximates
    ``target_points`` (the original mesh vertices).

    The limit map is linear in the cage, but a full sparse solve is heavy; a
    cheaper, robust Gauss-Seidel-style projection works well for a manufacturing
    target: each iteration measures the limit position of every cage vertex,
    finds the nearest original-mesh point, and moves the cage vertex by the
    residual (limit->target) so the *limit* — not the cage — lands on the mesh.
    Since ``V_inf`` is an affine, mass-preserving average of the cage, nudging
    the cage by the limit residual reduces the limit deviation monotonically in
    practice. Returns the fitted cage vertices (quads unchanged)."""
    verts = np.asarray(verts, dtype=float).copy()
    tree = _build_kdtree(np.asarray(target_points, dtype=float))
    if tree is None:
        return verts
    for _ in range(max(1, iterations)):
        limit = limit_positions(verts, quads)
        _, idx = tree.query(limit)
        residual = np.asarray(target_points)[idx] - limit
        verts = verts + step * residual
    return verts


def _quad_neighbor_across(fi, a, b, edge_faces):
    for f in edge_faces.get((min(a, b), max(a, b)), []):
        if f != fi:
            return f
    return None


def regular_patch_grid(fi, quads, edge_faces, v_quads, val):
    """4x4 control-point index grid for a regular quad (Stam), or ``None``.

    A quad is regular when all four corners are interior valence-4 vertices; its
    exact uniform bicubic B-spline patch is defined by the 4x4 lattice formed by
    the quad and its one ring of surrounding quads. The central quad occupies the
    (1,1)-(2,2) block; the returned array holds vertex indices into the cage."""
    q = [int(x) for x in quads[fi]]
    if any(val[v] != 4 for v in q):
        return None
    v0, v1, v2, v3 = q
    grid: list[list[int | None]] = [[None] * 4 for _ in range(4)]
    grid[1][1], grid[1][2], grid[2][2], grid[2][1] = v0, v1, v2, v3

    def fill_edge(a, b, gc, gd):
        nf = _quad_neighbor_across(fi, a, b, edge_faces)
        if nf is None:
            return False
        qv = [int(x) for x in quads[nf]]
        i = qv.index(a)
        qvr = qv[i:] + qv[:i]
        if qvr[1] == b:
            far_a, far_b = qvr[3], qvr[2]
        elif qvr[3] == b:
            far_a, far_b = qvr[1], qvr[2]
        else:
            return False
        grid[gc[0]][gc[1]] = far_a
        grid[gd[0]][gd[1]] = far_b
        return True

    ok = True
    ok &= fill_edge(v0, v1, (0, 1), (0, 2))   # top edge
    ok &= fill_edge(v1, v2, (1, 3), (2, 3))   # right edge
    ok &= fill_edge(v2, v3, (3, 2), (3, 1))   # bottom edge
    ok &= fill_edge(v3, v0, (2, 0), (1, 0))   # left edge
    if not ok:
        return None

    def corner_diag(vc, gi_, gj_, adj1, adj2):
        a1 = grid[adj1[0]][adj1[1]]
        a2 = grid[adj2[0]][adj2[1]]
        if a1 is None or a2 is None:
            return False
        for f in v_quads[vc]:
            qs = {int(x) for x in quads[f]}
            if {vc, a1, a2} <= qs:
                fourth = qs - {vc, a1, a2}
                if len(fourth) == 1:
                    grid[gi_][gj_] = fourth.pop()
                    return True
        return False

    ok &= corner_diag(v0, 0, 0, (0, 1), (1, 0))
    ok &= corner_diag(v1, 0, 3, (0, 2), (1, 3))
    ok &= corner_diag(v2, 3, 3, (2, 3), (3, 2))
    ok &= corner_diag(v3, 3, 0, (3, 1), (2, 0))
    if not ok or any(grid[i][j] is None for i in range(4) for j in range(4)):
        return None
    return np.array([[int(grid[i][j]) for j in range(4)] for i in range(4)])


def build_vertex_quads(verts: np.ndarray, quads: np.ndarray) -> list[list[int]]:
    """Per-vertex list of incident quad face ids."""
    v_quads: list[list[int]] = [[] for _ in range(len(verts))]
    for fi, q in enumerate(quads):
        for vid in q:
            v_quads[vid].append(fi)
    return v_quads


def _bspline_basis(t: float) -> np.ndarray:
    """Uniform cubic B-spline basis at parameter ``t`` in [0,1]."""
    return np.array([
        (1 - t) ** 3,
        3 * t ** 3 - 6 * t ** 2 + 4,
        -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1,
        t ** 3,
    ]) / 6.0


def bspline_patch_value(grid_pts: np.ndarray, u: float, v: float) -> np.ndarray:
    """Evaluate the uniform bicubic B-spline patch of a 4x4 control grid at
    ``(u,v)`` in [0,1] (the central span — the limit surface over the quad)."""
    bu = _bspline_basis(u)
    bv = _bspline_basis(v)
    return np.einsum("i,j,ijk->k", bu, bv, grid_pts)


def classify_patches(verts, quads):
    """Return ``(regular, ev)``: regular is a list of ``(fi, grid_idx[4,4])`` for
    exact bicubic patches; ev is the list of remaining (EV/boundary) face ids."""
    edge_faces = quad_edge_faces(quads)
    val, _ = vertex_valence(verts, quads)
    v_quads = build_vertex_quads(verts, quads)
    regular = []
    ev = []
    for fi in range(len(quads)):
        gi = regular_patch_grid(fi, quads, edge_faces, v_quads, val)
        if gi is None:
            ev.append(fi)
        else:
            regular.append((fi, gi))
    return regular, ev
