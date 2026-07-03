"""Candidate A assembly: quad-patch network -> B-spline shell -> solid.

This is the only FreeCAD-touching part of the whole-body organic pipeline
(docs/ORGANIC_CONVERSION_RESEARCH.md, Candidate A). The pure-numpy stages live
in ``quadremesh`` (remesh + cage validation) and ``catmull_clark`` (subdivision,
limit-fit, patch classification); here we

  1. run those stages to get a fitted, subdivided all-quad cage;
  2. emit one exact uniform bicubic ``Part.BSplineSurface`` face per regular quad
     (Stam) — adjacent regular patches share control rows so their boundary
     curves are mathematically identical;
  3. cap each extraordinary-vertex (EV) face with a filled patch built from its
     four neighbour B-spline boundary edges (so it shares them exactly); and
  4. sew into a shell and, when it closes, a solid.

Every result is *gated*: the caller adopts it only when it is a watertight solid
that re-reads valid and lowers RTAF — otherwise the pipeline keeps its existing
output (never regress). Because every face is a B-spline (not a planar strip),
a successful result carries RTAF ~ 0.

The B-spline face construction is deterministic; the remesh is made
deterministic upstream, so the whole path is reproducible.
"""

from __future__ import annotations

import numpy as np

from .catmull_clark import (
    catmull_clark_subdivide,
    classify_patches,
    fit_cage_to_mesh,
    limit_positions,
    quad_edge_faces,
)
from .config import ConversionConfig
from .quadremesh import quad_remesh, validate_quad_cage


def _bspline_patch_face(grid_pts, Part, FreeCAD):
    """One exact uniform bicubic B-spline face from a 4x4 control grid.

    A uniform knot vector clamps the parameter domain to the central span, so
    ``toShape()`` returns exactly the Face over the central quad (the limit
    surface there). Adjacent regular patches sharing control rows produce
    identical boundary curves -> the shell connects without gaps."""
    poles = [[FreeCAD.Vector(*grid_pts[i][j]) for j in range(4)] for i in range(4)]
    mults = [1, 1, 1, 1, 1, 1, 1, 1]
    knots = [0, 1, 2, 3, 4, 5, 6, 7]
    bs = Part.BSplineSurface()
    bs.buildFromPolesMultsKnots(poles, mults, mults, knots, knots, False, False, 3, 3)
    return bs.toShape()


def should_attempt(reconstructed_stats: dict, faces, config: ConversionConfig) -> bool:
    """Route to Candidate A only for a mostly-organic whole body.

    The gate mirrors the research doc's whole-part shortcut: attempt the organic
    multipatch tier when the after-analytic reconstruction is dominated by
    residual tessellation (RTAF at or above ``organic_multipatch_min_residual``)
    — i.e. the analytic + swept + sphere + freeform tiers left most of the
    surface faceted, so the body is genuinely organic and wraps past any single
    projection. A prismatic-with-features part (low RTAF) keeps its clean
    analytic faces and is never routed here. Bounded by a face-count guard and
    the availability of the optional remesher."""
    from .quadremesh import available

    if not config.organic_multipatch:
        return False
    if not available():
        return False
    cap = config.organic_multipatch_max_faces
    if cap is not None and len(faces) > cap:
        return False
    rtaf = reconstructed_stats.get("rtaf")
    if not isinstance(rtaf, (int, float)):
        # RTAF wasn't computed (e.g. huge shell); fall back on the skipped-facet
        # fraction as a coarse proxy for "mostly organic".
        skipped = reconstructed_stats.get("skipped_facets", 0)
        return skipped >= 0.5 * len(faces)
    return rtaf >= config.organic_multipatch_min_residual


def build_organic_shell(vertices, faces, config: ConversionConfig, on_progress=None):
    """Run the Candidate A geometry pipeline and return ``(shape, stats)``.

    ``shape`` is a watertight ``Part.Solid`` on success, else the best (open)
    shell so the caller can inspect/decline. ``stats`` always reports the cage,
    patch, EV, deviation and watertight figures. Raises only on hard setup
    failures (dependency missing) — the caller wraps this and falls back."""
    import FreeCAD  # type: ignore
    import Part  # type: ignore

    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    stats: dict = {"organic_attempted": True}
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0

    # 1. Quad remesh -> coarse all-quad cage; validate closed-manifold-all-quad.
    target = int(config.organic_multipatch_target_quads)
    progress(f"Organic: quad-remeshing to ~{target} quads")
    qv, quads = quad_remesh(vertices, faces, target)
    ok, cage_detail = validate_quad_cage(quads)
    stats["organic_cage"] = cage_detail
    progress(f"Organic: cage {cage_detail.get('quads')} quads, "
             f"closed-manifold={ok}"
             + ("" if ok else f" ({cage_detail.get('reason')})"))
    if not ok:
        stats["organic_reason"] = f"cage not usable: {cage_detail.get('reason')}"
        return None, stats

    # 3. Shrink-wrap the cage so its Catmull-Clark limit approximates the mesh.
    if config.organic_multipatch_fit_iters > 0:
        progress(f"Organic: fitting cage ({config.organic_multipatch_fit_iters} iters)")
        qv = fit_cage_to_mesh(qv, quads, vertices,
                              iterations=config.organic_multipatch_fit_iters)

    # 4. Subdivide 1-2x to isolate extraordinary vertices.
    for _ in range(max(0, int(config.organic_multipatch_subdiv))):
        qv, quads = catmull_clark_subdivide(qv, quads)
    progress(f"Organic: cage subdivided to {len(quads)} quads")

    # 5. Classify + extract patches. Use the LIMIT positions of the cage as the
    # control net so the emitted surface is the true limit (dimensionally honest).
    limit = limit_positions(qv, quads)
    regular, ev = classify_patches(qv, quads)
    stats["organic_patches"] = len(regular)
    stats["organic_ev_faces"] = len(ev)
    progress(f"Organic: {len(regular)} regular bicubic patches, {len(ev)} EV faces")
    if not regular:
        stats["organic_reason"] = "no regular patches"
        return None, stats

    reg_faces = []
    for _fi, gi in regular:
        try:
            reg_faces.append(_bspline_patch_face(limit[gi], Part, FreeCAD))
        except Exception:  # noqa: BLE001 - a bad grid must not abort the shell
            continue
    if not reg_faces:
        stats["organic_reason"] = "no B-spline faces built"
        return None, stats

    # 6. Cap EV regions. A single extraordinary vertex is surrounded by a ring of
    # faces that all touch it, so EV faces cluster: they can't be capped one by
    # one from regular neighbours. Instead group EV faces into connected regions
    # and cap each region's OUTER boundary loop (the edges it shares with regular
    # patches) with one n-sided filled patch that reuses those exact B-spline
    # boundary curves (refine-and-cap; research doc §5).
    progress("Organic: capping extraordinary-vertex regions")
    edge_faces = quad_edge_faces(quads)
    edge_curve = _index_patch_boundary_edges(regular, limit, Part, FreeCAD)
    ev_regions = _ev_regions(ev, quads, edge_faces)
    stats["organic_ev_regions"] = len(ev_regions)
    cap_faces = []
    capped = 0
    for region in ev_regions:
        loop_edges = _region_boundary_edges(region, quads, edge_faces, edge_curve)
        if loop_edges is None:
            continue
        try:
            cap_faces.append(Part.makeFilledFace(loop_edges))
            capped += 1
        except Exception:  # noqa: BLE001
            try:
                cap_faces.append(Part.Face(Part.Wire(loop_edges)))
                capped += 1
            except Exception:  # noqa: BLE001
                pass
    stats["organic_ev_capped"] = capped

    # 7. Sew everything into a shell -> solid.
    progress(f"Organic: sewing {len(reg_faces) + len(cap_faces)} faces")
    all_faces = reg_faces + cap_faces
    shell = Part.Shell(all_faces)
    sewn = shell.copy()
    try:
        sewn.sewShape(config.sew_tolerance)
    except Exception:  # noqa: BLE001
        sewn = shell

    shape = sewn
    watertight = False
    for cand in _closed_shells(sewn) + _closed_shells(shell):
        try:
            solid = Part.Solid(cand)
        except Exception:  # noqa: BLE001
            continue
        if solid.isValid():
            shape, watertight = solid, True
            break

    if not watertight:
        # Last resort: ShapeFix over the sewn shell can close small remaining gaps
        # (same healing pattern the boolean tier uses). Only ever adopted when it
        # validates into a solid, so it can't make things worse.
        healed = _heal_to_solid(sewn, Part)
        if healed is not None:
            shape, watertight = healed, True

    stats["organic_watertight"] = watertight
    if watertight:
        dev = _limit_deviation(shape, vertices, faces)
        stats["organic_deviation_mm"] = round(dev, 4)
        stats["organic_deviation_pct_diag"] = round(100.0 * dev / diag, 3)
        tol = max(config.organic_multipatch_dev_tol_abs,
                  config.organic_multipatch_dev_tol_rel * diag)
        stats["organic_deviation_tol_mm"] = round(tol, 4)
        if dev > tol:
            stats["organic_reason"] = (
                f"deviation {dev:.2f} mm > tol {tol:.2f} mm")
            progress(f"Organic: rejected — deviation {dev:.2f} mm > {tol:.2f} mm")
            return None, stats
        progress(f"Organic: watertight solid, deviation {dev:.2f} mm "
                 f"({stats['organic_deviation_pct_diag']}% of diag)")
        return shape, stats

    stats["organic_reason"] = "shell did not close into a solid"
    progress("Organic: shell did not close; declining")
    return None, stats


def _ev_regions(ev, quads, edge_faces):
    """Connected components of EV faces (faces adjacent across a shared cage
    edge). Each component is capped as one n-sided hole."""
    ev_set = set(ev)
    nbr: dict[int, set[int]] = {fi: set() for fi in ev}
    for fs in edge_faces.values():
        evs = [f for f in fs if f in ev_set]
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                nbr[evs[i]].add(evs[j])
                nbr[evs[j]].add(evs[i])
    seen: set[int] = set()
    regions = []
    for fi in ev:
        if fi in seen:
            continue
        comp = []
        stack = [fi]
        seen.add(fi)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in nbr[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        regions.append(comp)
    return regions


def _region_boundary_edges(region, quads, edge_faces, edge_curve):
    """Ordered B-spline boundary edges of one EV region's outer loop.

    The boundary cage edges are those on the region border (shared with a
    non-region face); each maps to a regular patch's exact B-spline curve via
    ``edge_curve``. Returns the edge list forming the loop, or ``None`` if any
    boundary edge lacks a regular-patch curve (so the region can't be sewn and
    is left to the caller to decline)."""
    region_set = set(region)
    boundary = []          # (a,b) cage-edge tuples on the region border
    for fi in region:
        q = [int(x) for x in quads[fi]]
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            key = (min(a, b), max(a, b))
            others = [f for f in edge_faces[key] if f not in region_set]
            if others:                        # border edge
                boundary.append((a, b))
    if not boundary:
        return None
    edges = []
    for a, b in boundary:
        e = edge_curve.get((min(a, b), max(a, b)))
        if e is None:
            return None                       # a border edge with no patch curve
        edges.append(e)
    return edges


def _index_patch_boundary_edges(regular, limit, Part, FreeCAD):
    """Map each cage edge (min,max cage index) to a regular patch's B-spline
    boundary edge over it, so EV caps can reuse the exact neighbour curve."""
    edge_curve: dict[tuple[int, int], object] = {}
    for fi, gi in regular:
        # Central-quad corners are grid[1][1],[1][2],[2][2],[2][1].
        corners = [gi[1][1], gi[1][2], gi[2][2], gi[2][1]]
        try:
            poles = [[FreeCAD.Vector(*limit[gi[i][j]]) for j in range(4)]
                     for i in range(4)]
            mults = [1] * 8
            knots = [0, 1, 2, 3, 4, 5, 6, 7]
            bs = Part.BSplineSurface()
            bs.buildFromPolesMultsKnots(poles, mults, mults, knots, knots,
                                        False, False, 3, 3)
            u0, u1, v0, v1 = bs.bounds()
        except Exception:  # noqa: BLE001
            continue
        # ``bs.value(u,v)`` indexes poles as row=u, col=v, so the central corners
        # map as: c0=(u0,v0)=grid[1][1], c1=(u0,v1)=grid[1][2],
        # c2=(u1,v1)=grid[2][2], c3=(u1,v0)=grid[2][1]. The four seam edges are
        # then the iso-curves below.
        try:
            b_edges = [
                bs.uIso(u0).toShape(v0, v1),   # c0-c1 (v varies at u0)
                bs.vIso(v1).toShape(u0, u1),   # c1-c2 (u varies at v1)
                bs.uIso(u1).toShape(v0, v1),   # c2-c3 (v varies at u1)
                bs.vIso(v0).toShape(u0, u1),   # c3-c0 (u varies at v0)
            ]
        except Exception:  # noqa: BLE001
            continue
        seam = [(corners[0], corners[1]), (corners[1], corners[2]),
                (corners[2], corners[3]), (corners[3], corners[0])]
        for (a, b), e in zip(seam, b_edges):
            edge_curve.setdefault((min(a, b), max(a, b)), e)
    return edge_curve


def _closed_shells(shape):
    shells = getattr(shape, "Shells", [])
    return [s for s in shells if s.isClosed()]


def _heal_to_solid(shape, Part):
    """ShapeFix a near-closed shell into a valid solid, or return None.

    Tries a few tolerances (the boolean tier's healing recipe). Only returns a
    result that validates into a single solid, so adoption can never regress."""
    for tol in (1e-3, 1e-2, 1e-1):
        try:
            s = shape.copy()
            s.fix(tol, tol, tol)
            for cand in _closed_shells(s):
                solid = Part.Solid(cand)
                if solid.isValid():
                    return solid
            if getattr(s, "Solids", []) and s.Solids[0].isValid():
                return s.Solids[0]
        except Exception:  # noqa: BLE001 - healing is best-effort
            pass
    return None


def _limit_deviation(shape, vertices, faces, n_sample: int = 60000) -> float:
    """Symmetric max point-cloud deviation between the solid's surface and the
    original mesh (area-weighted sampling, nearest-neighbour each way)."""
    try:
        from scipy.spatial import cKDTree
    except Exception:  # noqa: BLE001
        return float("inf")

    def sample(v, f, n):
        tri = v[f]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        tot = float(area.sum())
        if tot <= 0:
            return a
        cnt = np.maximum(1, np.round(n * area / tot)).astype(int)
        idx = np.repeat(np.arange(len(f)), cnt)
        r1 = np.sqrt(np.random.rand(len(idx)))
        r2 = np.random.rand(len(idx))
        return ((1 - r1)[:, None] * a[idx] + (r1 * (1 - r2))[:, None] * b[idx]
                + (r1 * r2)[:, None] * c[idx])

    try:
        tess = shape.tessellate(0.1)
        sv = np.array([[p.x, p.y, p.z] for p in tess[0]], float)
        sf = np.array(tess[1], int)
    except Exception:  # noqa: BLE001
        return float("inf")
    if len(sv) == 0 or len(sf) == 0:
        return float("inf")
    sp = sample(sv, sf, n_sample)
    mp = sample(np.asarray(vertices, float), np.asarray(faces, int), n_sample)
    d1, _ = cKDTree(mp).query(sp)
    d2, _ = cKDTree(sp).query(mp)
    return float(max(d1.max(), d2.max()))
