"""Candidate A assembly: quad-patch network -> B-spline shell -> solid.

This is the only FreeCAD-touching part of the whole-body organic pipeline
(docs/ORGANIC_CONVERSION_RESEARCH.md, Candidate A). The pure-numpy stages live
in ``quadremesh`` (remesh + cage validation) and ``catmull_clark`` (subdivision,
limit-fit, patch classification); here we

  1. run those stages to get a fitted, subdivided all-quad cage (with cage
     repair + coarser-target back-off so a remesher that leaves small holes on a
     given target still yields a closed-manifold cage);
  2. emit one exact uniform bicubic B-spline face per regular quad (Stam) —
     adjacent regular patches share control rows so their boundary curves are
     mathematically identical (verified bit-identical to ~1e-15);
  3. cap each *extraordinary-vertex (EV) region* (a connected cluster of EV
     faces) with one filled surface built over the region's outer boundary loop,
     where every boundary edge reuses the exact neighbouring regular patch's
     boundary iso-curve. Two fillers are tried — ``BRepOffsetAPI_MakeFilling``
     first, then ``BRepFill_Filling`` pinned by the region's interior limit
     points — and each candidate is geometry-validated (no balloon, no local
     spike) so a self-intersecting fill is rejected; and
  4. sew regular + cap faces into a shell and make a solid.

If some EV region can't be validly capped, the whole build is retried one
Catmull-Clark level finer (each level shrinks the EV regions), up to a small
ceiling; on failure at every level the tier declines.

The closure recipe uses OpenCASCADE directly through the ``OCC.Core`` bindings
that FreeCAD ships (pythonocc-core, verified importable inside FreeCAD's
interpreter). The earlier ``Part``-only attempt could not close: the regular
patches sew flawlessly (their shared boundary curves are bit-identical), but
``Part.makeFilledFace`` EV caps never re-sewed into the shell. Building the caps
from the neighbour patches' exact ``Geom`` boundary curves and running a single
low-level ``BRepBuilderAPI_Sewing`` closes the shell to zero free edges (GT
sphere: watertight, isValid, re-reads valid, max radius deviation 0.21 mm, RTAF
0). When pythonocc is unavailable the tier declines gracefully.

Every result is *gated*: the caller adopts it only when it is a watertight solid
that re-reads valid and lowers RTAF — otherwise the pipeline keeps its existing
output (never regress). Because every face is a B-spline (not a planar strip),
a successful result carries RTAF ~ 0.

The construction is deterministic; the remesh is made deterministic upstream, so
the whole path is reproducible.
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
from .quadremesh import build_quad_cage


# The uniform bicubic B-spline patch of a 4x4 control net is defined over the
# central knot span [3,4]x[3,4] (uniform knots [0..7]); that central quad is the
# limit surface over the cage face. All patches share this parametrisation, and
# the ``side`` index below labels the four central-span boundary iso-curves so an
# EV cap can reuse a neighbour patch's exact boundary curve.
_UNIFORM_KNOTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
_SPAN_LO, _SPAN_HI = 3.0, 4.0


def _occ():
    """Import the OpenCASCADE bindings FreeCAD ships (pythonocc-core), or None.

    The whole low-level closure recipe lives behind this: when the bindings are
    absent (some FreeCAD builds), ``build_organic_shell`` declines and the
    pipeline keeps its existing output (never regress)."""
    try:
        from OCC.Core.BRepBuilderAPI import (  # type: ignore  # noqa: F401
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_Sewing,
        )
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeFilling  # noqa: F401
        from OCC.Core.Geom import (  # noqa: F401
            Geom_BSplineSurface,
            Geom_TrimmedCurve,
        )

        return True
    except Exception:  # noqa: BLE001
        return False


def _occ_bspline_surface(grid_pts):
    """A ``Geom_BSplineSurface`` (uniform bicubic) from a 4x4 control grid."""
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

    poles = TColgp_Array2OfPnt(1, 4, 1, 4)
    for i in range(4):
        for j in range(4):
            p = grid_pts[i][j]
            poles.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    uk = TColStd_Array1OfReal(1, 8)
    vk = TColStd_Array1OfReal(1, 8)
    um = TColStd_Array1OfInteger(1, 8)
    vm = TColStd_Array1OfInteger(1, 8)
    for k in range(8):
        uk.SetValue(k + 1, _UNIFORM_KNOTS[k])
        vk.SetValue(k + 1, _UNIFORM_KNOTS[k])
        um.SetValue(k + 1, 1)
        vm.SetValue(k + 1, 1)
    return Geom_BSplineSurface(poles, uk, vk, um, vm, 3, 3, False, False)


def _occ_patch_face(surf):
    """The trimmed Face over the central span [3,4]x[3,4] of ``surf`` (with
    pcurves — ``BRepBuilderAPI_MakeFace`` on a surface + bounds builds them)."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    mf = BRepBuilderAPI_MakeFace(surf, _SPAN_LO, _SPAN_HI, _SPAN_LO, _SPAN_HI, 1e-7)
    return mf.Face() if mf.IsDone() else None


def _occ_boundary_curve(surf, side):
    """The exact central-span boundary iso-curve of a patch as a Geom curve.

    ``side`` labels the four sides of the central quad, matching the corner
    order ``grid[1][1], grid[1][2], grid[2][2], grid[2][1]``:
    0 = corner0->corner1 (u = _SPAN_LO), 1 = corner1->corner2 (v = _SPAN_HI),
    2 = corner2->corner3 (u = _SPAN_HI), 3 = corner3->corner0 (v = _SPAN_LO).
    Two adjacent patches return the *same* curve here (their control rows match),
    so an EV cap built from it shares the neighbour's boundary exactly."""
    from OCC.Core.Geom import Geom_TrimmedCurve

    if side == 0:
        c = surf.UIso(_SPAN_LO)
    elif side == 1:
        c = surf.VIso(_SPAN_HI)
    elif side == 2:
        c = surf.UIso(_SPAN_HI)
    else:
        c = surf.VIso(_SPAN_LO)
    return Geom_TrimmedCurve(c, _SPAN_LO, _SPAN_HI)


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
    import Part  # type: ignore

    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    stats: dict = {"organic_attempted": True}
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0

    # 1. Quad remesh -> coarse all-quad cage. The remesher can leave small holes /
    # non-manifold quads at a given target; build_quad_cage repairs them and, if
    # needed, backs off to a coarser (more robust) target until the cage validates
    # closed-manifold-all-quad. On failure it returns None and we decline.
    target = int(config.organic_multipatch_target_quads)
    progress(f"Organic: quad-remeshing to ~{target} quads")
    qv, quads, cage_detail = build_quad_cage(
        vertices, faces, target, on_progress=on_progress)
    stats["organic_cage"] = cage_detail
    ok = cage_detail.get("ok", False)
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

    # The low-level closure recipe needs the OCC bindings; without them we can't
    # build the shared-curve EV caps, so decline (never regress).
    if not _occ():
        stats["organic_reason"] = "OCC bindings (pythonocc) unavailable"
        progress("Organic: OCC bindings unavailable; declining")
        return None, stats

    # 4-7. Subdivide, extract patches, cap EVs, sew. Each subdivision level isolates
    # extraordinary vertices further and shrinks their capped regions, so a cap that
    # self-intersects at a sharp feature (and is rejected) usually behaves one level
    # finer. Start at the configured level and escalate up to a small ceiling.
    base_subdiv = max(1, int(config.organic_multipatch_subdiv))
    shape = None
    watertight = False
    for nsub in range(base_subdiv, base_subdiv + 3):
        sv, sq = qv, quads
        for _ in range(nsub):
            sv, sq = catmull_clark_subdivide(sv, sq)
        progress(f"Organic: cage subdivided {nsub}x to {len(sq)} quads")
        shape, watertight, sub_stats = _build_shell_from_cage(sv, sq, diag, Part, progress)
        stats.update(sub_stats)
        stats["organic_subdiv_used"] = nsub
        if watertight:
            break
        progress(f"Organic: not closed at subdiv {nsub} "
                 f"({sub_stats.get('organic_reason', '?')})")

    stats["organic_watertight"] = watertight
    if watertight:
        stats.pop("organic_reason", None)   # clear any stale reason from a coarser level
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


def _build_shell_from_cage(qv, quads, diag, Part, progress):
    """Extract patches from one subdivided cage, cap EVs, sew into a solid.

    Returns ``(shape, watertight, stats)``. Split out from ``build_organic_shell``
    so it can be retried at a finer subdivision when an EV cap self-intersects."""
    stats: dict = {}
    limit = limit_positions(qv, quads)
    regular, ev = classify_patches(qv, quads)
    stats["organic_patches"] = len(regular)
    stats["organic_ev_faces"] = len(ev)
    if not regular:
        stats["organic_reason"] = "no regular patches"
        return None, False, stats

    # One Geom_BSplineSurface + trimmed Face per regular quad. Keep the surfaces so
    # EV caps can reuse their exact boundary curves; the cage-edge -> (surface, side)
    # index lets a cap look up the neighbour patch's boundary iso-curve.
    reg_surfs: dict = {}
    reg_faces = []
    edge_curve: dict = {}   # (min,max cage vtx) -> (regular fi, side)
    for fi, gi in regular:
        try:
            surf = _occ_bspline_surface(limit[gi])
            face = _occ_patch_face(surf)
            if face is None:
                continue
        except Exception:  # noqa: BLE001 - a bad grid must not abort the shell
            continue
        reg_surfs[fi] = surf
        reg_faces.append(face)
        corners = [int(gi[1][1]), int(gi[1][2]), int(gi[2][2]), int(gi[2][1])]
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            edge_curve.setdefault((min(a, b), max(a, b)), (fi, k))
    if not reg_faces:
        stats["organic_reason"] = "no B-spline faces built"
        return None, False, stats

    # Cap each EV region with one validated MakeFilling surface over its outer
    # boundary loop (every boundary edge reuses the neighbour regular patch's exact
    # iso-curve). A region whose cap self-intersects is rejected -> we decline this
    # subdivision level so the caller retries finer (never regress).
    edge_faces = quad_edge_faces(quads)
    ev_regions = _ev_regions(ev, quads, edge_faces)
    stats["organic_ev_regions"] = len(ev_regions)
    cap_faces = []
    capped = 0
    for region in ev_regions:
        cap = _cap_ev_region(region, quads, edge_faces, edge_curve, reg_surfs,
                             limit, diag)
        if cap is not None:
            cap_faces.append(cap)
            capped += 1
    stats["organic_ev_capped"] = capped
    if capped < len(ev_regions):
        stats["organic_reason"] = (
            f"only {capped}/{len(ev_regions)} EV regions validly capped")
        return None, False, stats

    edge_len = float(np.median([
        np.linalg.norm(qv[int(q[k])] - qv[int(q[(k + 1) % 4])])
        for q in quads for k in range(4)])) if len(quads) else 1.0
    progress(f"Organic: sewing {len(reg_faces) + len(cap_faces)} faces")
    shape, watertight = _sew_to_solid(reg_faces + cap_faces, Part, stats, edge_len)
    if not watertight:
        stats.setdefault("organic_reason", "shell did not close into a solid")
    return shape, watertight, stats


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


def _region_boundary_cage_edges(region, quads, edge_faces):
    """The cage edges on an EV region's outer border (shared with a non-region
    face), as ``(min,max)`` vertex keys — the loop the cap must span."""
    region_set = set(region)
    boundary = []
    for fi in region:
        q = [int(x) for x in quads[fi]]
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            key = (min(a, b), max(a, b))
            if any(f not in region_set for f in edge_faces[key]):
                boundary.append(key)
    return boundary


def _ordered_boundary_loop(boundary):
    """Chain the unordered ``(min,max)`` boundary edges into an ordered vertex-pair
    loop ``[(v0,v1),(v1,v2),...,(vn,v0)]``, or ``None`` if they don't form one
    simple closed loop (a planar cap wire needs the edges in loop order)."""
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(vs) != 2 for vs in adj.values()):
        return None
    start = boundary[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        nxts = [x for x in adj[cur] if x != prev]
        if not nxts:
            return None
        nb = nxts[0]
        if nb == start:
            break
        if nb in loop:
            return None
        loop.append(nb)
        prev, cur = cur, nb
        if len(loop) > len(boundary):
            return None
    return [(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]


def _region_interior_points(region, quads, boundary, limit):
    """Limit positions of an EV region's non-boundary (interior) cage vertices,
    plus per-face centroids — used as ``BRepFill_Filling`` shape constraints so the
    cap follows the surface instead of ballooning."""
    bverts = {v for key in boundary for v in key}
    pts = []
    interior = {int(x) for fi in region for x in quads[fi]} - bverts
    for vid in interior:
        pts.append(limit[vid])
    for fi in region:
        pts.append(limit[[int(x) for x in quads[fi]]].mean(0))
    return pts


def _cap_ok(face, region, quads, limit, diag):
    """Accept a cap only if it neither balloons (bbox) nor spikes (deviation).

    We do NOT gate on ``BRepCheck_Analyzer(face)`` — many valid, sewable caps are
    only C0 and trip it yet heal into a valid solid. Instead two geometric guards
    catch the genuinely broken (self-intersecting) fills: the bounding box must
    stay within half the part diagonal (a global balloon), and the cap surface's
    max distance to the region's limit-surface samples must stay small relative to
    the region (a local spike). Returns True/False."""
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box

    try:
        bb = Bnd_Box()
        brepbndlib.Add(face, bb)
        xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
        region_pts = np.array([limit[int(x)] for fi in region for x in quads[fi]])
        region_ext = float(np.linalg.norm(region_pts.max(0) - region_pts.min(0)))
        if max(xmax - xmin, ymax - ymin, zmax - zmin) > 0.5 * diag:
            return False
    except Exception:  # noqa: BLE001
        return False
    # Local-spike guard: sample the cap and measure its farthest excursion from the
    # region's limit points. A clean cap hugs them; a self-intersecting one spikes.
    try:
        import Part  # type: ignore
        from scipy.spatial import cKDTree

        pf = Part.__fromPythonOCC__(face)
        tess = pf.tessellate(max(1e-3, 0.02 * region_ext))
        sv = np.array([[p.x, p.y, p.z] for p in tess[0]])
        if len(sv) == 0:
            return True
        d, _ = cKDTree(region_pts).query(sv)
        # Allow up to ~40% of the region extent (curved caps legitimately bulge a
        # little past the corner samples); a self-intersection spikes far past it.
        if d.max() > 0.4 * max(region_ext, 1e-6):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _cap_ev_region(region, quads, edge_faces, edge_curve, reg_surfs, limit, diag):
    """One OCC filled Face over an EV region's outer boundary loop, or ``None``.

    Each boundary cage edge maps (via ``edge_curve``) to the neighbouring regular
    patch's ``(surface, side)``; the exact boundary iso-curve of that patch is the
    cap's boundary edge, so the cap shares the shell geometry bit-for-bit and sews
    without a gap. Two fillers are tried: ``BRepOffsetAPI_MakeFilling`` (fast, good
    on gentle regions), then ``BRepFill_Filling`` with the region's interior limit
    points as shape constraints (more robust on saddle/sharp regions — it pins the
    surface to the actual shape). Each candidate is validated by :func:`_cap_ok`;
    the first acceptable one is returned, else ``None`` (caller declines / retries
    at a finer subdivision — never regress)."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.BRepFill import BRepFill_Filling
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCC.Core.GeomAbs import GeomAbs_C0
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopoDS import topods

    boundary = _region_boundary_cage_edges(region, quads, edge_faces)
    if len(boundary) < 3:
        return None
    curves = []
    for key in boundary:
        ref = edge_curve.get(key)
        if ref is None:
            return None                       # border edge with no patch curve
        fi, side = ref
        surf = reg_surfs.get(fi)
        if surf is None:
            return None
        try:
            curves.append(_occ_boundary_curve(surf, side))
        except Exception:  # noqa: BLE001
            return None

    # Attempt 1: BRepOffsetAPI_MakeFilling (smooth, cheap).
    try:
        fill = BRepOffsetAPI_MakeFilling()
        for c in curves:
            fill.Add(BRepBuilderAPI_MakeEdge(c).Edge(), GeomAbs_C0)
        fill.Build()
        if fill.IsDone():
            face = topods.Face(fill.Shape())
            if _cap_ok(face, region, quads, limit, diag):
                return face
    except Exception:  # noqa: BLE001
        pass

    # Attempt 2: BRepFill_Filling constrained by the region's interior limit points.
    try:
        bf = BRepFill_Filling()
        for c in curves:
            bf.Add(BRepBuilderAPI_MakeEdge(c).Edge(), GeomAbs_C0, True)
        for p in _region_interior_points(region, quads, boundary, limit):
            bf.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        bf.Build()
        if bf.IsDone():
            face = topods.Face(bf.Face())
            if _cap_ok(face, region, quads, limit, diag):
                return face
    except Exception:  # noqa: BLE001
        pass

    # Attempt 3: a planar face over the (ordered) boundary loop. The regions that
    # defeat both fillers are the tiny, near-planar knife-edge EVs at sharp
    # features (verified: extent ~0.7x0.06x0.3 mm, planarity 0.02 on the cat).
    # A planar face over their exact boundary curves closes them within tolerance.
    try:
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakeWire,
        )

        loop = _ordered_boundary_loop(boundary)
        if loop is not None:
            mw = BRepBuilderAPI_MakeWire()
            ok = True
            for a, b in loop:
                ref = edge_curve.get((min(a, b), max(a, b)))
                if ref is None:
                    ok = False
                    break
                rfi, side = ref
                mw.Add(BRepBuilderAPI_MakeEdge(
                    _occ_boundary_curve(reg_surfs[rfi], side)).Edge())
            if ok:
                mf = BRepBuilderAPI_MakeFace(mw.Wire(), True)
                if mf.IsDone():
                    face = topods.Face(mf.Face())
                    if _cap_ok(face, region, quads, limit, diag):
                        return face
    except Exception:  # noqa: BLE001
        pass
    return None


def _solid_from_shell(shell):
    """A valid, outward-oriented OCC solid from a closed shell, or ``None``.

    First tries the shell as-is. If the raw solid is invalid — the ``MakeFilling``
    EV caps can be C0 / slightly self-intersecting, which OCC flags but which heal
    trivially — run ``ShapeFix_Shell`` + ``ShapeFix_Solid`` (research doc's allowed
    last-step healing). Only a solid that passes ``BRepCheck_Analyzer`` is
    returned, so the caller's watertight/deviation/re-read gates still apply."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepLib import breplib
    from OCC.Core.ShapeFix import ShapeFix_Shell, ShapeFix_Solid

    def _mk(sh):
        mk = BRepBuilderAPI_MakeSolid(sh)
        mk.Build()
        solid = mk.Solid()
        breplib.OrientClosedSolid(solid)
        return solid

    try:
        solid = _mk(shell)
        if BRepCheck_Analyzer(solid).IsValid():
            return solid
    except Exception:  # noqa: BLE001
        pass
    # Heal: fix the shell, then fix the solid.
    try:
        sfs = ShapeFix_Shell(shell)
        sfs.Perform()
        healed_shell = sfs.Shell()
        solid = _mk(healed_shell)
        if not BRepCheck_Analyzer(solid).IsValid():
            sfsol = ShapeFix_Solid(solid)
            sfsol.Perform()
            solid = sfsol.Solid()
        if BRepCheck_Analyzer(solid).IsValid():
            return solid
    except Exception:  # noqa: BLE001
        pass
    return None


def _sew_once(occ_faces, tol):
    """One ``BRepBuilderAPI_Sewing`` pass; returns ``(sewed_shape, n_free_edges)``."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing

    sew = BRepBuilderAPI_Sewing(tol)
    for f in occ_faces:
        sew.Add(f)
    sew.Perform()
    return sew.SewedShape(), int(sew.NbFreeEdges())


def _sew_to_solid(occ_faces, Part, stats, edge_len):
    """Sew OCC faces into a shell and make a solid. Returns ``(shape, watertight)``.

    Regular Stam patches share bit-identical boundary curves and sew at a tight
    tolerance, but ``MakeFilling`` EV caps re-approximate their boundary and can
    drift by a small fraction of the cage edge length. We escalate the sewing
    tolerance over a short ladder (tight first, so smooth parts stay tight) up to
    a cap proportional to the cage edge length, stopping at the first tolerance
    that closes the shell into a valid solid. ``edge_len`` is the median subdivided
    cage edge length; the loosest tolerance stays ~1 % of it (sub-feature, and the
    caller's deviation + bbox gates still protect correctness). The returned
    ``shape`` is a FreeCAD ``Part`` shape; on failure it is the best open shell and
    ``watertight`` is False."""
    from OCC.Core.TopAbs import TopAbs_SHELL
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    cap = min(5e-2, 0.03 * max(edge_len, 1e-3))
    tols = sorted({1e-4, 1e-3, 1e-2, cap})
    best_open = None
    best_free = None
    for tol in tols:
        result, n_free = _sew_once(occ_faces, tol)
        if best_free is None or n_free < best_free:
            best_free = n_free
            try:
                best_open = Part.__fromPythonOCC__(result)
            except Exception:  # noqa: BLE001
                pass
        exp = TopExp_Explorer(result, TopAbs_SHELL)
        while exp.More():
            shell = topods.Shell(exp.Current())
            exp.Next()
            solid = _solid_from_shell(shell)
            if solid is None:
                continue
            try:
                part_solid = Part.__fromPythonOCC__(solid)
            except Exception:  # noqa: BLE001
                continue
            solids = getattr(part_solid, "Solids", [])
            if solids and solids[0].isValid() and solids[0].Shells and \
                    solids[0].Shells[0].isClosed():
                stats["organic_free_edges"] = 0
                stats["organic_sew_tol"] = tol
                return solids[0], True
    stats["organic_free_edges"] = best_free if best_free is not None else -1
    return best_open, False


def _limit_deviation(shape, vertices, faces, n_sample: int = 60000) -> float:
    """Symmetric max point-cloud deviation between the solid's surface and the
    original mesh (area-weighted sampling, nearest-neighbour each way).

    Uses a fixed-seed RNG so the deviation figure (and therefore the accept/reject
    gate) is reproducible run-to-run."""
    try:
        from scipy.spatial import cKDTree
    except Exception:  # noqa: BLE001
        return float("inf")

    rng = np.random.default_rng(0)

    def sample(v, f, n):
        tri = v[f]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        tot = float(area.sum())
        if tot <= 0:
            return a
        cnt = np.maximum(1, np.round(n * area / tot)).astype(int)
        idx = np.repeat(np.arange(len(f)), cnt)
        r1 = np.sqrt(rng.random(len(idx)))
        r2 = rng.random(len(idx))
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
