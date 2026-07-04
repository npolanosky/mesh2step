"""Region-level Candidate A: rebuild ONE residual organic region as a smooth
B-spline surface and integrate it with the analytic base by a guarded extrude+cut
boolean.

The whole-body organic tier (``organic.py``) only fires when the *entire* body is
organic (a sculpted cat) — it needs a closed-manifold cage. Real engineering parts
are analytic with one (or a few) large residual organic region(s): the port_cover
cast top, a curved cast pad. Those regions are neither analytic, nor an injective
height field the freeform sheet (Candidate B) can fit *directly* from the noisy
mesh, nor a closed organic body — so they ship faceted (high RTAF).

This pass claims them with the Catmull-Clark machinery used as a **clean sampler**:

  1. quad-remesh the OPEN region submesh (pynanoinstantmeshes handles open meshes —
     it preserves the boundary loop, verified) into an all-quad cage;
  2. subdivide once, then least-squares shrink-wrap the DENSE cage so its
     Catmull-Clark LIMIT surface approximates the region's real facets (fitting the
     already-subdivided cage keeps the open boundary from ballooning — the coarse
     cage's boundary limit overshoots). The limit points are a smooth, dense,
     noise-free sample of the region's true surface — far cleaner than the raw mesh;
  3. project the limit points onto the region's mean-normal (u,v) plane, grid them,
     and fit ONE ``Part.BSplineSurface`` through that clean grid. One B-spline
     surface (not a 5000-patch network) keeps the boolean tool a single prism, so
     the CUT is as fast as the proven freeform-sheet op; and
  4. integrate exactly like the freeform sheet (``builder._boolean_clean_freeform``):
     extrude the surface along the region axis into a solid tool and CUT it from the
     base, so the smooth surface replaces the faceted region. The boolean recomputes
     the intersection, so the region boundary need not match the analytic edges.

Every step is best-effort and returns ``None`` on failure; the builder wraps the
boolean in ``_try_boolean_step`` + an RTAF/bbox gate and reverts a region that
doesn't validly lower the residual tessellation (never regress). The remesher and
FreeCAD are both required — absent either, the pass declines and the region keeps
its faceted output.
"""

from __future__ import annotations

import numpy as np

from .catmull_clark import (
    catmull_clark_subdivide,
    fit_cage_to_mesh,
    limit_positions,
)
from .config import ConversionConfig
from .quadremesh import quad_remesh
from .segmentation import (
    OrganicRegion,
    _axis_basis,
    build_edge_adjacency,
    face_normals_and_areas,
)


def region_charts(vertices, faces, region, config: ConversionConfig):
    """Split a folded/wrapping organic region into injective sub-charts.

    A thin wrapping cast shell (the ``port_cover`` cast top, the tweezer shell, the
    ``patton_pad`` residual) has no single injective projection: its facet normals
    span far more than a hemisphere, so a single mean-normal (u,v) grid folds (its
    fitted surface area is many times its footprint) and the extrude tool
    self-intersects — exactly what :func:`build_region_surface` rejects with its fold
    guard. This decomposes the region into connected **single-sided** charts, each of
    which IS individually parametrizable: the growth fixes a chart axis from its seed
    facet's normal and admits a connected neighbour only while its normal stays within
    ``organic_region_chart_half_angle`` degrees of that axis (refreshed to the chart's
    running area-weighted mean). The half-angle gate is the injectivity ("stop at the
    fold") criterion — the direct multi-chart generalisation of the region pass's
    single-projection foldover gate. Each chart is returned as its own
    :class:`OrganicRegion` (``axis`` = chart mean normal) so the existing
    :func:`build_region_surface` + :func:`boolean_clean_region` reconstruct and cut it
    with no further special-casing.

    Facets are consumed largest-normal-cluster first (biggest chart = biggest RTAF
    win), and a chart is emitted only if it is large enough to reconstruct
    (``organic_region_chart_min_facets`` / a fraction of the region's area). Leftover
    facets too small or too curved for any chart are left faceted (never regress).
    Pure numpy; returns ``[]`` when the region does not usefully decompose."""
    fa_all = np.asarray(region.face_indices, dtype=int)
    if len(fa_all) < config.organic_region_chart_min_facets:
        return []
    normals, areas = face_normals_and_areas(vertices, faces)

    # Adjacency restricted to the region's own facets (charts must be connected
    # patches of the region, not leak into claimed/other geometry).
    region_set = set(int(x) for x in fa_all)
    adjacency = build_edge_adjacency(faces)
    nbr: dict[int, list[int]] = {int(x): [] for x in fa_all}
    for incident in adjacency.values():
        if len(incident) == 2:
            a, b = int(incident[0]), int(incident[1])
            if a in region_set and b in region_set:
                nbr[a].append(b)
                nbr[b].append(a)

    cos_half = float(np.cos(np.radians(config.organic_region_chart_half_angle)))
    region_area = float(areas[fa_all].sum())
    min_facets = int(config.organic_region_chart_min_facets)
    min_area = max(config.organic_region_chart_min_area,
                   config.organic_region_chart_min_area_frac * region_area)

    seed_normal_ref = {"n": None}

    def grow(seed: int, axis0: np.ndarray, taken: set[int]) -> list[int]:
        """Flood single-sided from ``seed``: admit a connected region facet only
        while its normal stays within the half-angle cone about ``axis0`` AND within
        that cone of the SEED's own normal. The second (seed-anchored) gate is what
        keeps each chart a COMPACT geodesic patch instead of an annular ring: on a
        shell that wraps around an axis (a sphere's equatorial band, a pipe), a
        pure-mean-axis cone would flood the whole ring — one connected chart that is
        not a height field. Anchoring to the fixed seed normal bounds the chart's
        angular extent to the half-angle, so every chart is a bounded cap that DOES
        project single-valued."""
        sn = seed_normal_ref["n"]
        comp = [seed]
        local = {seed}
        stack = [seed]
        while stack:
            x = stack.pop()
            for y in nbr[x]:
                if y in local or y in taken:
                    continue
                if float(normals[y] @ axis0) < cos_half:
                    continue
                if sn is not None and float(normals[y] @ sn) < cos_half:
                    continue
                local.add(y)
                comp.append(y)
                stack.append(y)
        return comp

    # Seed each chart from the most CENTRAL unused facet — the one whose normal is
    # closest to the mean normal of the still-unused facets. This grows the maximal
    # single-sided cap first (a rim-facet seed would tilt the chart axis and clip
    # the far side), then the leftover ring re-centres on its own mean and forms the
    # next chart. Recomputing the seed dynamically keeps every chart maximal.
    region_axis = np.asarray(region.axis, dtype=float)
    region_axis = region_axis / (np.linalg.norm(region_axis) or 1.0)
    taken: set[int] = set()
    charts: list[OrganicRegion] = []

    def next_seed() -> int | None:
        rest = [int(x) for x in fa_all if int(x) not in taken]
        if not rest:
            return None
        ra = np.array(rest, dtype=int)
        mn = (normals[ra] * areas[ra][:, None]).sum(axis=0)
        nn = float(np.linalg.norm(mn))
        centre = (mn / nn) if nn > 1e-6 else region_axis
        # Most-aligned facet with the unused-set mean normal (ties: larger area).
        aligned = normals[ra] @ centre
        best = int(ra[int(np.lexsort((areas[ra], aligned))[-1])])
        return best

    while True:
        seed = next_seed()
        if seed is None:
            break
        axis = normals[seed].astype(float).copy()
        seed_normal_ref["n"] = axis.copy()
        comp = grow(seed, axis, taken)
        # Refresh the axis to the chart's area-weighted mean normal and re-grow
        # once so a tilted seed facet doesn't clip one edge of the chart.
        fa0 = np.array(comp, dtype=int)
        mn0 = (normals[fa0] * areas[fa0][:, None]).sum(axis=0)
        nmn = float(np.linalg.norm(mn0))
        if nmn > 1e-6:
            new_axis = mn0 / nmn
            if float(new_axis @ axis) < 0.9995:
                axis = new_axis
                comp = grow(seed, axis, taken)
        for c in comp:
            taken.add(c)
        fa = np.array(comp, dtype=int)
        ar = areas[fa]
        tot = float(ar.sum())
        if len(fa) < min_facets or tot < min_area:
            continue
        mn = (normals[fa] * ar[:, None]).sum(axis=0)
        na = float(np.linalg.norm(mn))
        if na < 1e-6:
            continue
        axis = mn / na
        dots = normals[fa] @ axis
        fold = float(ar[dots < -0.05].sum()) / tot if tot > 0 else 1.0
        charts.append(OrganicRegion(
            face_indices=sorted(int(i) for i in comp),
            axis=axis, area=tot, foldover=fold))
        if len(charts) >= config.organic_region_chart_max:
            break
    charts.sort(key=lambda c: -c.area)
    return charts


def _region_submesh(vertices, faces, face_indices):
    """Extract a region's open submesh: its facets re-indexed to a compact vertex
    set. Returns ``(sub_verts, sub_faces)`` (numpy)."""
    fa = np.asarray(face_indices, dtype=int)
    tri = faces[fa]
    used = np.unique(tri)
    remap = {int(v): i for i, v in enumerate(used)}
    sub_v = vertices[used]
    sub_f = np.array([[remap[int(a)] for a in t] for t in tri], dtype=int)
    return sub_v, sub_f


def _fit_region_limit(vertices, faces, region, config: ConversionConfig):
    """Quad-remesh + subdivide + limit-fit a region into a dense, clean point cloud
    of its Catmull-Clark limit surface. Returns ``(limit_pts, sub_v, detail)`` or
    ``(None, None, detail)`` on failure. Pure numpy."""
    detail: dict = {"region_facets": len(region.face_indices),
                    "region_area": round(float(region.area), 1)}
    sub_v, sub_f = _region_submesh(vertices, faces, region.face_indices)
    diag = float(np.linalg.norm(sub_v.max(axis=0) - sub_v.min(axis=0))) or 1.0
    detail["diag_mm"] = round(diag, 2)

    base = int(config.organic_region_target_quads)
    target = int(min(base * 3, max(base, len(region.face_indices) // 6)))
    try:
        qv, quads = quad_remesh(sub_v, sub_f, target, config=config)
    except Exception as exc:  # noqa: BLE001 - incl. RemeshTimeout (hang/runaway)
        detail["reason"] = f"remesh failed: {exc}"
        return None, None, detail
    if quads.ndim != 2 or quads.shape[1] != 4 or len(quads) < 8:
        detail["reason"] = "cage not all-quad / too small"
        return None, None, detail
    quads = np.array([q for q in quads if len({int(x) for x in q}) == 4], dtype=int)
    if len(quads) < 8:
        detail["reason"] = "cage degenerate"
        return None, None, detail
    detail["cage_quads"] = int(len(quads))

    # Subdivide FIRST (densify), THEN fit the dense cage's limit to the mesh — the
    # coarse cage's open-boundary limit overshoots, so fitting after subdivision is
    # what keeps the shell from ballooning (port_cover: 8 mm -> <1 mm deviation).
    sv, sq = qv, quads
    for _ in range(max(1, int(config.organic_region_subdiv))):
        sv, sq = catmull_clark_subdivide(sv, sq)
    if config.organic_region_fit_iters > 0:
        sv = fit_cage_to_mesh(sv, sq, sub_v,
                              iterations=int(config.organic_region_fit_iters))
    limit = limit_positions(sv, sq)
    detail["limit_points"] = int(len(limit))
    return limit, sub_v, detail


def build_region_surface(vertices, faces, region, config: ConversionConfig, Part,
                         on_progress=None):
    """Build ONE ``Part.BSplineSurface`` for a residual organic region.

    Samples the region's clean Catmull-Clark limit point cloud onto a (u,v) grid in
    the plane perpendicular to the region's mean normal, then approximates one
    degree-3 B-spline surface through the grid (the same robust C1/centripetal
    regime the freeform sheet uses). Returns ``(surface, detail)`` or ``(None,
    detail)``. The surface's max deviation to the region facets is reported in
    ``detail`` for the caller's gate. Never raises for a geometry failure."""
    import FreeCAD  # type: ignore

    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    progress(f"  organic region: remesh+limit-fit {len(region.face_indices)} facets")
    limit, sub_v, detail = _fit_region_limit(vertices, faces, region, config)
    if limit is None:
        return None, detail

    ax = np.asarray(region.axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) or 1.0)
    e1, e2 = _axis_basis(ax)
    origin = limit.mean(axis=0)
    rel = limit - origin
    u = rel @ e1
    w = rel @ e2
    h = rel @ ax
    if float(u.max() - u.min()) < 1e-6 or float(w.max() - w.min()) < 1e-6:
        detail["reason"] = "degenerate (u,v) footprint"
        return None, detail

    try:
        from scipy.interpolate import griddata
    except Exception:  # noqa: BLE001
        detail["reason"] = "scipy unavailable"
        return None, detail
    # Grid the limit cloud into a height field h(u,v) by SMOOTH (linear)
    # interpolation. A nearest-node fill grabs low corner points on a non-rectangular
    # footprint and makes the surface dip below the region, which fragments the base
    # in the boolean; linear interpolation of the dense, noise-free limit cloud gives
    # a clean single-valued surface. The grid is oversized ~15% past the footprint
    # (like the freeform sheet's skirt) so the fitted surface extends into empty
    # space and the CUT trims the overshoot against the surrounding walls — a surface
    # stopping exactly at the boundary leaves a partial face and an invalid cut. The
    # oversized ring (outside the convex hull, where linear returns NaN) is filled by
    # nearest so the surface simply flat-extends there.
    ng = int(config.organic_region_grid)
    umid, wmid = 0.5 * (u.min() + u.max()), 0.5 * (w.min() + w.max())
    uhalf = 0.5 * (u.max() - u.min()) * 1.15
    whalf = 0.5 * (w.max() - w.min()) * 1.15
    us = np.linspace(umid - uhalf, umid + uhalf, ng)
    ws = np.linspace(wmid - whalf, wmid + whalf, ng)
    uu_grid, ww_grid = np.meshgrid(us, ws, indexing="ij")
    uw = np.column_stack([u, w])
    hg = griddata(uw, h, (uu_grid, ww_grid), method="linear")
    hg_nn = griddata(uw, h, (uu_grid, ww_grid), method="nearest")
    hg = np.where(np.isnan(hg), hg_nn, hg)
    grid = [[None] * ng for _ in range(ng)]
    for i in range(ng):
        for j in range(ng):
            p = origin + us[i] * e1 + ws[j] * e2 + float(hg[i, j]) * ax
            grid[i][j] = FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))

    # Approximate one degree-3 B-spline through the clean limit grid. Unlike the
    # freeform sheet (which fits the NOISY mesh, where pole saturation means it
    # interpolated noise), the source here is the smooth Catmull-Clark limit, so a
    # saturated pole count just means the surface needed the resolution — the honest
    # gate is the deviation-to-mesh the caller applies, not the pole count. We take
    # the tightest tolerance that approximates, and only reject an outright failure.
    surf = None
    for tol in (0.1, 0.25, 0.5, 1.0):
        cand = Part.BSplineSurface()
        try:
            cand.approximate(Points=grid, DegMin=3, DegMax=3, Tolerance=float(tol),
                             Continuity=1, ParamType="Centripetal")
        except Exception:  # noqa: BLE001
            continue
        surf = cand
        break
    if surf is None:
        detail["reason"] = "B-spline approximation failed"
        return None, detail

    # Deviation of the fitted surface to the region facets (the honest metric): the
    # worst distance from a real facet centroid to the surface. A surface that
    # misses the mesh would distort the part, so the caller gates on this.
    try:
        face = surf.toShape()
        if not face.isValid() or face.Area <= 0.0:
            detail["reason"] = "surface face invalid"
            return None, detail
    except Exception as exc:  # noqa: BLE001
        detail["reason"] = f"surface toShape failed: {exc}"
        return None, detail
    # Folded-surface guard: a region that does not project injectively folds the
    # (u,v) grid, giving a self-intersecting B-spline whose area balloons far past
    # the region's true footprint (and whose extruded boolean tool is pathological).
    # The footprint area (u,v bounding rectangle) is an upper bound on a clean
    # single-valued surface's projected area; a fitted area several times larger is
    # a fold, so reject it before the caller pays a boolean.
    footprint = float(u.max() - u.min()) * float(w.max() - w.min())
    if footprint > 0 and float(face.Area) > 4.0 * footprint:
        detail["reason"] = (f"surface folded (area {face.Area:.0f} >> footprint "
                            f"{footprint:.0f})")
        detail["surface_area"] = round(float(face.Area), 1)
        return None, detail
    dev = _surface_deviation(surf, sub_v)
    detail["deviation_mm"] = round(dev, 4)
    detail["poles"] = f"{surf.NbUPoles}x{surf.NbVPoles}"
    return surf, detail


def _surface_deviation(surf, region_verts, n_cap: int = 3000) -> float:
    """Max distance from the region's vertices to the fitted B-spline surface
    (nearest-parameter projection). One-sided (region -> surface): how far the
    surface strays from the facets it replaces."""
    import FreeCAD  # type: ignore

    pts = np.asarray(region_verts, dtype=float)
    if len(pts) > n_cap:
        idx = np.linspace(0, len(pts) - 1, n_cap).astype(int)
        pts = pts[idx]
    worst = 0.0
    for p in pts:
        vp = FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))
        try:
            uu, vv = surf.parameter(vp)
            q = surf.value(uu, vv)
            d = float((q - vp).Length)
        except Exception:  # noqa: BLE001
            continue
        if d > worst:
            worst = d
    return worst


def boolean_clean_region(solid, surf, region, Part, config=None):
    """Replace a region's faceted facets with its B-spline surface via a guarded CUT.

    Mirrors ``builder._boolean_clean_freeform``: extrude the (oversized) surface
    along the region's mean normal a full solid diagonal into a half-space tool and
    CUT it from ``solid``, removing the faceted overshoot and planting the smooth
    B-spline face. Raises on any failure so the caller's ``_try_boolean_step``
    reverts. Verifies the cut actually planted a B-spline face (a no-op cut is
    rejected)."""
    import FreeCAD  # type: ignore

    face = surf.toShape()
    if not face.isValid() or face.Area <= 0.0:
        raise ValueError("region surface face invalid")
    bb = solid.BoundBox
    diag = (bb.XLength ** 2 + bb.YLength ** 2 + bb.ZLength ** 2) ** 0.5 or 1.0
    axis = np.asarray(region.axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    tool = face.extrude(FreeCAD.Vector(*(axis * diag)))
    tool_solids = getattr(tool, "Solids", [])
    if not tool_solids:
        raise ValueError("region tool is not a solid")
    # Reject a self-intersecting extrude tool BEFORE the (expensive) cut: a region
    # that wraps too far under the extrude axis folds the prism onto itself, which
    # produces an invalid tool and a pathologically slow / invalid boolean. A quick
    # validity + prism-volume sanity check (the prism volume should be ~ the swept
    # face area x diag, so a self-intersecting collapse reads far smaller) catches
    # it cheaply so the caller defers the region instead of hanging.
    if not tool.isValid():
        raise ValueError("region tool self-intersects (invalid extrude)")
    expected_vol = float(face.Area) * float(diag)
    if expected_vol > 0 and tool.Volume < 0.25 * expected_vol:
        raise ValueError("region tool collapsed (wrapping extrude)")
    n_before = sum(1 for f in solid.Faces if "BSpline" in f.Surface.TypeId)
    # Timeout-guarded cut (P0): a valid B-spline tool can still send OCC's face-face
    # intersection into a multi-minute uninterruptible grind, so on a dense base run
    # it in a child process under a wall-clock ceiling (a timeout raises and the
    # caller's guard reverts, region stays faceted). See builder._maybe_isolated_cut.
    from .builder import _maybe_isolated_cut
    result = _maybe_isolated_cut(solid, tool, Part, config)
    solids = getattr(result, "Solids", [])
    if len(solids) != 1 or not solids[0].isValid():
        raise ValueError("region cut did not yield one valid solid")
    n_after = sum(1 for f in result.Faces if "BSpline" in f.Surface.TypeId)
    if n_after <= n_before:
        raise ValueError("region cut planted no B-spline face")
    return result
