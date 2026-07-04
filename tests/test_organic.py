"""Candidate A (whole-body organic multi-patch) against ground truth.

The numpy core — Catmull-Clark subdivision, limit projection, cage shrink-wrap,
and exact bicubic patch extraction (Stam) — is validated here with no FreeCAD:
a coarse quad sphere is the ground truth (its Catmull-Clark limit surface should
reproduce the sphere radius within tolerance, and each regular quad's bicubic
patch should sit on the sphere). The quad-remesh dependency and the OCC assembly
are exercised only when available/importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh2step import catmull_clark as cc
from mesh2step.config import ConversionConfig


# --------------------------------------------------------------------------- #
# Ground-truth quad sphere (numpy only)
# --------------------------------------------------------------------------- #

def _quad_sphere(nu: int = 16, nv: int = 10, r: float = 10.0):
    """A closed all-quad UV sphere: (verts (N,3), quads (M,4)). Poles are merged
    to single vertices so the mesh is closed and manifold (quads at the poles
    degenerate to triangles are avoided by starting rings one step in)."""
    verts = []
    for i in range(1, nv):                        # interior latitude rings
        th = np.pi * i / nv
        for j in range(nu):
            ph = 2 * np.pi * j / nu
            verts.append([r * np.sin(th) * np.cos(ph),
                          r * np.sin(th) * np.sin(ph),
                          r * np.cos(th)])
    north = len(verts)
    verts.append([0, 0, r])
    south = len(verts)
    verts.append([0, 0, -r])
    verts = np.array(verts, dtype=float)

    def ring(i, j):
        return i * nu + (j % nu)

    quads = []
    for i in range(nv - 2):                        # bands between interior rings
        for j in range(nu):
            quads.append([ring(i, j), ring(i, j + 1),
                          ring(i + 1, j + 1), ring(i + 1, j)])
    # Cap the poles with quads that fold to the pole vertex (degenerate corner):
    # instead, use triangles-as-quads by repeating the pole index — but that
    # breaks all-quad. So cap with quads sharing the pole via two adjacent
    # segments -> pole. Simpler: fan the top/bottom ring into quads to the pole
    # by pairing segments.
    for j in range(0, nu, 2):
        quads.append([ring(0, j), ring(0, j + 1), ring(0, j + 2), north])
    last = nv - 2
    for j in range(0, nu, 2):
        quads.append([ring(last, j + 2), ring(last, j + 1), ring(last, j), south])
    return verts, np.array(quads, dtype=int)


def test_catmull_clark_subdivide_grows_and_stays_closed():
    v, q = _quad_sphere()
    ok, _ = cc.is_closed_manifold(q)
    assert ok, "ground-truth quad sphere should be closed-manifold"
    v1, q1 = cc.catmull_clark_subdivide(v, q)
    # One CC step turns each quad into 4.
    assert len(q1) == 4 * len(q)
    ok1, detail = cc.is_closed_manifold(q1)
    assert ok1, f"subdivided mesh should stay closed-manifold: {detail}"


def test_catmull_clark_isolates_extraordinary_vertices():
    """Each subdivision step makes new vertices valence-4, so the fraction of
    faces touching an extraordinary vertex shrinks."""
    v, q = _quad_sphere()
    _, ev0 = cc.classify_patches(v, q)
    v1, q1 = cc.catmull_clark_subdivide(v, q)
    reg1, ev1 = cc.classify_patches(v1, q1)
    frac0 = len(ev0) / len(q)
    frac1 = len(ev1) / len(q1)
    assert frac1 < frac0, "EV-face fraction should drop after subdivision"
    assert reg1, "most faces should be regular after one subdivision"


def test_limit_surface_reproduces_sphere_radius():
    """The Catmull-Clark LIMIT positions of a quad-sphere cage lie close to the
    sphere; each regular quad's bicubic patch samples on the sphere too."""
    v, q = _quad_sphere(nu=20, nv=12, r=10.0)
    v1, q1 = cc.catmull_clark_subdivide(v, q)
    limit = cc.limit_positions(v1, q1)
    radii = np.linalg.norm(limit, axis=1)
    # Cage limit points sit within a few % of R (a coarse cage, so allow slack).
    assert abs(radii.mean() - 10.0) < 0.6
    # Sample regular patches and confirm the emitted surface is near the sphere.
    regular, _ = cc.classify_patches(v1, q1)
    assert regular
    devs = []
    for _fi, gi in regular[:200]:
        grid = limit[gi]
        for u in (0.25, 0.5, 0.75):
            for w in (0.25, 0.5, 0.75):
                p = cc.bspline_patch_value(grid, u, w)
                devs.append(abs(np.linalg.norm(p) - 10.0))
    devs = np.array(devs)
    assert devs.max() < 0.8, f"patch deviation too large: {devs.max():.3f}"


def test_cage_fit_reduces_limit_deviation():
    """Shrink-wrapping the cage to a target point cloud lowers the limit-surface
    deviation to that cloud (the dimensional-honesty step)."""
    v, q = _quad_sphere(nu=20, nv=12, r=10.0)
    # Dense sphere point cloud as the fit target.
    target = []
    for i in range(1, 40):
        th = np.pi * i / 40
        for j in range(60):
            ph = 2 * np.pi * j / 60
            target.append([10 * np.sin(th) * np.cos(ph),
                           10 * np.sin(th) * np.sin(ph),
                           10 * np.cos(th)])
    target = np.array(target)

    def limit_dev(cage):
        lim = cc.limit_positions(cage, q)
        return float(np.abs(np.linalg.norm(lim, axis=1) - 10.0).mean())

    before = limit_dev(v)
    fitted = cc.fit_cage_to_mesh(v, q, target, iterations=3)
    after = limit_dev(fitted)
    assert after <= before + 1e-9, "cage fit should not worsen limit deviation"


def test_regular_patch_grid_is_4x4_of_distinct_vertices():
    v, q = _quad_sphere(nu=20, nv=12)
    v1, q1 = cc.catmull_clark_subdivide(v, q)
    regular, _ = cc.classify_patches(v1, q1)
    assert regular
    _fi, gi = regular[0]
    assert gi.shape == (4, 4)
    # Central quad corners must be the four grid[1..2][1..2] entries.
    corners = {gi[1][1], gi[1][2], gi[2][2], gi[2][1]}
    assert len(corners) == 4


# --------------------------------------------------------------------------- #
# Routing gate (numpy only)
# --------------------------------------------------------------------------- #

def test_routing_gate_respects_flag_and_residual():
    from mesh2step import organic

    faces = np.zeros((100, 3), dtype=int)
    # Disabled flag -> never attempt.
    cfg_off = ConversionConfig(organic_multipatch=False)
    assert organic.should_attempt({"rtaf": 0.9}, faces, cfg_off) is False
    # Low residual -> not organic, don't attempt (even if remesher present).
    cfg = ConversionConfig(organic_multipatch_min_residual=0.6)
    assert organic.should_attempt({"rtaf": 0.2}, faces, cfg) is False
    # Prismatic-signal veto: high RTAF but many swept walls -> vetoed (the
    # gridfinity_bin_1x1x3 P0; its sew merely failed to close, it isn't organic).
    cfg_v = ConversionConfig(organic_multipatch_max_swept_walls=6)
    assert organic.should_attempt(
        {"rtaf": 0.9, "swept_walls_detected": 11}, faces, cfg_v) is False


def test_quad_cage_validation_rejects_open_and_nonquad():
    from mesh2step.quadremesh import validate_quad_cage

    # An open quad grid (boundary edges everywhere) is rejected.
    grid = np.array([[i, i + 1, i + 5, i + 4] for i in range(12)], dtype=int)
    ok, detail = validate_quad_cage(grid)
    assert ok is False
    assert "reason" in detail
    # A closed quad sphere is accepted (>=8 quads, closed, manifold, all-quad).
    _v, q = _quad_sphere(nu=16, nv=10)
    ok2, detail2 = validate_quad_cage(q)
    assert ok2 is True, detail2


# --------------------------------------------------------------------------- #
# Quad remesh + OCC assembly (dependency / FreeCAD required)
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - environment probe
    from mesh2step.quadremesh import available

    HAVE_PNIM = available()
except Exception:  # noqa: BLE001
    HAVE_PNIM = False


try:  # pragma: no cover - environment probe
    import FreeCAD  # type: ignore  # noqa: F401
    import Part  # type: ignore  # noqa: F401

    from mesh2step.organic import _occ

    HAVE_OCC = _occ()
except Exception:  # noqa: BLE001
    HAVE_OCC = False


@pytest.mark.skipif(not HAVE_PNIM, reason="pynanoinstantmeshes not installed")
def test_quad_remesh_produces_closed_all_quad_cage():
    from mesh2step.quadremesh import quad_remesh, validate_quad_cage

    v, q = _quad_sphere(nu=24, nv=16, r=10.0)
    # Triangulate the quad sphere to feed the remesher a triangle mesh.
    tris = np.array([[a, b, c] for a, b, c, d in q] + [[a, c, d] for a, b, c, d in q])
    qv, quads = quad_remesh(v, tris, 120)
    assert quads.shape[1] == 4, "remesher must emit all-quad output"
    ok, detail = validate_quad_cage(quads)
    # The remesher is robust on a clean closed sphere; the cage should validate.
    assert ok, f"quad cage should be closed-manifold: {detail}"


def test_repair_quad_cage_fills_boundary_hole():
    """A quad cage with one small even boundary hole is repaired to closed-manifold
    by the centroid-fan fill (pure numpy, no remesher/FreeCAD)."""
    from mesh2step.quadremesh import repair_quad_cage, validate_quad_cage

    v, q = _quad_sphere(nu=16, nv=10, r=10.0)
    # Punch a hole: drop 4 quads forming an even boundary loop.
    holed = np.array([row for i, row in enumerate(q) if i not in (0, 1, 16, 17)])
    ok0, _ = validate_quad_cage(holed)
    assert not ok0, "punched cage should be open"
    rv, rq = repair_quad_cage(v, holed)
    ok1, detail = validate_quad_cage(rq)
    assert ok1, f"repaired cage should be closed-manifold: {detail}"


@pytest.mark.skipif(not HAVE_PNIM, reason="pynanoinstantmeshes not installed")
def test_build_quad_cage_backs_off_to_clean_target():
    """build_quad_cage returns a usable closed-manifold cage on a clean sphere,
    backing off to a coarser (cleaner) target if the requested one isn't clean."""
    from mesh2step.quadremesh import build_quad_cage

    v, q = _quad_sphere(nu=24, nv=16, r=10.0)
    tris = np.array([[a, b, c] for a, b, c, d in q] + [[a, c, d] for a, b, c, d in q])
    qv, quads, detail = build_quad_cage(v, tris, 220)
    assert detail.get("ok"), f"cage should be usable: {detail}"
    assert quads is not None and quads.shape[1] == 4


@pytest.mark.skipif(not (HAVE_PNIM and HAVE_OCC),
                    reason="needs pynanoinstantmeshes + FreeCAD OCC bindings")
def test_organic_shell_closes_watertight_sphere():
    """End-to-end: the whole-body organic pipeline reconstructs a ground-truth
    sphere as a WATERTIGHT solid that re-reads valid, with deviation and RTAF at
    the expected accuracy. This is the closure regression guard."""
    import Part  # type: ignore

    from mesh2step import organic
    from mesh2step.config import ConversionConfig

    v, q = _quad_sphere(nu=24, nv=16, r=10.0)
    tris = np.array([[a, b, c] for a, b, c, d in q] + [[a, c, d] for a, b, c, d in q])
    shape, stats = organic.build_organic_shell(v, tris, ConversionConfig())
    assert stats.get("organic_watertight"), stats.get("organic_reason")
    assert shape is not None and shape.isValid()
    assert shape.Solids and shape.Solids[0].Shells[0].isClosed()
    assert stats.get("organic_free_edges") == 0
    # Re-read via STEP round-trip (the pipeline's export-revalidation gate).
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        shape.exportStep(tmp.name)
        reread = Part.Shape()
        reread.read(tmp.name)
    assert reread.isValid() and len(reread.Solids) == 1
    assert reread.Solids[0].Shells[0].isClosed()
    # Radius deviation of the reconstructed surface to the true R=10 sphere.
    tess = shape.tessellate(0.05)
    sv = np.array([[p.x, p.y, p.z] for p in tess[0]])
    dev = np.abs(np.linalg.norm(sv, axis=1) - 10.0)
    assert dev.max() < 0.5, f"radius deviation too high: {dev.max():.3f} mm"
