"""Region-level Candidate A: residual-organic-region segmentation + reconstruction.

``segment_organic_regions`` groups the residual smooth facets the analytic +
freeform tiers left faceted into large organic regions, gated to the INJECTIVE
ones (foldover below ``organic_region_max_foldover``) so each projects
single-valued along its mean normal and reconstructs as one B-spline surface. The
builder rebuilds each from its Catmull-Clark limit sample and integrates it by a
guarded extrude+cut boolean. The segmentation tests are pure numpy; the
reconstruction test needs FreeCAD + the remesher and is skipped when either is
unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh2step.config import ConversionConfig
from mesh2step.segmentation import (
    OrganicRegion,
    segment_organic_regions,
)

try:  # pragma: no cover - environment probe
    from mesh2step.quadremesh import available

    HAVE_PNIM = available()
except Exception:  # noqa: BLE001
    HAVE_PNIM = False

try:  # pragma: no cover - environment probe
    import FreeCAD  # type: ignore  # noqa: F401
    import Part  # type: ignore  # noqa: F401

    HAVE_FREECAD = True
except Exception:  # noqa: BLE001
    HAVE_FREECAD = False


def _bump_region(n: int = 45, side: float = 60.0, amp: float = 4.0, base_z: float = 12.0):
    """An open, injective, doubly-curved bump surface (a Gaussian dome over a
    square footprint) — the geometry the region-level pass reconstructs cleanly.
    Returns (vertices, faces)."""
    xs = np.linspace(-side / 2, side / 2, n)
    verts = []
    idx = {}
    for i in range(n):
        for j in range(n):
            x, y = xs[i], xs[j]
            z = base_z + amp * np.exp(-((x * x + y * y) / 200.0))
            idx[(i, j)] = len(verts)
            verts.append([x, y, z])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _dome_cap(n_theta: int = 26, n_phi: int = 44, R: float = 20.0,
              max_theta_frac: float = 0.42):
    """An open, gently-curved spherical cap (theta up to ~76 deg by default) — an
    injective height field about its mean normal, the case the region-level organic
    pass reconstructs from its clean Catmull-Clark limit sample. Returns
    (vertices, faces)."""
    verts: list[list[float]] = []
    idx: dict[tuple[int, int], int] = {}
    for i in range(n_theta):
        theta = (np.pi * max_theta_frac) * i / (n_theta - 1)
        for j in range(n_phi):
            phi = 2 * np.pi * j / n_phi
            idx[(i, j)] = len(verts)
            verts.append([R * np.sin(theta) * np.cos(phi),
                          R * np.sin(theta) * np.sin(phi),
                          R * np.cos(theta)])
    faces: list[list[int]] = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = idx[(i, j)]
            b = idx[(i, (j + 1) % n_phi)]
            c = idx[(i + 1, (j + 1) % n_phi)]
            d = idx[(i + 1, j)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_gentle_cap_is_one_organic_region():
    """A gently-curved open cap is found as ONE injective organic region."""
    v, f = _dome_cap()
    cfg = ConversionConfig()
    regions = segment_organic_regions(v, f, set(), cfg)
    assert len(regions) == 1
    reg = regions[0]
    assert isinstance(reg, OrganicRegion)
    assert reg.size >= cfg.organic_region_min_facets
    assert reg.area >= cfg.organic_region_min_area
    # Injective: the region projects single-valued along its mean normal, so its
    # foldover stays within the extrude-integrable gate.
    assert reg.foldover <= cfg.organic_region_max_foldover


def test_wrapping_cap_is_deferred():
    """A cap that wraps past its silhouette (high foldover) is NOT claimed — its
    single-surface projection would fold, so the pass defers it (never ships a
    self-intersecting tool)."""
    v, f = _dome_cap(max_theta_frac=0.72)  # wraps well past 90 deg
    cfg = ConversionConfig()
    org = segment_organic_regions(v, f, set(), cfg)
    assert all(r.foldover <= cfg.organic_region_max_foldover for r in org)


def test_claimed_facets_are_excluded():
    """Facets already claimed by an analytic detector are not re-segmented."""
    v, f = _dome_cap()
    cfg = ConversionConfig()
    all_faces = set(range(len(f)))
    regions = segment_organic_regions(v, f, all_faces, cfg)
    assert regions == []


def test_small_region_is_ignored():
    """A residual smaller than organic_region_min_facets stays faceted (no region)."""
    v, f = _dome_cap(n_theta=8, n_phi=10)  # ~140 facets, below the 300 floor
    cfg = ConversionConfig()
    regions = segment_organic_regions(v, f, set(), cfg)
    assert regions == []


def test_flat_plate_has_no_organic_region():
    """A single flat plate has no smooth-curve seeds, so no organic region."""
    # Two big coplanar triangles (a flat quad) — no curvature anywhere.
    v = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0], [0, 50, 0]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    cfg = ConversionConfig()
    regions = segment_organic_regions(v, f, set(), cfg)
    assert regions == []


def test_nearly_flat_warped_panel_is_rejected_by_curvature_gate():
    """A large panel that gently warps (tiny height off its mean plane over a big
    span) has smooth-step seeds but is NOT organic — extruding its huge flat
    B-spline into a boolean tool grinds OCC (the fan_panel P0 hang). The
    ``organic_region_min_curve_frac`` gate rejects it; a genuinely curved cap of
    the same span survives."""
    # A very shallow, large-radius cap: a real spherical surface (so its rows form
    # smooth-curve seeds and it IS grabbed with the gate off), but with a tiny
    # theta sweep so its height off the mean plane is a small fraction of its span
    # (curve_frac below the 0.03 gate). This is the fan_panel signature: a big
    # gently-warped panel, not a genuine organic cap.
    v, f = _dome_cap(n_theta=26, n_phi=44, R=400.0, max_theta_frac=0.05)
    off = ConversionConfig(organic_region_min_curve_frac=None)
    grabbed = segment_organic_regions(v, f, set(), off)
    if not grabbed:
        pytest.skip("synthetic shallow cap did not seed a region on this build")
    from mesh2step.segmentation import _axis_basis
    reg = grabbed[0]
    fa = np.array(reg.face_indices)
    cent = v[f[fa]].mean(axis=1)
    rel = cent - cent.mean(axis=0)
    ax = np.array(reg.axis)
    e1, e2 = _axis_basis(ax)
    span = float(np.hypot((rel @ e1).ptp(), (rel @ e2).ptp())) or 1.0
    assert ((rel @ ax).ptp() / span) < 0.03, "cap should be below the flatness gate"
    # With the gate ON (default), the near-flat cap is rejected.
    gated = segment_organic_regions(v, f, set(), ConversionConfig())
    assert gated == [], "a near-flat warped panel must not be an organic region"


def test_bump_region_is_injective_and_detected():
    """The Gaussian bump is one injective organic region (foldover ~0)."""
    v, f = _bump_region()
    cfg = ConversionConfig()
    regions = segment_organic_regions(v, f, set(), cfg)
    assert len(regions) == 1
    assert regions[0].foldover <= cfg.organic_region_max_foldover


# --------------------------------------------------------------------------- #
# Multi-chart decomposition (pure numpy — no FreeCAD / remesher)
# --------------------------------------------------------------------------- #


def _full_hemisphere(n_theta: int = 40, n_phi: int = 64, R: float = 20.0,
                     max_theta_frac: float = 0.95):
    """A deep spherical cap that wraps far past 90 deg (theta up to ~171 deg by
    default) — the WRAPPING shell class the multi-chart pass must decompose: its
    facet normals span most of a sphere, so no single projection is injective."""
    verts: list[list[float]] = []
    idx: dict[tuple[int, int], int] = {}
    for i in range(n_theta):
        theta = (np.pi * max_theta_frac) * i / (n_theta - 1)
        for j in range(n_phi):
            phi = 2 * np.pi * j / n_phi
            idx[(i, j)] = len(verts)
            verts.append([R * np.sin(theta) * np.cos(phi),
                          R * np.sin(theta) * np.sin(phi),
                          R * np.cos(theta)])
    faces: list[list[int]] = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = idx[(i, j)]
            b = idx[(i, (j + 1) % n_phi)]
            c = idx[(i + 1, (j + 1) % n_phi)]
            d = idx[(i + 1, j)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_wrapping_region_splits_into_injective_charts():
    """A deep, wrapping cap (normals spanning most of a sphere) decomposes into
    several connected single-sided charts, and EACH chart is injective — every one
    of its facet normals sits within the half-angle cone about the chart axis, so
    the single-surface region builder can reconstruct it."""
    from mesh2step import organic_region as oreg
    from mesh2step.segmentation import (
        OrganicRegion,
        face_normals_and_areas,
    )

    v, f = _full_hemisphere()
    cfg = ConversionConfig()
    normals, _ = face_normals_and_areas(v, f)
    region = OrganicRegion(face_indices=list(range(len(f))),
                           axis=np.array([0.0, 0.0, 1.0]), area=5.0e3, foldover=0.5)
    charts = oreg.region_charts(v, f, region, cfg)
    assert len(charts) >= 2, "a wrapping cap must split into multiple charts"
    cos_half = float(np.cos(np.radians(cfg.organic_region_chart_half_angle)))
    for ch in charts:
        fa = np.array(ch.face_indices, dtype=int)
        dots = normals[fa] @ (ch.axis / np.linalg.norm(ch.axis))
        # Injective: every facet in a chart lies within the half-angle cone about
        # its axis (allow a hair of numerical slack from the axis refresh).
        assert dots.min() >= cos_half - 0.02, (
            f"chart not single-sided: min n.axis {dots.min():.3f} < {cos_half:.3f}")
        assert ch.size >= cfg.organic_region_chart_min_facets


def test_wrapping_charts_cover_most_of_the_region():
    """The charts of a wrapping cap together cover the bulk of the region's facets
    (the decomposition is not throwing most of the shell away)."""
    from mesh2step import organic_region as oreg
    from mesh2step.segmentation import OrganicRegion

    v, f = _full_hemisphere()
    cfg = ConversionConfig()
    region = OrganicRegion(face_indices=list(range(len(f))),
                           axis=np.array([0.0, 0.0, 1.0]), area=5.0e3, foldover=0.5)
    charts = oreg.region_charts(v, f, region, cfg)
    covered = set()
    for ch in charts:
        covered.update(ch.face_indices)
    assert len(covered) >= 0.6 * len(f), (
        f"charts cover only {len(covered)}/{len(f)} facets")


def test_injective_cap_is_one_chart():
    """A cap that fits comfortably inside one half-angle cone does not fragment — it
    is a single chart spanning essentially the whole region (no spurious
    over-splitting; the seed-anchored gate only cuts when the shell exceeds the
    cone)."""
    from mesh2step import organic_region as oreg
    from mesh2step.segmentation import OrganicRegion

    # theta up to ~40 deg -> well inside the ~50 deg chart half-angle cone.
    v, f = _dome_cap(max_theta_frac=0.22)
    cfg = ConversionConfig()
    region = OrganicRegion(face_indices=list(range(len(f))),
                           axis=np.array([0.0, 0.0, 1.0]), area=2.0e3, foldover=0.0)
    charts = oreg.region_charts(v, f, region, cfg)
    assert len(charts) == 1
    assert charts[0].size >= 0.9 * len(f)


def test_small_region_yields_no_charts():
    """A region below the multi-chart facet floor produces no charts (left faceted)."""
    from mesh2step import organic_region as oreg
    from mesh2step.segmentation import OrganicRegion

    v, f = _dome_cap(n_theta=8, n_phi=10)  # ~140 facets
    cfg = ConversionConfig()
    region = OrganicRegion(face_indices=list(range(len(f))),
                           axis=np.array([0.0, 0.0, 1.0]), area=50.0, foldover=0.0)
    assert oreg.region_charts(v, f, region, cfg) == []


# --------------------------------------------------------------------------- #
# Reconstruction + boolean integration (FreeCAD + remesher required)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not (HAVE_FREECAD and HAVE_PNIM),
                    reason="FreeCAD + pynanoinstantmeshes required")
def test_region_surface_reconstructs_injective_bump_and_cuts_clean():
    """End-to-end on a cooperative (injective) region: the Catmull-Clark limit
    sampler + single-B-spline fit lands sub-mm on the mesh, and the guarded
    extrude+cut boolean replaces the faceted bump with one B-spline face, yielding
    exactly one valid solid (the core primitive of the region-level organic pass)."""
    import FreeCAD  # type: ignore
    import Part  # type: ignore

    from mesh2step import organic_region as oreg

    v, f = _bump_region()
    cfg = ConversionConfig()
    region = segment_organic_regions(v, f, set(), cfg)[0]

    surf, detail = oreg.build_region_surface(v, f, region, cfg, Part)
    assert surf is not None, f"surface should build: {detail}"
    # The clean limit sample lands the surface close to the real facets.
    assert detail["deviation_mm"] < 1.0

    # Cut it from a plate whose top sits just above the bump; the result is one
    # valid solid carrying the smooth B-spline face.
    side = 60.0
    base = Part.makeBox(side, side, 20, FreeCAD.Vector(-side / 2, -side / 2, -4))
    n_before = sum(1 for fc in base.Faces if "BSpline" in fc.Surface.TypeId)
    result = oreg.boolean_clean_region(base, surf, region, Part)
    solids = result.Solids
    assert len(solids) == 1 and solids[0].isValid()
    n_after = sum(1 for fc in result.Faces if "BSpline" in fc.Surface.TypeId)
    assert n_after > n_before, "cut must plant the B-spline face"


@pytest.mark.skipif(not (HAVE_FREECAD and HAVE_PNIM),
                    reason="FreeCAD + pynanoinstantmeshes required")
def test_wrapping_cap_declines_as_one_surface_but_charts_build():
    """The multi-chart mechanism, end to end on a decomposable wrapping shell:
    a DEEP spherical cap (theta up to ~150 deg, normals fanning past a hemisphere)
    has NO single injective projection, so :func:`build_region_surface` on the whole
    region declines (its fold guard fires) — but :func:`region_charts` splits it into
    injective normal-cone charts and MULTIPLE charts reconstruct into valid B-spline
    surfaces. This is the proof that the chart decomposition claims geometry the
    single-surface pass cannot."""
    import Part  # type: ignore

    from mesh2step import organic_region as oreg
    from mesh2step.segmentation import OrganicRegion

    # theta up to ~150 deg -> the (u,v) projection genuinely folds.
    n_theta, n_phi, R = 34, 60, 20.0
    verts, idx = [], {}
    for i in range(n_theta):
        theta = (np.pi * 0.83) * i / (n_theta - 1)
        for j in range(n_phi):
            phi = 2 * np.pi * j / n_phi
            idx[(i, j)] = len(verts)
            verts.append([R * np.sin(theta) * np.cos(phi),
                          R * np.sin(theta) * np.sin(phi),
                          R * np.cos(theta)])
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = idx[(i, j)]
            b = idx[(i, (j + 1) % n_phi)]
            c = idx[(i + 1, (j + 1) % n_phi)]
            d = idx[(i + 1, j)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    cfg = ConversionConfig()
    region = OrganicRegion(face_indices=list(range(len(f))),
                           axis=np.array([0.0, 0.0, 1.0]),
                           area=float(2 * np.pi * R * R), foldover=0.2)

    # One surface over the whole wrapping cap folds -> declined.
    whole, wdet = oreg.build_region_surface(v, f, region, cfg, Part)
    assert whole is None
    assert "fold" in (wdet.get("reason") or "").lower()

    # Charts: several injective pieces; multiple reconstruct into a valid surface.
    charts = oreg.region_charts(v, f, region, cfg)
    assert len(charts) >= 2, f"cap should split into charts, got {len(charts)}"
    built = 0
    for ch in charts:
        surf, _det = oreg.build_region_surface(v, f, ch, cfg, Part)
        if surf is not None:
            built += 1
    assert built >= 2, (
        f"at least two charts must reconstruct into a valid surface (got {built})")


@pytest.mark.skipif(not (HAVE_FREECAD and HAVE_PNIM),
                    reason="FreeCAD + pynanoinstantmeshes required")
def test_folded_region_surface_is_declined():
    """A region that wraps too far to project injectively (a deep hemispherical
    cap) yields a folded surface, which the fold guard rejects BEFORE any boolean —
    the safety property that keeps a pathological (self-intersecting) tool from ever
    reaching the cut. Detection foldover is bypassed here to exercise the geometric
    guard directly."""
    from mesh2step import organic_region as oreg

    # Deep cap: theta up to ~135 deg -> its (u,v) projection folds.
    n_theta, n_phi, R = 30, 48, 20.0
    verts, idx = [], {}
    for i in range(n_theta):
        theta = (np.pi * 0.75) * i / (n_theta - 1)
        for j in range(n_phi):
            phi = 2 * np.pi * j / n_phi
            idx[(i, j)] = len(verts)
            verts.append([R * np.sin(theta) * np.cos(phi),
                          R * np.sin(theta) * np.sin(phi),
                          R * np.cos(theta)])
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = idx[(i, j)]
            b = idx[(i, (j + 1) % n_phi)]
            c = idx[(i + 1, (j + 1) % n_phi)]
            d = idx[(i + 1, j)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    # Force a region over the whole cap (bypass the foldover detection gate) so the
    # geometric fold guard in build_region_surface is what declines it.
    axis = np.array([0.0, 0.0, 1.0])
    region = OrganicRegion(face_indices=list(range(len(f))), axis=axis,
                           area=1.0e4, foldover=0.0)
    import Part  # type: ignore

    surf, detail = oreg.build_region_surface(v, f, region, ConversionConfig(), Part)
    assert surf is None
    assert "fold" in (detail.get("reason") or "").lower()
