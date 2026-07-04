"""Pure-numpy tests for the pipeline's bbox-ceiling gate (P0-1b) and the
self-intersection-resolve logging (P1-2). No FreeCAD required — the gate's
decision maths and the resolve reason strings are exercised via a lightweight
fake shape and a synthetic mesh.
"""

from __future__ import annotations

import numpy as np

from mesh2step.config import ConversionConfig
from mesh2step.pipeline import _bbox_delta
from mesh2step.segmentation import planar_coverage


class _FakeBBox:
    def __init__(self, x, y, z):
        self.XLength, self.YLength, self.ZLength = x, y, z


class _FakeShape:
    def __init__(self, x, y, z):
        self.BoundBox = _FakeBBox(x, y, z)


# input_dims are sorted (desc) side lengths, as the pipeline computes them.
def test_bbox_delta_zero_for_matching_box():
    dims = [210.0, 126.0, 12.0]
    assert _bbox_delta(_FakeShape(210.0, 126.0, 12.0), dims) < 1e-9


def test_bbox_delta_flags_the_base_lid_collapse():
    # base_lid: 210x126x12 mesh, output collapsed to a ~6mm cube.
    dims = [210.0, 126.0, 12.0]
    delta = _bbox_delta(_FakeShape(6.5, 6.5, 5.6), dims)
    # Dominant axis 210 -> 6.5 is ~97% off; well past the 25% ceiling.
    assert delta > 0.9


def test_bbox_delta_gate_ceiling_admits_legit_drift():
    # carabiner's ~16% Z drift is the worst legitimate case; the 25% default
    # ceiling must NOT reject it.
    cfg = ConversionConfig()
    dims = [76.487, 37.0, 5.0]
    delta = _bbox_delta(_FakeShape(76.487, 37.0, 5.782), dims)
    assert delta < cfg.bbox_reject_delta  # 0.156 < 0.25 -> not gated


def test_bbox_delta_gate_ceiling_rejects_collapse():
    cfg = ConversionConfig()
    dims = [210.0, 126.0, 12.0]
    delta = _bbox_delta(_FakeShape(6.5, 6.5, 5.6), dims)
    assert delta > cfg.bbox_reject_delta  # gated


def test_bbox_reject_delta_default_is_sane():
    # Between the worst legit drift (~16%) and any real collapse.
    cfg = ConversionConfig()
    assert cfg.bbox_reject_delta is not None
    assert 0.16 < cfg.bbox_reject_delta < 0.9


# --- Planarity-damage back-off gate (task §1): the ratio-vs-threshold decision -

def _grid(n, jitter, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 60.0, n + 1)
    gx, gy = np.meshgrid(xs, xs)
    gz = np.zeros_like(gx)
    if jitter > 0:
        gz[1:-1, 1:-1] = rng.uniform(-jitter, jitter, size=gz[1:-1, 1:-1].shape)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(float)
    faces = []
    for r in range(n):
        for c in range(n):
            a, b, d, e = (r * (n + 1) + c, r * (n + 1) + c + 1,
                          (r + 1) * (n + 1) + c, (r + 1) * (n + 1) + c + 1)
            faces += [[a, b, e], [a, e, d]]
    return verts, np.asarray(faces, dtype=np.int64)


def test_planarity_damage_default_threshold_is_sane():
    cfg = ConversionConfig()
    assert cfg.planarity_damage_check is True
    # Above the Patton 12k damaged ratio (~0.64-0.67, rejected) and below both the
    # Patton 24k rung (~0.74-0.84) and every prismatic part's 12k ratio (>=0.87),
    # which must be kept — so the gate rejects only the flat-destroying rung.
    assert cfg.planarity_damage_min_ratio is not None
    assert 0.67 < cfg.planarity_damage_min_ratio < 0.74


def test_planarity_ratio_classifies_damaged_vs_clean():
    # Reproduce the pipeline's rung decision on synthetic geometry: a clean rung
    # (ratio ~1) passes the gate; a warped rung (ratio well below) is rejected.
    cfg = ConversionConfig()
    mrf = cfg.planarity_min_region_facets
    raw = planar_coverage(*_grid(14, 0.0), config=cfg, min_region_facets=mrf)["coverage"]
    clean = planar_coverage(*_grid(14, 0.0), config=cfg,
                            min_region_facets=mrf)["coverage"]
    warped = planar_coverage(*_grid(14, 0.6, seed=1), config=cfg,
                             min_region_facets=mrf)["coverage"]
    assert raw > 0
    assert (clean / raw) >= cfg.planarity_damage_min_ratio      # clean rung kept
    assert (warped / raw) < cfg.planarity_damage_min_ratio      # damaged rung backed off


# --- P1-2: resolve_self_intersections logs a distinct reason on each failure --

def test_resolve_logs_import_failure(monkeypatch):
    """When manifold3d is unavailable the reason names an environment issue, not
    a geometry rejection."""
    import builtins

    import mesh2step.meshprep as mp

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "manifold3d":
            raise ImportError("no manifold3d")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    msgs: list[str] = []
    v = np.zeros((3, 3))
    f = np.array([[0, 1, 2]])
    out = mp.resolve_self_intersections(v, f, on_progress=msgs.append)
    assert out is None
    assert any("not installed" in m and "environment" in m for m in msgs)
