"""Regression tests for the sphere op budget and the resolution-scaled
planar-merge tolerances (patton user-reported fixes).

Pure-Python guard logic (budget skip, tolerance scaling) — no FreeCAD required.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mesh2step.config import ConversionConfig
from mesh2step.segmentation import planar_merge_tols


# --------------------------------------------------------------------------- #
# Resolution-scaled planar-merge tolerances
# --------------------------------------------------------------------------- #
def _noisy_flat(n=20, noise_deg=1.7, seed=0):
    """A grid-triangulated flat whose facet normals scatter by ~noise_deg — the
    coarse-scan signature that fragments a flat under the strict 1° tolerance."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
    z = np.zeros_like(xs)
    # jitter z by an amount that yields ~noise_deg facet tilt over a cell
    cell = 10.0 / (n - 1)
    z += rng.normal(0, np.tan(np.radians(noise_deg)) * cell * 0.5, z.shape)
    verts = np.stack([xs.ravel(), ys.ravel(), z.ravel()], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return verts.astype(float), np.asarray(faces, dtype=int)


def test_planar_merge_tols_default_is_strict_absolute():
    """With the rel factors 0 (default) the effective tolerances are exactly the
    absolute config values — legacy behaviour, no resolution scaling."""
    v, f = _noisy_flat()
    cfg = ConversionConfig()  # planar_*_tol_rel default to 0.0
    cos_tol, dist_tol = planar_merge_tols(v, f, cfg)
    assert dist_tol == pytest.approx(cfg.dist_tol)
    assert cos_tol == pytest.approx(cfg.angle_tol_cos)


def test_planar_merge_tols_scales_when_opted_in():
    """Opting in (rel > 0) loosens both tolerances above the absolute floor, but
    the angle stays clamped by the conservative cap so curved detectors are safe."""
    v, f = _noisy_flat()
    cfg = dataclasses.replace(
        ConversionConfig(),
        planar_angle_tol_rel=1.0,
        planar_dist_tol_rel=0.05,
    )
    cos_tol, dist_tol = planar_merge_tols(v, f, cfg)
    # effective dist grows above the 0.01 floor (median edge ~0.7mm here)
    assert dist_tol > cfg.dist_tol
    assert dist_tol <= cfg.planar_dist_tol_cap
    # effective angle grows above 1° but never past the cap
    eff_angle = np.degrees(np.arccos(min(1.0, cos_tol)))
    assert eff_angle >= cfg.angle_tol_deg
    assert eff_angle <= cfg.planar_angle_tol_cap_deg + 1e-9


def test_planar_merge_tols_caps_are_hard():
    """Even with huge rel factors the effective tolerances never exceed the caps."""
    v, f = _noisy_flat()
    cfg = dataclasses.replace(
        ConversionConfig(),
        planar_angle_tol_rel=100.0,
        planar_dist_tol_rel=100.0,
    )
    cos_tol, dist_tol = planar_merge_tols(v, f, cfg)
    assert dist_tol <= cfg.planar_dist_tol_cap + 1e-12
    eff_angle = np.degrees(np.arccos(min(1.0, cos_tol)))
    assert eff_angle <= cfg.planar_angle_tol_cap_deg + 1e-9


# --------------------------------------------------------------------------- #
# Sphere op budget (pure-Python skip logic via a lightweight stand-in)
# --------------------------------------------------------------------------- #
class _FakeSolid:
    def __init__(self, n_faces):
        self.Faces = list(range(n_faces))
        self.Solids = [self]

    def isValid(self):
        return True


def test_sphere_op_budget_skips_on_dense_base(monkeypatch):
    """On a dense base (spheres × faces over budget) the sphere ops are skipped
    wholesale — the graceful degradation that stops the M3 pass hanging — and no
    _boolean_clean_sphere op is ever attempted."""
    from mesh2step import builder

    called = {"n": 0}

    def _boom(*a, **k):  # must never be reached when the budget trips
        called["n"] += 1
        raise AssertionError("sphere op attempted despite budget")

    monkeypatch.setattr(builder, "_boolean_clean_sphere", _boom)

    dense = _FakeSolid(200_000)
    spheres = [object() for _ in range(8)]  # 8 × 200k = 1.6M > 1.5M budget
    cfg = ConversionConfig()  # sphere_op_budget default 1.5M
    msgs = []
    out, built = builder._apply_sphere_ball_ops(
        dense, spheres, Part=None, progress=msgs.append,
        bbox_guard=cfg.boolean_max_bbox_growth, config=cfg)
    assert built == 0
    assert out is dense
    assert called["n"] == 0
    assert any("sphere ops skipped" in m for m in msgs)


def test_sphere_op_budget_allows_normal_base(monkeypatch):
    """Below budget, ops run through _try_boolean_step normally (here each op is a
    no-op identity, so all 'succeed')."""
    from mesh2step import builder

    monkeypatch.setattr(builder, "_boolean_clean_sphere",
                        lambda solid, sph, Part, **k: solid)
    # _try_boolean_step needs a valid single-solid candidate + bbox reads; stub it
    monkeypatch.setattr(builder, "_try_boolean_step",
                        lambda cur, fn, **k: (fn(cur), True))

    base = _FakeSolid(12_000)
    spheres = [object() for _ in range(9)]  # 9 × 12k = 108k << 1.5M
    cfg = ConversionConfig()
    out, built = builder._apply_sphere_ball_ops(
        base, spheres, Part=None, progress=lambda _m: None,
        bbox_guard=cfg.boolean_max_bbox_growth, config=cfg)
    assert built == 9


def test_sphere_rtaf_gate_reverts_regression_keeps_flat(monkeypatch):
    """The sphere RTAF gate (task §2) is a REGRESSION gate: a cap that WORSENS the
    RTAF is reverted, but one that leaves it flat (a legitimate low-coverage dome
    whose area is tiny beside the part's dominant surface) is KEPT. Requiring
    strict improvement would wrongly drop real shallow caps."""
    from mesh2step import builder

    monkeypatch.setattr(builder, "_boolean_clean_sphere",
                        lambda solid, sph, Part, **k: solid)
    monkeypatch.setattr(builder, "_try_boolean_step",
                        lambda cur, fn, **k: (fn(cur), True))
    # First op keeps RTAF flat (0.30 -> 0.30) => kept; second regresses
    # (0.30 -> 0.40) => reverted. compute_rtaf is called before the batch and
    # after each successful op.
    rtaf_seq = iter([0.30, 0.30, 0.40])
    monkeypatch.setattr(builder, "compute_rtaf",
                        lambda solid, cfg: {"rtaf": next(rtaf_seq)})

    base = _FakeSolid(1_000)
    spheres = [object(), object()]
    cfg = ConversionConfig()
    _out, built = builder._apply_sphere_ball_ops(
        base, spheres, Part=None, progress=lambda _m: None,
        bbox_guard=cfg.boolean_max_bbox_growth, config=cfg)
    assert built == 1  # flat kept, regression reverted


def test_new_freeform_and_sphere_config_defaults():
    """Lock the task's new defaults so an accidental edit is caught."""
    cfg = ConversionConfig()
    assert cfg.freeform_inpaint is True
    assert cfg.freeform_max_missing == 0.55
    assert cfg.freeform_claim_max_missing == 0.30
    assert cfg.freeform_max_split_depth == 2
    assert cfg.sphere_rtaf_gate is True
    # P0/P1 hardening defaults (regression net): freeform pre-emption of the swept
    # pool is ON but gated by buildability + a swept-arc veto; the organic passes
    # are budgeted; the region flatness gate is on.
    assert cfg.freeform_claim_swept_pool is True
    assert cfg.freeform_claim_requires_buildable is True
    assert cfg.freeform_claim_max_swept_arc_frac == 0.15
    assert cfg.organic_pass_time_budget == 120.0
    assert cfg.organic_remesh_timeout == 45.0
    assert cfg.organic_boolean_timeout == 90.0
    assert cfg.organic_boolean_isolate_min_base_faces == 3000
    assert cfg.organic_multipatch_max_swept_walls == 6
    assert cfg.organic_multipatch_max_subdiv_quads == 2500
    assert cfg.organic_region_min_curve_frac == 0.03
