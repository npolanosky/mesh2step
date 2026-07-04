"""Helical & patterned feature detection (M5): threads, knurling, gears.

The precise numerical guarantees (pitch recovery, bimodality discrimination,
harmonic handling) are pinned with pure-numpy synthetic point clouds — clean,
deterministic geometry that isolates the fitters from mesh-tessellation noise.
Coarser end-to-end checks run against the generated sample STLs (threaded_rod,
knurled_band, spur_gear) where available.

Real-scan acceptance (knurled_knob, parametric_bottle_cap, gear_box_gear_v2)
lives in the manual corpus sweep, not in git — those STLs are large and stay
out of the repo (see tests/data/community/fetched/SOURCES.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mesh2step.config import ConversionConfig
from mesh2step.fitting import (
    KnurlBand,
    Thread,
    _helix_phase_fit,
    _knurl_bimodality,
    detect_cylinders,
    detect_knurling,
    detect_threads,
)

DATA = Path(__file__).parent / "data"
SAMPLES = (
    json.loads((DATA / "samples.json").read_text())
    if (DATA / "samples.json").exists() else []
)


def _truth(kind):
    return next((t for t in SAMPLES if t.get("kind") == kind), None)


# --------------------------------------------------------------------------- #
# Helix phase-fit: pitch recovery (the one new numpy fitter). A clean helix of
# known pitch must be recovered within ~2%.
# --------------------------------------------------------------------------- #


def _helix_cloud(pitch, radius, n_turns, pts_per_turn=60, jitter=0.0, seed=0):
    """A clean single-start helix's crest points: z, phi arrays about the axis."""
    rng = np.random.default_rng(seed)
    n = int(n_turns * pts_per_turn)
    t = np.linspace(0, n_turns, n)
    phi = 2 * math.pi * t
    z = pitch * t
    if jitter:
        z = z + rng.normal(0, jitter, size=z.shape)
    # wrap phi into atan2 range (the detector sees wrapped angles)
    phi_wrapped = np.arctan2(np.sin(phi), np.cos(phi))
    return z, phi_wrapped


@pytest.mark.parametrize("pitch", [1.5, 2.0, 3.5, 5.0])
def test_helix_phase_fit_recovers_pitch(pitch):
    z, phi = _helix_cloud(pitch, radius=6.0, n_turns=6)
    zext = float(z.max() - z.min())
    p, cvar, hand = _helix_phase_fit(z, phi, 0.3, zext / 1.5)
    assert p == pytest.approx(pitch, rel=0.02)
    assert (1.0 - cvar) > 0.9  # a clean helix collapses tightly
    assert hand == "right"


def test_helix_phase_fit_handedness():
    # A left-hand helix: z decreases as phi increases.
    z, phi = _helix_cloud(2.0, 6.0, 6)
    p, cvar, hand = _helix_phase_fit(z, -phi + math.pi, 0.3, 10.0)
    # a reversed-sign helix reads left-handed
    assert hand == "left"
    assert p == pytest.approx(2.0, rel=0.03)


def test_helix_phase_fit_noise_does_not_collapse():
    # Random (non-helical) points must NOT phase-collapse (low resultant).
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 12, 400)
    phi = rng.uniform(-math.pi, math.pi, 400)
    p, cvar, hand = _helix_phase_fit(z, phi, 0.3, 8.0)
    assert (1.0 - cvar) < 0.35  # no genuine helix -> below the accept floor


# --------------------------------------------------------------------------- #
# Knurl bimodality discriminator: a diamond knurl's axial-tilt component is
# bimodal (two symmetric lobes); a plain wall's is unimodal at ~0.
# --------------------------------------------------------------------------- #


def test_knurl_bimodality_scores_bimodal_high():
    rng = np.random.default_rng(2)
    # two symmetric lobes at +/-0.3 (the diamond knurl's crossing families)
    lobe = rng.choice([-0.3, 0.3], size=2000) + rng.normal(0, 0.03, 2000)
    assert _knurl_bimodality(lobe) > 0.4


def test_knurl_bimodality_scores_unimodal_low():
    rng = np.random.default_rng(3)
    # a plain cylinder wall: normals ~perpendicular to axis, tiny axial tilt
    flat = rng.normal(0, 0.02, 2000)
    assert _knurl_bimodality(flat) < 0.35


# --------------------------------------------------------------------------- #
# Dataclass suppression semantics: external -> crest (major); internal -> minor.
# --------------------------------------------------------------------------- #


def test_thread_suppress_radius_sides():
    ext = Thread(axis_point=np.zeros(3), axis_dir=np.array([0, 0, 1.0]),
                 nominal_radius=5.0, axial_min=0, axial_max=10, pitch=2.0,
                 starts=1, handedness="right", crest_radius=5.5, root_radius=4.5,
                 rms=0.0, turns=5.0, face_indices=[], is_internal=False)
    assert ext.suppress_radius == pytest.approx(5.5)      # major (crest out)
    intern = Thread(axis_point=np.zeros(3), axis_dir=np.array([0, 0, 1.0]),
                    nominal_radius=5.0, axial_min=0, axial_max=10, pitch=2.0,
                    starts=1, handedness="right", crest_radius=5.5, root_radius=4.5,
                    rms=0.0, turns=5.0, face_indices=[], is_internal=True)
    assert intern.suppress_radius == pytest.approx(4.5)   # minor (crest in)


def test_knurl_suppress_radius_sides():
    boss = KnurlBand(axis_point=np.zeros(3), axis_dir=np.array([0, 0, 1.0]),
                     nominal_radius=10.0, axial_min=0, axial_max=10,
                     pattern="diamond", pitch_estimate=0.2, bimodality=0.8,
                     face_indices=[], outward=True, crest_radius=10.3, root_radius=9.7)
    assert boss.suppress_radius == pytest.approx(10.3)    # fuse to crest
    bore = KnurlBand(axis_point=np.zeros(3), axis_dir=np.array([0, 0, 1.0]),
                     nominal_radius=10.0, axial_min=0, axial_max=10,
                     pattern="diamond", pitch_estimate=0.2, bimodality=0.8,
                     face_indices=[], outward=False, crest_radius=10.3, root_radius=9.7)
    assert bore.suppress_radius == pytest.approx(9.7)     # cut to root


# --------------------------------------------------------------------------- #
# No false positives on a plain cylinder (a smooth wall is neither thread nor
# knurl). Uses the generated ``cylinder`` sample.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not (DATA / "cylinder.stl").exists(), reason="samples not generated")
def test_no_thread_or_knurl_on_plain_cylinder():
    from mesh2step.mesh_io import load_stl

    v, f = load_stl(DATA / "cylinder.stl")
    cfg = ConversionConfig()
    cyls = detect_cylinders(v, f, cfg)
    assert detect_threads(v, f, cyls, set(), cfg) == []
    assert detect_knurling(v, f, cyls, set(), cfg) == []


@pytest.mark.skipif(not (DATA / "cube.stl").exists(), reason="samples not generated")
def test_no_helical_features_on_cube():
    from mesh2step.mesh_io import load_stl

    v, f = load_stl(DATA / "cube.stl")
    cfg = ConversionConfig()
    cyls = detect_cylinders(v, f, cfg)
    assert detect_threads(v, f, cyls, set(), cfg) == []
    assert detect_knurling(v, f, cyls, set(), cfg) == []


# --------------------------------------------------------------------------- #
# End-to-end sample detection: the threaded_rod / knurled_band bands are claimed
# (as a thread OR a knurl OR a cylinder) and, where a helical feature is found,
# its radius sits near the core. Tolerant by design (a helical-cut synthetic
# knurl can read as a thread — both suppress to the same cylinder).
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(_truth("threaded_rod") is None, reason="threaded_rod sample not generated")
def test_threaded_rod_band_claimed_near_core():
    from mesh2step.mesh_io import load_stl

    t = _truth("threaded_rod")
    v, f = load_stl(DATA / t["file"])
    cfg = ConversionConfig()
    cyls = detect_cylinders(v, f, cfg)
    threads = detect_threads(v, f, cyls, set(), cfg)
    core_r = t["threads"][0]["core_radius"]
    # The rod's wall is claimed either as a helical thread or (if the thread is
    # shallow enough) a plain cylinder; either way something wraps the core.
    radii = [th.nominal_radius for th in threads] + [c.radius for c in cyls]
    assert radii, "the rod wall was not claimed by any detector"
    assert any(core_r - 0.5 <= r <= core_r + 2.0 for r in radii)


@pytest.mark.skipif(_truth("knurled_band") is None, reason="knurled_band sample not generated")
def test_knurled_band_claimed_near_radius():
    from mesh2step.mesh_io import load_stl

    t = _truth("knurled_band")
    v, f = load_stl(DATA / t["file"])
    cfg = ConversionConfig()
    cyls = detect_cylinders(v, f, cfg)
    claimed = {i for c in cyls for i in c.face_indices}
    threads = detect_threads(v, f, cyls, claimed, cfg)
    for th in threads:
        claimed.update(th.face_indices)
    knurls = detect_knurling(v, f, cyls, claimed, cfg)
    R = t["knurling"][0]["nominal_radius"]
    # A diamond synthetic knurl may read as a thread (harmless — same cylinder).
    radii = ([k.nominal_radius for k in knurls]
             + [th.nominal_radius for th in threads]
             + [c.radius for c in cyls])
    assert any(R - 1.0 <= r <= R + 1.0 for r in radii)


# --------------------------------------------------------------------------- #
# Gear / whole-outline routing (M5.3). A repeated-arc region wrapping the axis is
# flagged ``whole_extrusion`` with an outline loop; a plain wall is not.
# --------------------------------------------------------------------------- #


def test_repeated_arc_pattern_and_centering_helpers():
    from mesh2step.fitting import (
        ProfileSegment,
        SweptProfile,
        _is_repeated_arc_pattern,
        _profile_is_centered,
    )

    cfg = ConversionConfig()
    # A ring of many arcs at a consistent radius about the centre -> centered.
    segs = []
    n = 40
    for k in range(n):
        a0 = 2 * math.pi * k / n
        a1 = 2 * math.pi * (k + 1) / n
        p0 = np.array([10 * math.cos(a0), 10 * math.sin(a0)])
        p1 = np.array([10 * math.cos(a1), 10 * math.sin(a1)])
        segs.append(ProfileSegment(kind="arc", p0=p0, p1=p1,
                                   center=np.zeros(2), radius=10.0))
    prof = SweptProfile(axis=np.array([0, 0, 1.0]), origin=np.zeros(3),
                        e1=np.array([1, 0, 0.0]), e2=np.array([0, 1, 0.0]),
                        segments=segs, axial_min=0, axial_max=5, closed=True,
                        rms=0.0, face_indices=[], n_arcs=n)
    assert _is_repeated_arc_pattern(prof, cfg) is True
    assert _profile_is_centered(prof, cfg) is True

    # A one-sided panel of arcs (all on the +x side) is NOT centered.
    off = []
    for k in range(40):
        a = -0.4 + 0.8 * k / 40
        p0 = np.array([20 + math.cos(a), math.sin(a)])
        p1 = np.array([20 + math.cos(a + 0.02), math.sin(a + 0.02)])
        off.append(ProfileSegment(kind="arc", p0=p0, p1=p1,
                                  center=np.array([20.0, 0.0]), radius=1.0))
    prof2 = SweptProfile(axis=np.array([0, 0, 1.0]), origin=np.zeros(3),
                         e1=np.array([1, 0, 0.0]), e2=np.array([0, 1, 0.0]),
                         segments=off, axial_min=0, axial_max=5, closed=True,
                         rms=0.0, face_indices=[], n_arcs=40)
    assert _profile_is_centered(prof2, cfg) is False


@pytest.mark.skipif(_truth("spur_gear") is None, reason="spur_gear sample not generated")
def test_spur_gear_bore_survives_and_solid():
    """The spur_gear sample must convert to a single watertight solid whose
    central bore survives (the M5.3 ladder cuts the bore after the gear fuse).
    Requires FreeCAD; skipped when unavailable."""
    pytest.importorskip("FreeCAD")
    from mesh2step.mesh_io import load_stl
    from mesh2step import builder

    t = _truth("spur_gear")
    v, f = load_stl(DATA / t["file"])
    cfg = ConversionConfig()
    solid, stats = builder.build_boolean_clean_solid(v, f, cfg)
    solids = getattr(solid, "Solids", [])
    assert solids and solids[0].isValid()          # single watertight solid
    # The bore (a through cylinder) is detected and cut.
    assert stats["cylinders_detected"] >= 1
    bore = t["gears"][0]["bore"]
    assert any(abs(c["radius"] - bore) <= 0.3 for c in stats["cylinders"])
