"""Cylinder detection tests against ground-truth sample parts (numpy only).

Requires the generated samples under tests/data (scripts/generate_samples.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh2step.config import ConversionConfig
from mesh2step.fitting import detect_cylinders
from mesh2step.mesh_io import load_stl

DATA = Path(__file__).parent / "data"
SAMPLES = json.loads((DATA / "samples.json").read_text()) if (DATA / "samples.json").exists() else []


@pytest.mark.skipif(not SAMPLES, reason="samples not generated")
@pytest.mark.parametrize("truth", SAMPLES, ids=lambda t: t["file"])
def test_detected_radii_match_ground_truth(truth):
    vertices, faces = load_stl(DATA / truth["file"])
    cylinders = detect_cylinders(vertices, faces, ConversionConfig())

    found = sorted(round(c.radius, 2) for c in cylinders)
    expected = sorted(c["radius"] for c in truth["cylinders"])
    assert found == pytest.approx(expected, abs=0.05)


@pytest.mark.skipif(not SAMPLES, reason="samples not generated")
def test_no_false_cylinders_on_prismatic_parts():
    for name in ("cube", "l_bracket"):
        vertices, faces = load_stl(DATA / f"{name}.stl")
        assert detect_cylinders(vertices, faces, ConversionConfig()) == []


@pytest.mark.skipif(not SAMPLES, reason="samples not generated")
def test_hole_vs_boss_classification():
    vertices, faces = load_stl(DATA / "flanged_pipe.stl")
    cyls = {round(c.radius): c for c in detect_cylinders(vertices, faces, ConversionConfig())}
    assert cyls[15].outward is True   # outer wall = boss
    assert cyls[9].outward is False   # central bore = hole


@pytest.mark.skipif(not SAMPLES, reason="samples not generated")
def test_countersink_cone_detected():
    from mesh2step.fitting import detect_cones

    vertices, faces = load_stl(DATA / "countersink_plate.stl")
    cfg = ConversionConfig()
    cyls = detect_cylinders(vertices, faces, cfg)
    cones = detect_cones(vertices, faces, cyls, cfg)
    assert len(cones) == 1
    assert cones[0].half_angle_deg == pytest.approx(45.0, abs=2.0)


@pytest.mark.skipif(not SAMPLES, reason="samples not generated")
def test_no_false_cones_on_plain_holes():
    from mesh2step.fitting import detect_cones

    for name in ("plate_with_holes", "flanged_pipe", "cylinder"):
        vertices, faces = load_stl(DATA / f"{name}.stl")
        cyls = detect_cylinders(vertices, faces, ConversionConfig())
        assert detect_cones(vertices, faces, cyls, ConversionConfig()) == []


@pytest.mark.skipif(not SAMPLES, reason="samples not generated")
def test_straight_fillet_detected_at_ground_truth_radius():
    """The fillet_chamfer_plate has one R=3 filleted edge and one 45deg chamfer.

    The fillet must be recovered as a partial-arc analytic cylinder with a
    tangency-derived radius within ~1% of 3.0, and the chamfer must NOT be
    mistaken for a fillet (it stays a plane).
    """
    from mesh2step.fitting import detect_fillets_straight
    from mesh2step.segmentation import (
        mesh_resolution,
        segment_planar,
        segment_smooth_bands,
    )

    truth = next((t for t in SAMPLES if t["file"] == "fillet_chamfer_plate.stl"), None)
    if truth is None:
        pytest.skip("fillet_chamfer_plate sample not generated")

    vertices, faces = load_stl(DATA / truth["file"])
    cfg = ConversionConfig()
    resolution = mesh_resolution(vertices, faces, cfg)
    cylinders = detect_cylinders(vertices, faces, cfg)
    claimed: set[int] = set()
    for c in cylinders:
        claimed.update(c.face_indices)
    regions = segment_planar(vertices, faces, cfg)
    bands = segment_smooth_bands(vertices, faces, claimed, regions, cfg)
    fillets = detect_fillets_straight(
        vertices, faces, bands, regions, claimed, cfg, resolution)

    expected_r = truth["fillets"][0]["radius"]
    assert len(fillets) == 1, f"expected exactly one fillet, got {len(fillets)}"
    fl = fillets[0]
    # Radius within ~1% of ground truth, derived from the tangency constraint.
    assert fl.radius == pytest.approx(expected_r, rel=0.01)
    assert fl.radius_source == "tangency"
    assert fl.tangent is True
    assert fl.is_fillet is True
    # A fillet is a partial arc, not a full cylinder.
    assert 0.0 < fl.coverage < 0.6
    # The chamfer (a single planar strip) must not be picked up as a fillet.
    assert all(f.radius_source in ("tangency", "fit") for f in fillets)


def test_no_fillets_on_plain_and_chamfered_prisms():
    """Fillet detection must not fire on a plain box (no rounded edges). Uses a
    synthetic grid-tessellated box built without FreeCAD."""
    import numpy as np

    from mesh2step.fitting import detect_fillets_straight
    from mesh2step.segmentation import (
        mesh_resolution,
        segment_planar,
        segment_smooth_bands,
    )

    # Grid-tessellated axis-aligned box (many small flat facets, sharp edges).
    def grid_box(L=40.0, W=20.0, H=10.0, step=2.0):
        tris = []

        def quad(a, b, c, d):
            tris.append([a, b, c])
            tris.append([a, c, d])

        def gq(p00, p10, p11, p01):
            p00, p10, p11, p01 = (np.asarray(p, float) for p in (p00, p10, p11, p01))
            nu = max(2, int(np.linalg.norm(p10 - p00) / step))
            nv = max(2, int(np.linalg.norm(p01 - p00) / step))
            for iu in range(nu):
                for iv in range(nv):
                    su0, su1 = iu / nu, (iu + 1) / nu
                    sv0, sv1 = iv / nv, (iv + 1) / nv

                    def bl(su, sv):
                        return ((1 - su) * (1 - sv) * p00 + su * (1 - sv) * p10
                                + su * sv * p11 + (1 - su) * sv * p01)

                    quad(bl(su0, sv0), bl(su1, sv0), bl(su1, sv1), bl(su0, sv1))

        gq([0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0])
        gq([0, 0, H], [0, W, H], [L, W, H], [L, 0, H])
        gq([0, 0, 0], [0, 0, H], [L, 0, H], [L, 0, 0])
        gq([0, W, 0], [L, W, 0], [L, W, H], [0, W, H])
        gq([0, 0, 0], [0, W, 0], [0, W, H], [0, 0, H])
        gq([L, 0, 0], [L, 0, H], [L, W, H], [L, W, 0])
        V = np.array([p for t in tris for p in t], float)
        keys = np.round(V / 1e-5).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        verts = np.zeros((inv.max() + 1, 3))
        verts[inv] = V
        f = inv.reshape(-1, 3)
        good = [i for i, (a, b, c) in enumerate(f) if a != b and b != c and a != c]
        return verts, f[good]

    vertices, faces = grid_box()
    cfg = ConversionConfig()
    resolution = mesh_resolution(vertices, faces, cfg)
    regions = segment_planar(vertices, faces, cfg)
    bands = segment_smooth_bands(vertices, faces, set(), regions, cfg)
    fillets = detect_fillets_straight(
        vertices, faces, bands, regions, set(), cfg, resolution)
    assert fillets == []


def test_harmonize_snaps_near_equal_radii():
    from mesh2step.fitting import Cylinder, _harmonize_radii

    def cyl(r):
        return Cylinder(
            axis_point=__import__("numpy").zeros(3),
            axis_dir=__import__("numpy").array([0.0, 0, 1]),
            radius=r, axial_min=0, axial_max=1, rms=0.0,
            face_indices=list(range(100)),
        )

    cfg = ConversionConfig(harmonize_rel_tol=0.03, harmonize_round=0.05)
    cyls = [cyl(6.041), cyl(6.047), cyl(6.064), cyl(3.02), cyl(2.98)]
    _harmonize_radii(cyls, cfg)
    radii = sorted({round(c.radius, 3) for c in cyls})
    # 6.04/6.05/6.06 collapse to one value; 3.02/2.98 collapse to 3.00.
    assert radii == [3.0, 6.05]
