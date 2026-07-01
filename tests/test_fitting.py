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
