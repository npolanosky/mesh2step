"""Sphere / dome / corner-blend reconstruction (M3) against ground truth.

The ``domed_plate`` sample is a 60x60x10 plate with a convex spherical cap of
known radius R=20 tangent to the top face. Detection/fitting tests are numpy-
only; the reconstruction test needs FreeCAD and is skipped when it isn't
importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mesh2step.config import ConversionConfig
from mesh2step.mesh_io import load_stl
from mesh2step.fitting import _fit_sphere, detect_spheres
from mesh2step.segmentation import (
    mesh_resolution,
    segment_planar,
    segment_smooth_bands,
)

DATA = Path(__file__).parent / "data"
SAMPLES = (
    json.loads((DATA / "samples.json").read_text())
    if (DATA / "samples.json").exists() else []
)
TRUTH = next((t for t in SAMPLES if t.get("kind") == "domed_plate"), None)


def _detect(name: str):
    """Run the full sphere-detection helper on a sample, return the spheres."""
    from mesh2step import builder

    cfg = ConversionConfig()
    vertices, faces = load_stl(DATA / name)
    claimed: set[int] = set()
    return builder._detect_spheres(vertices, faces, claimed, cfg, lambda m: None)


# --------------------------------------------------------------------------- #
# Numpy-only: algebraic fit + detection
# --------------------------------------------------------------------------- #


def test_fit_sphere_recovers_exact_radius():
    """The 4-parameter linear sphere fit recovers a known centre + radius from
    points sampled on a sphere (to numerical precision)."""
    rng = np.random.default_rng(0)
    c0 = np.array([3.0, -2.0, 5.0])
    r0 = 7.5
    u = rng.normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = c0 + r0 * u
    center, radius, rms = _fit_sphere(pts)
    assert radius == pytest.approx(r0, abs=1e-6)
    assert np.allclose(center, c0, atol=1e-6)
    assert rms < 1e-6


@pytest.mark.skipif(TRUTH is None, reason="domed_plate sample not generated")
def test_dome_radius_recovered_from_ground_truth():
    """The domed_plate's cap fits as a convex sphere at the design radius R=20."""
    spheres = _detect(TRUTH["file"])
    assert len(spheres) >= 1, "no sphere detected on the domed plate"
    truth = TRUTH["spheres"][0]
    # The dominant cap is the one with the most facets.
    cap = max(spheres, key=lambda s: len(s.face_indices))
    assert cap.outward is True
    assert cap.radius == pytest.approx(truth["radius"], abs=0.1)
    assert np.allclose(cap.center, truth["center"], atol=0.1)


@pytest.mark.parametrize("name", [
    "cube", "plate_with_holes", "cylinder", "l_bracket", "fillet_chamfer_plate",
])
def test_no_false_positive_sphere_on_prismatic_parts(name):
    """Flat / cylindrical / filleted prismatic parts have no spherical cap:
    detection must return nothing (no dozens of bogus micro-spheres)."""
    path = DATA / f"{name}.stl"
    if not path.exists():
        pytest.skip("sample not generated")
    assert _detect(f"{name}.stl") == []


def test_detect_spheres_disabled_flag():
    """The whole feature can be switched off."""
    if TRUTH is None:
        pytest.skip("domed_plate sample not generated")
    cfg = ConversionConfig(detect_spheres=False)
    vertices, faces = load_stl(DATA / TRUTH["file"])
    resolution = mesh_resolution(vertices, faces, cfg)
    regions = segment_planar(vertices, faces, cfg)
    bands = segment_smooth_bands(vertices, faces, set(), regions, cfg)
    assert detect_spheres(
        vertices, faces, bands, regions, set(), cfg, resolution) == []


# --------------------------------------------------------------------------- #
# Reconstruction (FreeCAD required)
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - environment probe
    from mesh2step.freecad_env import ensure_freecad

    ensure_freecad(None)
    import Part  # type: ignore  # noqa: F401

    HAVE_FREECAD = True
except Exception:  # noqa: BLE001
    HAVE_FREECAD = False


@pytest.mark.skipif(not HAVE_FREECAD, reason="FreeCAD not importable")
@pytest.mark.skipif(TRUTH is None, reason="domed_plate sample not generated")
def test_dome_reconstructs_as_analytic_sphere_and_zero_rtaf():
    """End-to-end: the domed plate reconstructs as a watertight solid whose cap
    is a single analytic sphere face at the design radius, and RTAF drops to ~0."""
    from mesh2step import builder

    cfg = ConversionConfig(full_closed=True, repair_mesh=True)
    vertices, faces = load_stl(DATA / TRUTH["file"])
    solid, stats = builder.build_boolean_clean_solid(vertices, faces, cfg)

    assert stats["is_solid"] is True
    assert stats["spheres_detected"] >= 1
    assert stats["spheres_built"] >= 1

    sphere_faces = [f for f in solid.Faces
                    if f.Surface.TypeId == "Part::GeomSphere"]
    assert len(sphere_faces) >= 1
    radii = [f.Surface.Radius for f in sphere_faces]
    assert any(r == pytest.approx(TRUTH["spheres"][0]["radius"], abs=0.1)
               for r in radii)

    rtaf = builder.compute_rtaf(solid, cfg)
    assert rtaf["rtaf"] is not None and rtaf["rtaf"] < 0.05
