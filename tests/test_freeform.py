"""Freeform B-spline sheet reconstruction (Candidate B) against ground truth.

The ``freeform_bump`` sample is a thin plate whose top is a genuinely
doubly-curved sinusoidal surface (z = amp*sin(0.3x)*cos(0.25y) + 6 over a
40x40 footprint) — not a cylinder, cone, sphere, or constant-cross-section
sweep, so only a fitted B-spline sheet can reproduce it. Detection/sampling
tests are numpy-only; the reconstruction test needs FreeCAD and is skipped when
it isn't importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh2step.config import ConversionConfig
from mesh2step.mesh_io import load_stl
from mesh2step.segmentation import (
    sample_freeform_grid,
    segment_freeform_sheets,
)

DATA = Path(__file__).parent / "data"
SAMPLES = (
    json.loads((DATA / "samples.json").read_text())
    if (DATA / "samples.json").exists() else []
)
TRUTH = next((t for t in SAMPLES if t.get("kind") == "freeform_bump"), None)


# --------------------------------------------------------------------------- #
# Detection / sampling (numpy only)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(TRUTH is None, reason="freeform_bump sample not generated")
def test_freeform_region_detected_as_injective_double_curved_field():
    """The bump top is found as ONE height-field region: injective (foldover ~0)
    under a projection axis close to +Z, and genuinely doubly-curved (its
    peak-to-peak height ~ the design amplitude)."""
    cfg = ConversionConfig()
    vertices, faces = load_stl(DATA / TRUTH["file"])
    regions = segment_freeform_sheets(vertices, faces, set(), cfg)

    assert len(regions) == 1, f"expected 1 freeform region, got {len(regions)}"
    region = regions[0]
    assert region.foldover <= cfg.freeform_max_foldover
    # Projection axis is essentially vertical (the plate top faces up).
    assert abs(float(region.axis[2])) > 0.95
    # Curvature (peak-to-peak height) is on the order of the design amplitude.
    amp = TRUTH["freeform"]["amp"]
    assert region.curvature == pytest.approx(2 * amp, rel=0.5)


@pytest.mark.skipif(TRUTH is None, reason="freeform_bump sample not generated")
def test_freeform_grid_sampling_covers_footprint():
    """Resampling the region into a (u,v) grid fills nearly the whole footprint
    (few cells fall outside), so the fitted sheet is honest, not fabricated."""
    cfg = ConversionConfig()
    vertices, faces = load_stl(DATA / TRUTH["file"])
    regions = segment_freeform_sheets(vertices, faces, set(), cfg)
    assert regions
    sampled = sample_freeform_grid(vertices, faces, regions[0], cfg.freeform_grid)
    assert sampled is not None
    grid, missing = sampled
    assert grid.shape == (cfg.freeform_grid, cfg.freeform_grid, 3)
    assert missing <= cfg.freeform_max_missing


def test_no_freeform_sheets_on_prismatic_parts():
    """A plain box / L-bracket has no doubly-curved residual: nothing to fit."""
    for name in ("cube", "l_bracket", "plate_with_holes"):
        path = DATA / f"{name}.stl"
        if not path.exists():
            pytest.skip("samples not generated")
        cfg = ConversionConfig()
        vertices, faces = load_stl(path)
        assert segment_freeform_sheets(vertices, faces, set(), cfg) == []


def test_no_freeform_sheet_on_single_curvature_wall():
    """A constant-cross-section swept wall curves in ONE direction only; the
    double-curvature gate must reject it (the swept detector owns it)."""
    path = DATA / "swept_wavy_wall.stl"
    if not path.exists():
        pytest.skip("samples not generated")
    cfg = ConversionConfig()
    vertices, faces = load_stl(path)
    # No region should survive the double-curvature gate on a pure sweep.
    assert segment_freeform_sheets(vertices, faces, set(), cfg) == []


def test_freeform_disabled_by_flag():
    """The whole feature is behind ``fit_freeform_sheets``; off => no sheets."""
    from mesh2step.fitting import fit_freeform_sheets

    path = DATA / (TRUTH["file"] if TRUTH else "cube.stl")
    if not path.exists():
        pytest.skip("samples not generated")
    cfg = ConversionConfig(fit_freeform_sheets=False)
    vertices, faces = load_stl(path)
    assert fit_freeform_sheets(vertices, faces, set(), cfg) == []


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
@pytest.mark.skipif(TRUTH is None, reason="freeform_bump sample not generated")
def test_freeform_reconstruction_plants_bspline_and_drops_rtaf():
    """End-to-end: the bump top reconstructs as a watertight solid carrying an
    analytic B-spline face (not a strip fan); the residual tessellation area
    fraction drops sharply and the sheet deviates from the mesh within tol."""
    from mesh2step import builder

    cfg = ConversionConfig()
    vertices, faces = load_stl(DATA / TRUTH["file"])
    solid, stats = builder.build_boolean_clean_solid(vertices, faces, cfg)

    assert stats["is_solid"] is True
    assert stats["freeform_sheets_detected"] >= 1
    assert stats["freeform_sheets_built"] >= 1

    n_bspline = sum(
        1 for f in solid.Faces if "BSpline" in f.Surface.TypeId
    )
    assert n_bspline >= 1, "no analytic B-spline face in the result"

    rtaf = builder.compute_rtaf(solid, cfg)
    assert rtaf["rtaf"] is not None and rtaf["rtaf"] < 0.15


@pytest.mark.skipif(not HAVE_FREECAD, reason="FreeCAD not importable")
@pytest.mark.skipif(TRUTH is None, reason="freeform_bump sample not generated")
def test_freeform_reconstruction_is_bbox_stable():
    """The freeform boolean must not distort the part's overall dimensions."""
    from mesh2step import builder

    cfg = ConversionConfig()
    vertices, faces = load_stl(DATA / TRUTH["file"])
    in_dims = sorted((vertices.max(axis=0) - vertices.min(axis=0)).tolist(), reverse=True)
    solid, _ = builder.build_boolean_clean_solid(vertices, faces, cfg)
    bb = solid.BoundBox
    out_dims = sorted([bb.XLength, bb.YLength, bb.ZLength], reverse=True)
    for a, b in zip(in_dims, out_dims):
        assert abs(a - b) / a < 0.03
