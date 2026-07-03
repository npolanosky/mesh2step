"""Swept/extruded curved-wall reconstruction (M4) against ground truth.

The ``swept_wavy_wall`` sample is an extruded line+arc+line wall of known
profile (outer arc R=10, inner arc R=7, tangent joins by construction, 50 mm
extrusion in Z). Detection/fitting tests are numpy-only; the reconstruction
test needs FreeCAD and is skipped when it isn't importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mesh2step.config import ConversionConfig
from mesh2step.mesh_io import load_stl
from mesh2step.segmentation import mesh_resolution, segment_planar, segment_swept_walls

DATA = Path(__file__).parent / "data"
SAMPLES = (
    json.loads((DATA / "samples.json").read_text())
    if (DATA / "samples.json").exists() else []
)
TRUTH = next((t for t in SAMPLES if t.get("kind") == "swept_wavy_wall"), None)


def _fit_profiles():
    from mesh2step.fitting import detect_swept_walls

    cfg = ConversionConfig()
    vertices, faces = load_stl(DATA / TRUTH["file"])
    resolution = mesh_resolution(vertices, faces, cfg)
    regions = segment_planar(vertices, faces, cfg)
    sweeps = segment_swept_walls(vertices, faces, set(), regions, cfg)
    return detect_swept_walls(vertices, faces, sweeps, cfg, resolution)


@pytest.mark.skipif(TRUTH is None, reason="swept_wavy_wall sample not generated")
def test_swept_profile_fit_matches_ground_truth():
    """Both wall sides fit as line+arc+line; the arc radii are the design R=10
    (outer) and R-thickness=7 (inner), joins snapped to exact tangency."""
    truth = TRUTH["swept"][0]
    profiles = _fit_profiles()
    assert len(profiles) == 2, f"expected 2 swept walls, got {len(profiles)}"

    radii = []
    for prof in profiles:
        # Extrusion direction and extent match the ground truth.
        assert abs(float(prof.axis @ np.asarray(truth["axis"], float))) > 0.999
        assert prof.axial_max - prof.axial_min == pytest.approx(truth["extent"], abs=0.1)
        # Composition: two straight runs and one arc, all joins tangent-snapped.
        assert prof.n_lines == 2 and prof.n_arcs == 1 and prof.n_splines == 0
        assert prof.tangency_snaps == 2
        assert prof.rms < 0.05
        arc = next(s for s in prof.segments if s.kind == "arc")
        assert arc.tangent_start and arc.tangent_end
        radii.append(arc.radius)

    r_out = truth["arc_radius"]
    r_in = truth["arc_radius"] - truth["thickness"]
    assert sorted(radii) == pytest.approx([r_in, r_out], abs=0.02)


def test_no_swept_walls_on_prismatic_parts():
    """A plain box/L-bracket has no smooth strip chains: nothing to sweep."""
    from mesh2step.fitting import detect_swept_walls

    for name in ("cube", "l_bracket"):
        path = DATA / f"{name}.stl"
        if not path.exists():
            pytest.skip("samples not generated")
        cfg = ConversionConfig()
        vertices, faces = load_stl(path)
        resolution = mesh_resolution(vertices, faces, cfg)
        regions = segment_planar(vertices, faces, cfg)
        sweeps = segment_swept_walls(vertices, faces, set(), regions, cfg)
        assert detect_swept_walls(vertices, faces, sweeps, cfg, resolution) == []


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
@pytest.mark.skipif(TRUTH is None, reason="swept_wavy_wall sample not generated")
def test_swept_reconstruction_produces_analytic_walls_and_zero_rtaf():
    """End-to-end: the wall reconstructs as a watertight solid whose curved
    sides are analytic cylinder faces at the design radii (not strip fans), and
    the residual tessellation area fraction drops to ~0."""
    from mesh2step import builder

    cfg = ConversionConfig(fill_faceted_gaps=True)
    vertices, faces = load_stl(DATA / TRUTH["file"])
    shape, stats = builder.build_reconstructed_solid(vertices, faces, cfg)

    assert stats["is_solid"] is True
    assert stats["swept_walls_detected"] == 2
    assert stats["swept_walls_built"] == 2

    cyl_radii = sorted(
        f.Surface.Radius for f in shape.Faces
        if f.Surface.TypeId == "Part::GeomCylinder"
    )
    assert len(cyl_radii) == 2
    truth = TRUTH["swept"][0]
    r_out = truth["arc_radius"]
    r_in = truth["arc_radius"] - truth["thickness"]
    # The lens op leaves the wall at R + eps (micron-scale cut clearance).
    assert cyl_radii == pytest.approx([r_in, r_out], abs=0.05)

    rtaf = builder.compute_rtaf(shape, cfg)
    assert rtaf["rtaf"] is not None and rtaf["rtaf"] < 0.02
