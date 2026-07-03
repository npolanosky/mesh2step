"""Builder tests that need FreeCAD (skipped when it isn't importable).

Covers the ``_planar_face`` planarity fix (loops admitted within ``dist_tol``
that OCC's wire-based inference rejects as "Not planar" must still build via the
explicit fitted plane, without moving vertices) and a basic ``compute_rtaf``
sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest

# FreeCAD lives outside the test interpreter's site-packages; make it importable
# the same way the app does, then skip the whole module if it still isn't there.
try:  # pragma: no cover - environment probe
    from mesh2step.freecad_env import ensure_freecad

    ensure_freecad(None)
    import Part  # type: ignore

    HAVE_FREECAD = True
except Exception:  # noqa: BLE001
    HAVE_FREECAD = False

pytestmark = pytest.mark.skipif(not HAVE_FREECAD, reason="FreeCAD not importable")


def _loops(outer, holes=(), normal=(0.0, 0.0, 1.0)):
    from mesh2step.boundary import FaceLoops

    return FaceLoops(
        outer=np.asarray(outer, float),
        holes=[np.asarray(h, float) for h in holes],
        normal=np.asarray(normal, float),
    )


def test_planar_face_builds_slightly_nonplanar_loop():
    """A square whose corners sit a few microns off the plane (within dist_tol)
    fails ``Part.Face(wires)`` but must still build via the explicit-plane path,
    with the SAME vertex coordinates (no projection) so sewing is unaffected."""
    from mesh2step.builder import _planar_face

    # 10x10 quad with each corner nudged +/-0.005 mm off z=0 (< dist_tol 0.01).
    outer = [
        [0.0, 0.0, 0.005],
        [10.0, 0.0, -0.004],
        [10.0, 10.0, 0.006],
        [0.0, 10.0, -0.003],
    ]
    # Sanity: OCC's inference path genuinely rejects this loop.
    wire = Part.makePolygon([Part.Vertex(*p).Point for p in outer]
                            + [Part.Vertex(*outer[0]).Point])
    with pytest.raises(Exception):
        Part.Face(wire)

    face = _planar_face(_loops(outer), circles=[], Part=Part)
    assert face.isValid()
    assert face.Area == pytest.approx(100.0, rel=1e-3)

    # The wire vertices must be the ORIGINAL off-plane points (fix widens the
    # face tolerance rather than projecting), so shared edges stay put for sewing.
    zs = sorted(round(v.Z, 4) for v in face.Vertexes)
    assert zs == [-0.004, -0.003, 0.005, 0.006]


def test_planar_face_with_hole_builds_and_keeps_hole():
    """A near-planar quad with a near-planar hole still builds as a face with a
    hole (outer wire + one inner wire)."""
    from mesh2step.builder import _planar_face

    outer = [
        [0.0, 0.0, 0.004],
        [20.0, 0.0, -0.005],
        [20.0, 20.0, 0.003],
        [0.0, 20.0, -0.002],
    ]
    hole = [
        [8.0, 8.0, 0.002],
        [12.0, 8.0, -0.003],
        [12.0, 12.0, 0.004],
        [8.0, 12.0, -0.001],
    ]
    face = _planar_face(_loops(outer, holes=[hole]), circles=[], Part=Part)
    assert face.isValid()
    assert len(face.Wires) == 2  # outer + hole
    # 20x20 minus 4x4 hole = 400 - 16.
    assert face.Area == pytest.approx(384.0, rel=1e-3)


def test_compute_rtaf_flat_box_is_zero():
    """A plain box (6 planar faces meeting at 90 degrees) has no smooth chains,
    so RTAF is 0."""
    from mesh2step.builder import compute_rtaf
    from mesh2step.config import ConversionConfig

    box = Part.makeBox(10, 10, 10)
    info = compute_rtaf(box, ConversionConfig())
    assert info["rtaf"] == 0.0
    assert info["smooth_chains"] == 0


def test_compute_rtaf_respects_disable_flag():
    from mesh2step.builder import compute_rtaf
    from mesh2step.config import ConversionConfig

    box = Part.makeBox(1, 1, 1)
    info = compute_rtaf(box, ConversionConfig(compute_rtaf=False))
    assert info["rtaf"] is None
    assert info["skipped"] == "disabled"
