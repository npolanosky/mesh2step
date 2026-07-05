"""Regression tests for the P0 geometry-artifact guards (user-reported P0).

Three real converted parts (port_cover, patton_core, patton_pad) shipped with
visible geometry defects: spherical caps bulging out of / into surfaces, and
round bores clipped into D-shapes by later cut ops. The root causes and their
fixes are pinned here with fast synthetic repros (the corpus STLs stay out of
git, and a full end-to-end convert is minutes/part — too slow for CI):

- **Local deviation guard** (spheres): the global RTAF gate is area-weighted, so
  a small bogus cap next to a fillet barely moves it yet is a glaring wrong bump.
  ``_local_deviation_ok`` reverts a cap whose surface strays off the input mesh
  more than the region did faceted. A correct inscribed cap is kept.
- **Hole-coverage gate** (bores): a swept/freeform/organic CUT whose tool reaches
  across a round hole slices a chord (D-shape) through the wall. ``_bore_guards``
  + ``_bore_intact`` detect the filled void and ``_try_boolean_step`` reverts it.
- **Persistent boolean server**: the isolated-cut worker is warm (one FreeCAD
  import per run, not per op) and recovers from a per-request timeout.
"""

from __future__ import annotations

import numpy as np
import pytest

try:  # pragma: no cover - environment probe
    from mesh2step.freecad_env import ensure_freecad

    ensure_freecad(None)
    import FreeCAD  # type: ignore  # noqa: F401
    import Part  # type: ignore

    HAVE_FREECAD = True
except Exception:  # noqa: BLE001
    HAVE_FREECAD = False

pytestmark = pytest.mark.skipif(not HAVE_FREECAD, reason="FreeCAD not importable")


# --- hole-coverage gate: round bore intact, D-chord clipped ---------------- #


def _box_with_round_hole(R=6.0):
    from mesh2step.fitting import Cylinder

    box = Part.makeBox(40, 40, 20, FreeCAD.Vector(-20, -20, 0))
    hole = Part.makeCylinder(R, 30, FreeCAD.Vector(0, 0, -5))
    part = box.cut(hole)
    cyl = Cylinder(axis_point=np.array([0.0, 0.0, 0.0]),
                   axis_dir=np.array([0.0, 0.0, 1.0]), radius=R,
                   axial_min=0.0, axial_max=20.0, rms=0.0, face_indices=[0],
                   outward=False, coverage=1.0)
    return part, cyl


def test_bore_guard_keeps_round_hole():
    from mesh2step import builder

    part, cyl = _box_with_round_hole()
    guards = builder._bore_guards([cyl], Part)
    assert guards, "a full-circle bore must produce a guard"
    assert builder._bore_intact(part, guards, Part) is True


def test_bore_guard_flags_d_shaped_hole():
    """A chord cut across the bore fills part of the void with material — the
    guard must detect the clipped (D-shaped) wall."""
    from mesh2step import builder

    part, cyl = _box_with_round_hole()
    guards = builder._bore_guards([cyl], Part)
    # Plug an arc of the bore void with solid: a D-shape.
    filler = Part.makeBox(12, 3, 30, FreeCAD.Vector(-6, 2, -5))
    d_part = part.fuse(filler)
    assert builder._bore_intact(d_part, guards, Part) is False


def test_bore_guard_ignores_partial_arc_cylinders():
    """A partial-arc 'hole' (a counterbore lip, a fillet section) is not a closed
    round wall to preserve — guarding it would mis-fire, so it is skipped."""
    from mesh2step import builder
    from mesh2step.fitting import Cylinder

    partial = Cylinder(axis_point=np.array([0.0, 0.0, 0.0]),
                       axis_dir=np.array([0.0, 0.0, 1.0]), radius=6.0,
                       axial_min=0.0, axial_max=20.0, rms=0.0, face_indices=[0],
                       outward=False, coverage=0.3)
    assert builder._bore_guards([partial], Part) == []


def test_try_boolean_step_reverts_a_bore_clipping_cut():
    """End-to-end through ``_try_boolean_step``: an op that D-shapes a guarded
    bore is reverted to the pre-op solid, exactly like an invalid/bbox breach."""
    from mesh2step import builder

    part, cyl = _box_with_round_hole()
    guards = builder._bore_guards([cyl], Part)

    def clip_op(s):
        filler = Part.makeBox(12, 3, 30, FreeCAD.Vector(-6, 2, -5))
        return s.fuse(filler)

    result, ok = builder._try_boolean_step(part, clip_op, bore_guards=guards, Part=Part)
    assert ok is False
    assert result is part  # reverted


# --- local deviation guard: bulging cap reverts, inscribed cap kept -------- #


def _dome_solid_and_mesh():
    """A flat plate whose top carries a shallow spherical dome, plus a KD-tree of
    a mesh that has NO such dome (a flat top). A fuse that adds the dome therefore
    bulges off the mesh; the guard must catch it."""
    from mesh2step import builder

    plate = Part.makeBox(40, 40, 10, FreeCAD.Vector(-20, -20, 0))
    # Mesh = the plate's flat surfaces (triangle soup): sample the flat top plane.
    xs = np.linspace(-20, 20, 12)
    ys = np.linspace(-20, 20, 12)
    verts = []
    faces = []
    grid = {}
    idx = 0
    for x in xs:
        for y in ys:
            grid[(x, y)] = idx
            verts.append([x, y, 10.0])
            idx += 1
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            a = grid[(xs[i], ys[j])]
            b = grid[(xs[i + 1], ys[j])]
            c = grid[(xs[i + 1], ys[j + 1])]
            d = grid[(xs[i], ys[j + 1])]
            faces.append([a, b, c])
            faces.append([a, c, d])
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    tree = builder._mesh_kdtree(verts, faces)
    return plate, tree


def test_local_deviation_guard_reverts_bulging_cap():
    """A dome fused onto a plate whose mesh is flat there bulges ~mm off the mesh
    — the guard must reject it."""
    from mesh2step import builder
    from mesh2step.config import ConversionConfig

    plate, tree = _dome_solid_and_mesh()
    # A big ball fused on top adds a dome sticking well above the flat mesh top.
    ball = Part.makeSphere(8.0, FreeCAD.Vector(0, 0, 10))
    bulged = plate.fuse(ball)
    ok = builder._local_deviation_ok(
        plate, bulged, tree, np.array([0.0, 0.0, 10.0]), 10.0, Part,
        ConversionConfig(), edge=1.0)
    assert ok is False, "a cap bulging mm off the mesh must be reverted"


def test_local_deviation_guard_keeps_on_surface_op():
    """An op that does not move the surface off the mesh is kept (a no-op cut of a
    tiny sliver far from the sampled region leaves local deviation unchanged)."""
    from mesh2step import builder
    from mesh2step.config import ConversionConfig

    plate, tree = _dome_solid_and_mesh()
    # Cut a shallow notch at a corner, far from the sampled centre — the region
    # around (0,0,10) is unchanged, so the guard passes.
    notch = Part.makeBox(2, 2, 2, FreeCAD.Vector(-20, -20, 9))
    trimmed = plate.cut(notch)
    ok = builder._local_deviation_ok(
        plate, trimmed, tree, np.array([0.0, 0.0, 10.0]), 5.0, Part,
        ConversionConfig(), edge=1.0)
    assert ok is True


def test_local_deviation_guard_is_a_noop_without_tree():
    from mesh2step import builder
    from mesh2step.config import ConversionConfig

    plate, _ = _dome_solid_and_mesh()
    assert builder._local_deviation_ok(
        plate, plate, None, np.array([0.0, 0.0, 10.0]), 5.0, Part,
        ConversionConfig(), edge=1.0) is True


# --- persistent boolean server: warm reuse + timeout recovery -------------- #


def test_boolean_server_warm_reuse():
    from mesh2step import builder

    builder._BOOLEAN_SERVER = None
    box = Part.makeBox(20, 20, 20)
    for i in range(3):
        tool = Part.makeCylinder(3, 30, FreeCAD.Vector(5 + i, 5, -5))
        r = builder.isolated_cut(box, tool, Part, 60.0)
        assert r.Solids and r.Solids[0].isValid()
    assert builder._BOOLEAN_SERVER is not None
    proc = builder._BOOLEAN_SERVER._proc
    assert proc is not None and proc.poll() is None  # one worker, still alive


def test_boolean_server_recovers_after_timeout():
    """A per-request timeout kills the worker; the next cut respawns it."""
    from mesh2step import builder

    builder._BOOLEAN_SERVER = None
    box = Part.makeBox(20, 20, 20)
    tool = Part.makeCylinder(3, 30, FreeCAD.Vector(5, 5, -5))
    with pytest.raises((TimeoutError, RuntimeError)):
        builder.isolated_cut(box, tool, Part, 1e-4)  # absurdly tight -> timeout
    # The server must recover for the next (normal-timeout) call.
    r = builder.isolated_cut(box, Part.makeCylinder(3, 30, FreeCAD.Vector(5, 5, -5)),
                             Part, 60.0)
    assert r.Solids and r.Solids[0].isValid()
