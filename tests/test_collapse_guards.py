"""Regression tests for the sweep's P0/P1 fixes.

Pin the invariants that the full-corpus sweep found broken on five files, using
synthetic minimal repros (the corpus STLs stay out of git):

- P0-1: a degenerate boolean returning a valid-but-tiny solid must be rejected by
  the guarded-fuse/-cut collapse guards and by ``_try_boolean_step`` (base_lid).
- P1-1: a face list containing a ``Compound`` (not a bare ``Face``) must still
  build a shell — flattened, not aborted to a faceted fallback (double_4u's
  ``TopoDS_UnCompatibleShapes``).
- P0-3: the export re-read helper distinguishes a valid solid from garbage.
- P1-2 / the pipeline bbox-ceiling gate are covered in
  ``test_pipeline_gate.py`` (pure-numpy) so they run without FreeCAD.
"""

from __future__ import annotations

import pytest

try:  # pragma: no cover - environment probe
    from mesh2step.freecad_env import ensure_freecad

    ensure_freecad(None)
    import Part  # type: ignore

    HAVE_FREECAD = True
except Exception:  # noqa: BLE001
    HAVE_FREECAD = False

pytestmark = pytest.mark.skipif(not HAVE_FREECAD, reason="FreeCAD not importable")


# --- P1-1: shell build must survive a compound in the face list -----------

def test_shell_from_faces_flattens_a_compound():
    """double_4u leaked one ``Compound`` into ``occ_faces``; the bulk
    ``Part.Shell`` then threw ``TopoDS_UnCompatibleShapes`` and the whole part
    dropped to a faceted fallback. The helper must flatten the compound into its
    member faces and build the shell from all of them."""
    from mesh2step.builder import _shell_from_faces

    box = Part.makeBox(10, 10, 10)
    compound = Part.makeCompound([box.Faces[0], box.Faces[1]])
    faces = [box.Faces[2], box.Faces[3], compound, box.Faces[4], box.Faces[5]]

    # Sanity: the bulk call the old code used genuinely fails on this list.
    with pytest.raises(Exception):
        Part.Shell(faces)

    msgs: list[str] = []
    shell = _shell_from_faces(faces, Part, msgs.append)
    # All 6 box faces survive (4 bare + 2 from the exploded compound).
    assert len(shell.Faces) == 6
    assert any("flattened" in m for m in msgs)


def test_shell_from_faces_fast_path_untouched():
    """A clean list of bare faces takes the fast path and logs nothing."""
    from mesh2step.builder import _shell_from_faces

    box = Part.makeBox(5, 5, 5)
    msgs: list[str] = []
    shell = _shell_from_faces(list(box.Faces), Part, msgs.append)
    assert len(shell.Faces) == 6
    assert msgs == []  # no flatten/drop lines on the happy path


def test_shell_from_faces_drops_null_entries():
    from mesh2step.builder import _shell_from_faces

    box = Part.makeBox(4, 4, 4)
    faces = list(box.Faces) + [Part.Shape()]  # a null shape
    msgs: list[str] = []
    shell = _shell_from_faces(faces, Part, msgs.append)
    assert len(shell.Faces) == 6
    assert any("dropped" in m for m in msgs)


# --- P0-1: guarded fuse/cut must reject a part-collapsing boolean ---------

class _FakeSolid:
    """Minimal stand-in for a Part.Solid: ``_guarded_fuse``/``_guarded_cut`` only
    use ``.Volume`` and the ``.fuse``/``.cut`` methods. Part.Solid is an immutable
    C++ type (can't monkeypatch its methods), so a fake lets us force the exact
    degenerate result OCC returned on base_lid without needing that geometry."""

    def __init__(self, volume, result):
        self.Volume = volume
        self._result = result

    def fuse(self, _tool):
        return self._result

    def cut(self, _tool):
        return self._result


class _Vol:
    def __init__(self, volume):
        self.Volume = volume


def test_guarded_fuse_rejects_a_shrinking_fuse():
    """A fuse can only ADD material, so a result smaller than the input is a
    degenerate boolean (base_lid: Vol 173576 -> 119). The guard checked only
    over-addition (a NEGATIVE 'added' slipped through); it must now also reject a
    fuse that shrinks the part."""
    import mesh2step.builder as b

    # Fuse "returns" a tiny fragment (Vol 119) from a big part (Vol 173576).
    solid = _FakeSolid(173576.0, _Vol(119.0))
    tool = _Vol(314.0)
    with pytest.raises(ValueError, match="shrank|collapse"):
        b._guarded_fuse(solid, tool)


def test_guarded_cut_rejects_carving_the_whole_part():
    """A local cut removes a bounded share of the tool; one that carves >30% of
    the whole solid is a degenerate boolean and must be rejected even if the
    tool-relative check passes."""
    import mesh2step.builder as b

    # Cut "returns" a tiny fragment: removes 999997 of a 1e6 part. The large tool
    # keeps the tool-relative check lax so only the part-collapse net can fire.
    solid = _FakeSolid(1_000_000.0, _Vol(3.0))
    tool = _Vol(729_000.0)
    with pytest.raises(ValueError, match="collapse|remove"):
        b._guarded_cut(solid, tool, max_removed_frac=0.99)


def test_guarded_fuse_accepts_a_normal_boss():
    """A legitimate boss fuse (a small bump on a plate) must still pass."""
    import FreeCAD  # type: ignore

    import mesh2step.builder as b

    plate = Part.makeBox(50, 50, 5)
    boss = Part.makeCylinder(3, 3, FreeCAD.Vector(25, 25, 5), FreeCAD.Vector(0, 0, 1))
    fused = b._guarded_fuse(plate, boss, max_added_frac=1.5)
    assert fused.isValid()
    assert fused.Volume > plate.Volume


# --- P0-3: export round-trip helper distinguishes valid from garbage ------

def test_reread_valid_true_for_a_real_solid():
    from mesh2step.builder import _reread_valid

    assert _reread_valid(Part.makeBox(10, 10, 10), Part) is True


def test_reread_valid_false_for_a_bare_shell():
    """A shell (no solid) written to STEP re-reads with no valid solid."""
    from mesh2step.builder import _reread_valid

    shell = Part.makeBox(10, 10, 10).Shells[0].Faces[0]  # a single face
    assert _reread_valid(shell, Part) is False
