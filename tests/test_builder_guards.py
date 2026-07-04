"""Pure-numpy guard tests (no FreeCAD): degenerate-loop pre-validation and the
bounding-box growth/collapse guards used by the boolean clean-up ladder.
"""

from __future__ import annotations

import numpy as np

from mesh2step.builder import _bbox_collapsed, _bbox_grew, _loop_degenerate


def test_loop_degenerate_accepts_a_real_triangle():
    loop = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert _loop_degenerate(loop) is None


def test_loop_degenerate_rejects_collinear_loop():
    # Three distinct points on one line — zero area. This is exactly the loop
    # that made Part.Face abort() natively on the vase.
    loop = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    assert _loop_degenerate(loop) == "collinear loop (zero area)"


def test_loop_degenerate_rejects_too_few_points():
    assert _loop_degenerate(np.array([[0, 0, 0], [1, 0, 0]], dtype=float)) is not None


def test_loop_degenerate_rejects_consecutive_duplicates():
    loop = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert _loop_degenerate(loop) == "consecutive duplicate points"


def test_loop_degenerate_rejects_zero_span():
    loop = np.zeros((4, 3), dtype=float)
    assert _loop_degenerate(loop) is not None


def test_bbox_grew_flags_material_growth():
    before = (100.0, 50.0, 20.0)
    grown = (120.0, 50.0, 20.0)  # +20 % on the long axis
    assert _bbox_grew(before, grown, rel_tol=0.02)


def test_bbox_grew_ignores_fp_noise():
    before = (100.0, 50.0, 20.0)
    tiny = (100.001, 50.0, 20.0)  # sub-abs_tol wobble
    assert not _bbox_grew(before, tiny, rel_tol=0.02)


def test_bbox_grew_ignores_shrink():
    before = (100.0, 50.0, 20.0)
    smaller = (80.0, 50.0, 20.0)  # a cut removed material — never flagged
    assert not _bbox_grew(before, smaller, rel_tol=0.02)


# --- Collapse guard (P0-1): a degenerate boolean returning a tiny valid solid.
# gridfinity_base_lid's sphere fuse turned a 210x126x12mm plate into a ~6mm cube,
# a valid single solid that passed every growth/volume guard. _bbox_collapsed is
# the net that catches it; _try_boolean_step reverts on it unconditionally.

def test_bbox_collapsed_flags_catastrophic_collapse():
    # The exact base_lid signature: dominant side 209.5mm -> 6.53mm.
    before = (209.5, 125.69, 12.37)
    collapsed = (6.53, 6.53, 5.57)
    assert _bbox_collapsed(before, collapsed)


def test_bbox_collapsed_ignores_a_legitimate_cut():
    # A hole cut / edge trim removes at most a modest share of a side — never
    # collapses the dominant dimension by more than half.
    before = (100.0, 50.0, 20.0)
    trimmed = (98.0, 50.0, 20.0)  # 2% shorter after an edge slot
    assert not _bbox_collapsed(before, trimmed)


def test_bbox_collapsed_ignores_growth():
    before = (100.0, 50.0, 20.0)
    grown = (120.0, 50.0, 20.0)  # growth is _bbox_grew's job, not collapse
    assert not _bbox_collapsed(before, grown)


def test_bbox_collapsed_handles_empty_dims():
    assert not _bbox_collapsed((), (6.5, 6.5, 5.5))
    assert not _bbox_collapsed((100.0, 50.0), ())
