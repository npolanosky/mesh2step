"""Tests for the multi-body mode (auto / combine / separate).

Covers the config field + its validation, the pure-numpy routing heuristics
(bbox overlap and coincident-seam detection), and the meshprep.combine_bodies
union (manifold3d gated / its no-dep failure branch). No FreeCAD required — the
heuristics and the mode plumbing are exercised on synthetic meshes.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mesh2step.config import ConversionConfig
from mesh2step.mesh_io import (
    bodies_bbox_overlap,
    bodies_share_coincident_vertices,
    filter_junk_bodies,
    split_components,
    weld_vertices,
)

_HAS_MANIFOLD = importlib.util.find_spec("manifold3d") is not None


def _cube(ox: float = 0.0, oy: float = 0.0, oz: float = 0.0, s: float = 10.0):
    """A welded (verts, faces) unit-ish cube translated to the given origin."""
    c = np.array(
        [[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
         [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]], dtype=np.float64
    ) + [ox, oy, oz]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris.append([c[a], c[b], c[cc]])
        tris.append([c[a], c[cc], c[d]])
    return weld_vertices(np.asarray(tris, dtype=np.float64))


def _two_body_mesh(gap: float):
    """Two cubes as two disjoint shells, the second offset in +x by 10 + gap.

    gap == 0 -> shells share a coincident seam plane (touching, one part).
    gap  > 0 -> a clearance gap (a functional assembly; keep separate).
    """
    vA, fA = _cube(0.0)
    vB, fB = _cube(10.0 + gap)
    verts = np.vstack([vA, vB])
    faces = np.vstack([fA, fB + len(vA)])
    return split_components(verts, faces)


# --- junk-body filter (P0): a tiny sliver must not make a single-body part
#     read as multi-body / abort the sew ------------------------------------------

def _sliver():
    """An 8-facet degenerate sliver body (a stray shell / modelling artefact).

    A thin flake near the origin: 8 triangles over a tiny footprint, well below
    both the facet-count and the area-fraction thresholds beside a real cube.
    """
    pts = np.array([[0, 0, 100.0], [0.02, 0, 100.0], [0.01, 0.02, 100.0],
                    [0.01, 0.01, 100.002]], dtype=np.float64)
    tris = []
    faces = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
    for a, b, c in faces:
        tris.append([pts[a], pts[b], pts[c]])
        tris.append([pts[a], pts[c], pts[b]])  # both windings -> 8 facets
    return weld_vertices(np.asarray(tris, dtype=np.float64))


def test_filter_junk_drops_tiny_sliver_body():
    # Mirrors wc_sharpie_holder_v2: a real body + an 8-facet flake (both few
    # facets AND negligible area -> junk). At the defaults the flake is dropped,
    # so the part reads single-body and never hits the sew crash.
    vC, fC = _cube(0.0, s=100.0)  # big cube: 12 facets, dominant area
    vS, fS = _sliver()            # 8 facets, sub-mm² area
    verts = np.vstack([vC, vS])
    faces = np.vstack([fC, fS + len(vC)])
    comps = split_components(verts, faces)
    assert len(comps) == 2  # a real body + a flake
    kept, dropped = filter_junk_bodies(comps)  # defaults: 32 facets / 2% area
    assert dropped == 1
    assert len(kept) == 1
    assert len(kept[0][1]) == 12  # the cube's 12 facets survive


def test_filter_junk_keeps_two_real_bodies():
    # Two full cubes of equal size: each is 50% of the area, well above the 2%
    # floor, so neither is junk even though 12 < 32 facets (BOTH must be tiny).
    comps = _two_body_mesh(gap=5.0)
    kept, dropped = filter_junk_bodies(comps)
    assert dropped == 0
    assert len(kept) == 2


def test_filter_junk_never_empties():
    # Two slivers only: everything looks like junk -> keep the original split so a
    # caller never ends up with nothing to convert.
    vS1, fS1 = _sliver()
    vS2, fS2 = _sliver()
    vS2 = vS2 + [50.0, 0, 0]
    verts = np.vstack([vS1, vS2])
    faces = np.vstack([fS1, fS2 + len(vS1)])
    comps = split_components(verts, faces)
    kept, dropped = filter_junk_bodies(comps)
    assert dropped == 0
    assert len(kept) == len(comps)


def test_filter_junk_single_body_untouched():
    comps = [_cube(0.0)]
    kept, dropped = filter_junk_bodies(comps)
    assert dropped == 0
    assert len(kept) == 1


# --- config field + validation ------------------------------------------------

def test_multibody_mode_default_is_auto():
    assert ConversionConfig().multibody_mode == "auto"


def test_multibody_mode_accepts_all_three():
    for mode in ("auto", "combine", "separate"):
        assert ConversionConfig(multibody_mode=mode).multibody_mode == mode


def test_multibody_mode_rejects_unknown():
    with pytest.raises(ValueError, match="unknown multibody_mode"):
        ConversionConfig(multibody_mode="fuse")


def test_multibody_modes_constant_matches_validation():
    # The CLI/GUI derive their choices from this tuple; keep it in sync.
    assert ConversionConfig.MULTIBODY_MODES == ("auto", "combine", "separate")


# --- routing heuristics -------------------------------------------------------

def test_single_body_never_routes_to_combine():
    v, f = _cube(0.0)
    comps = [(v, f)]
    assert bodies_bbox_overlap(comps) is False
    assert bodies_share_coincident_vertices(comps) is False


def test_touching_shells_share_a_coincident_seam():
    """Two cubes meeting on a shared face plane => combine (one part)."""
    comps = _two_body_mesh(gap=0.0)
    assert len(comps) == 2
    assert bodies_bbox_overlap(comps) is True
    assert bodies_share_coincident_vertices(comps, tol=1e-3) is True


def test_near_coincident_seam_still_detected_within_tol():
    """A sub-tolerance FP gap between shells still reads as a shared seam."""
    comps = _two_body_mesh(gap=1e-4)  # < 1e-3 tol
    assert bodies_share_coincident_vertices(comps, tol=1e-3) is True


def test_clearance_gap_assembly_stays_separate():
    """A print-in-place-style clearance gap => NO coincident seam => separate.

    This is the conservative case: bboxes may even overlap for a nested part, but
    without a shared seam the bodies must not be fused.
    """
    comps = _two_body_mesh(gap=2.0)  # 2 mm clearance, well above tol
    assert bodies_share_coincident_vertices(comps, tol=1e-3) is False


def test_nested_bodies_overlap_bbox_but_no_seam():
    """A small cube fully inside a large one: bboxes overlap, no shared seam.

    Mirrors a hinge pin sitting inside its knuckle — the deciding signal
    (coincident seam) is absent, so auto keeps them separate.
    """
    vOuter, fOuter = _cube(0.0, 0.0, 0.0, s=30.0)
    vInner, fInner = _cube(10.0, 10.0, 10.0, s=5.0)  # floats inside, no contact
    verts = np.vstack([vOuter, vInner])
    faces = np.vstack([fOuter, fInner + len(vOuter)])
    comps = split_components(verts, faces)
    assert len(comps) == 2
    assert bodies_bbox_overlap(comps) is True          # boxes nest
    assert bodies_share_coincident_vertices(comps, tol=1e-3) is False  # but no seam


def test_disjoint_bodies_have_no_bbox_overlap():
    vA, fA = _cube(0.0)
    vB, fB = _cube(100.0)  # far away
    comps = split_components(np.vstack([vA, vB]), np.vstack([fA, fB + len(vA)]))
    assert bodies_bbox_overlap(comps) is False
    assert bodies_share_coincident_vertices(comps) is False


# --- meshprep.combine_bodies (manifold3d) ------------------------------------

def test_combine_bodies_skips_cleanly_without_manifold3d(monkeypatch):
    """When manifold3d is unavailable, combine returns None with a logged reason
    naming an environment issue (so the caller falls back to separate)."""
    import builtins

    import mesh2step.meshprep as mp

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "manifold3d":
            raise ImportError("no manifold3d")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    msgs: list[str] = []
    v = np.zeros((3, 3))
    f = np.array([[0, 1, 2]])
    out = mp.combine_bodies(v, f, on_progress=msgs.append)
    assert out is None
    assert any("not installed" in m and "environment" in m for m in msgs)


@pytest.mark.skipif(not _HAS_MANIFOLD, reason="manifold3d not importable here")
def test_combine_bodies_unions_two_touching_cubes():
    """Two coincident-seam cubes fuse into ONE solid (a single bar)."""
    import mesh2step.meshprep as mp

    comps = _two_body_mesh(gap=1e-4)
    verts = np.vstack([comps[0][0], comps[1][0]])
    faces = np.vstack([comps[0][1], comps[1][1] + len(comps[0][0])])
    out = mp.combine_bodies(verts, faces, weld=1e-3)
    assert out is not None
    v2, f2, report = out
    assert len(f2) > 0
    # The union of two abutting 10-mm cubes is one 20 x 10 x 10 bar.
    ext = v2.max(axis=0) - v2.min(axis=0)
    assert np.isclose(sorted(ext, reverse=True)[0], 20.0, atol=0.05)
    assert report["welded_vertices"] >= 4  # the shared seam corners merged
