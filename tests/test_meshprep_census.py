"""Pure-numpy tests for the fixDegenerations livelock guard (P0).

``meshprep._degeneracy_census`` decides whether FreeCAD's ``fixDegenerations``
(which livelocks on tiny-edge/tiny-area facet clusters) should run. It imports
FreeCAD only lazily inside the FreeCAD-facing functions, so the census helper is
importable and testable without FreeCAD.
"""

from __future__ import annotations

import numpy as np

from mesh2step.meshprep import (
    _FIXDEGEN_MAX_TINY_EDGES,
    _FIXDEGEN_TINY_EDGE,
    _degeneracy_census,
)


def _clean_triangles(n: int = 40):
    """A strip of well-formed (non-degenerate) triangles, all edges ~1 mm."""
    verts = []
    faces = []
    for i in range(n):
        base = len(verts)
        verts += [[i, 0, 0], [i + 1, 0, 0], [i, 1, 0]]
        faces.append([base, base + 1, base + 2])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_census_clean_mesh_runs_fixdegen():
    v, f = _clean_triangles(40)
    census = _degeneracy_census(v, f)
    assert census["tiny_edges"] == 0
    assert census["tiny_area_facets"] == 0
    assert census["should_skip_fixdegen"] is False


def test_census_tiny_edge_cluster_skips_fixdegen():
    # Reproduce the wc_sharpie signature: many facets with a sub-1e-3 mm edge and
    # near-zero area — the cluster that livelocks fixDegenerations.
    tiny = _FIXDEGEN_TINY_EDGE / 10.0
    v, f = _clean_triangles(40)
    verts = list(v)
    faces = list(f)
    for _ in range(_FIXDEGEN_MAX_TINY_EDGES + 4):
        base = len(verts)
        # a needle triangle: two coincident-ish points -> tiny edge + tiny area
        verts += [[0, 0, 0], [tiny, 0, 0], [tiny, tiny, 0]]
        faces.append([base, base + 1, base + 2])
    census = _degeneracy_census(np.asarray(verts), np.asarray(faces))
    assert census["tiny_edges"] >= _FIXDEGEN_MAX_TINY_EDGES
    assert census["should_skip_fixdegen"] is True


def test_census_handful_of_tiny_facets_still_runs():
    # A couple of degenerate facets is harmless and must NOT skip (fixDegenerations
    # is fast and useful here); only a real cluster trips the guard.
    tiny = _FIXDEGEN_TINY_EDGE / 10.0
    v, f = _clean_triangles(40)
    verts = list(v)
    faces = list(f)
    for _ in range(2):
        base = len(verts)
        verts += [[0, 0, 0], [tiny, 0, 0], [tiny, tiny, 0]]
        faces.append([base, base + 1, base + 2])
    census = _degeneracy_census(np.asarray(verts), np.asarray(faces))
    assert census["should_skip_fixdegen"] is False


def test_census_empty_mesh():
    census = _degeneracy_census(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert census["should_skip_fixdegen"] is False
