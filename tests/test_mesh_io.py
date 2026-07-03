"""Tests for STL loading and vertex welding (no FreeCAD required)."""

from __future__ import annotations

import numpy as np

from mesh2step.mesh_io import (
    load_stl,
    split_components,
    weld_vertices,
    write_binary_stl,
)


def test_weld_collapses_duplicate_vertices(cube_triangles):
    vertices, faces = weld_vertices(cube_triangles, weld_tol=1e-5)
    # A cube has 8 unique corners and 12 triangles.
    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    # Every face index must point at a real vertex.
    assert faces.max() < len(vertices)


def test_load_binary_stl(cube_binary_stl):
    vertices, faces = load_stl(cube_binary_stl)
    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    # Bounding box is the 0..10 cube.
    assert np.allclose(vertices.min(axis=0), [0, 0, 0])
    assert np.allclose(vertices.max(axis=0), [10, 10, 10])


def test_degenerate_triangles_dropped():
    # Two triangles: one valid, one with a repeated corner.
    tris = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1]],  # degenerate
        ],
        dtype=np.float64,
    )
    _, faces = weld_vertices(tris)
    assert faces.shape[0] == 1


def test_split_components_single_body(cube_triangles):
    """A single connected cube is one component, returned unchanged."""
    verts, faces = weld_vertices(cube_triangles)
    comps = split_components(verts, faces)
    assert len(comps) == 1
    assert comps[0][0] is verts  # same arrays, no copy for the common case
    assert comps[0][1] is faces


def test_split_components_two_disjoint_bodies(cube_triangles):
    """Two cubes that share no vertex split into two re-indexed sub-meshes."""
    v1, f1 = weld_vertices(cube_triangles)
    v2 = v1 + np.array([100.0, 0.0, 0.0])  # far-away second cube
    verts = np.vstack([v1, v2])
    faces = np.vstack([f1, f1 + len(v1)])
    comps = split_components(verts, faces)
    assert len(comps) == 2
    # Largest-first, equal size here; each body has 8 verts and 12 faces.
    for cv, cf in comps:
        assert cv.shape == (8, 3)
        assert cf.shape == (12, 3)
        assert cf.max() < len(cv)  # re-indexed to its own vertices
    # The two bodies are 100 mm apart on X.
    xspan = comps[1][0][:, 0].mean() - comps[0][0][:, 0].mean()
    assert abs(abs(xspan) - 100.0) < 1e-6


def test_write_binary_stl_round_trips(cube_triangles, tmp_path):
    """write_binary_stl output re-loads to the same welded geometry."""
    verts, faces = weld_vertices(cube_triangles)
    path = tmp_path / "out.stl"
    write_binary_stl(verts, faces, path)
    rv, rf = load_stl(path)
    assert rv.shape == (8, 3)
    assert rf.shape == (12, 3)
    assert np.allclose(sorted(rv.tolist()), sorted(verts.tolist()))
