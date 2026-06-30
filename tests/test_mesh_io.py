"""Tests for STL loading and vertex welding (no FreeCAD required)."""

from __future__ import annotations

import numpy as np

from mesh2step.mesh_io import load_stl, weld_vertices


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
