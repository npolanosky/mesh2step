"""Tests for planar segmentation and boundary extraction (no FreeCAD)."""

from __future__ import annotations

import numpy as np

from mesh2step.boundary import extract_face_loops
from mesh2step.config import ConversionConfig
from mesh2step.mesh_io import weld_vertices
from mesh2step.segmentation import (
    build_edge_adjacency,
    face_normals_and_areas,
    segment_planar,
)


def test_cube_segments_into_six_planes(cube_triangles):
    vertices, faces = weld_vertices(cube_triangles)
    regions = segment_planar(vertices, faces, ConversionConfig())
    # 12 triangles must collapse to exactly 6 planar regions.
    assert len(regions) == 6
    # Each region is two triangles.
    assert sorted(r.size for r in regions) == [2, 2, 2, 2, 2, 2]
    # Region normals should be the 6 axis directions.
    normals = np.array([r.plane_normal for r in regions])
    for axis in range(3):
        col = np.abs(normals[:, axis])
        assert np.any(col > 0.99)


def test_edge_adjacency_is_manifold(cube_triangles):
    vertices, faces = weld_vertices(cube_triangles)
    adjacency = build_edge_adjacency(faces)
    # A closed manifold mesh: every edge shared by exactly two triangles.
    assert all(len(v) == 2 for v in adjacency.values())


def test_cube_face_has_four_corners(cube_triangles):
    vertices, faces = weld_vertices(cube_triangles)
    cfg = ConversionConfig()
    regions = segment_planar(vertices, faces, cfg)
    loops = extract_face_loops(vertices, faces, regions[0], cfg)
    assert loops is not None
    # A square face, after collinear simplification, is a 4-point loop with no
    # holes — the whole point of reconstruction vs. faceting.
    assert len(loops.outer) == 4
    assert loops.holes == []


def test_normals_unit_length(cube_triangles):
    vertices, faces = weld_vertices(cube_triangles)
    normals, areas = face_normals_and_areas(vertices, faces)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # Each cube triangle is half of a 10x10 face -> area 50.
    assert np.allclose(areas, 50.0)
