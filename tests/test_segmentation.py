"""Tests for planar segmentation and boundary extraction (no FreeCAD)."""

from __future__ import annotations

import numpy as np

from mesh2step.boundary import extract_face_loops
from mesh2step.config import ConversionConfig
from mesh2step.mesh_io import weld_vertices
from mesh2step.segmentation import (
    build_edge_adjacency,
    face_normals_and_areas,
    planar_coverage,
    segment_planar,
)


def _grid_plane(n: int = 12, size: float = 60.0, jitter: float = 0.0, seed: int = 0):
    """An (n x n)-cell flat square in the z=0 plane, two triangles per cell.

    ``jitter`` perturbs each interior vertex's z by up to ``jitter`` mm — a
    controllable stand-in for decimation warping a genuinely-flat region past the
    coplanar gate. Returns welded ``(vertices, faces)``.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, size, n + 1)
    ys = np.linspace(0, size, n + 1)
    gx, gy = np.meshgrid(xs, ys)
    gz = np.zeros_like(gx)
    if jitter > 0:
        gz[1:-1, 1:-1] = rng.uniform(-jitter, jitter, size=gz[1:-1, 1:-1].shape)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)

    def idx(r, c):
        return r * (n + 1) + c

    faces = []
    for r in range(n):
        for c in range(n):
            a, b, d, e = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
            faces.append([a, b, e])
            faces.append([a, e, d])
    return verts, np.asarray(faces, dtype=np.int64)


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


def test_planar_coverage_full_on_clean_flat():
    # A clean flat grid: (almost) all area lands in ONE large planar region.
    verts, faces = _grid_plane(n=12, jitter=0.0)
    info = planar_coverage(verts, faces, ConversionConfig(), min_region_facets=8)
    assert info["coverage"] > 0.99
    # One dominant flat, not a shatter of micro-regions.
    assert info["n_big_regions"] == 1


def test_planar_coverage_full_on_cube(cube_triangles):
    # Every cube facet is in a 2-facet plane; with min_region_facets=2 all six
    # planes count and coverage is total.
    vertices, faces = weld_vertices(cube_triangles)
    info = planar_coverage(vertices, faces, ConversionConfig(), min_region_facets=2)
    assert info["coverage"] > 0.99
    assert info["n_big_regions"] == 6


def test_planar_coverage_drops_when_flat_is_warped():
    # Warping the interior of a flat past the coplanar gate (as coarse decimation
    # does) must shatter the one big region into sub-threshold micro-regions and
    # drop the coverage sharply — the signal the pipeline backs off on.
    cfg = ConversionConfig()
    clean = planar_coverage(*_grid_plane(n=14, jitter=0.0), config=cfg,
                            min_region_facets=8)
    # 0.6 mm jitter over ~4 mm cells -> normals step well past the 1.0 deg gate.
    warped = planar_coverage(*_grid_plane(n=14, jitter=0.6, seed=1), config=cfg,
                             min_region_facets=8)
    assert clean["coverage"] > 0.95
    assert warped["coverage"] < 0.75 * clean["coverage"]
    # The large flat fragmented into many more regions than the clean case.
    assert warped["n_regions"] > 5 * clean["n_regions"]
