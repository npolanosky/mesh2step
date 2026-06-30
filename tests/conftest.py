"""Shared test fixtures: synthetic meshes built without FreeCAD."""

from __future__ import annotations

import struct

import numpy as np
import pytest


def _cube_triangles(size: float = 10.0) -> np.ndarray:
    """Return a (12, 3, 3) array of triangles for an axis-aligned cube.

    Each of the 6 faces is split into 2 triangles, vertices duplicated per
    triangle exactly as a real STL would store them.
    """
    s = size
    # 8 corners
    c = np.array(
        [
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
        ],
        dtype=np.float64,
    )
    # 6 quads as (a,b,c,d) with outward winding
    quads = [
        (0, 3, 2, 1),  # bottom z=0
        (4, 5, 6, 7),  # top    z=s
        (0, 1, 5, 4),  # front  y=0
        (2, 3, 7, 6),  # back   y=s
        (1, 2, 6, 5),  # right  x=s
        (0, 4, 7, 3),  # left   x=0
    ]
    tris = []
    for a, b, cc, d in quads:
        tris.append([c[a], c[b], c[cc]])
        tris.append([c[a], c[cc], c[d]])
    return np.asarray(tris, dtype=np.float64)


@pytest.fixture
def cube_triangles() -> np.ndarray:
    return _cube_triangles()


@pytest.fixture
def cube_binary_stl(tmp_path) -> str:
    """Write the synthetic cube as a binary STL and return its path."""
    tris = _cube_triangles()
    path = tmp_path / "cube.stl"
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for tri in tris:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n = np.linalg.norm(normal)
            normal = normal / n if n else normal
            fh.write(struct.pack("<3f", *normal))
            for v in tri:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))
    return str(path)
