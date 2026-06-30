"""Planar region growing over a welded triangle mesh.

Groups facets that share a common plane into regions, so that downstream code
can rebuild each region as a single planar STEP face instead of many triangles.
Pure numpy; no FreeCAD dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConversionConfig


@dataclass
class Region:
    """A set of coplanar facets and the plane they lie on."""

    face_indices: list[int]
    plane_point: np.ndarray  # (3,) a point on the plane (region centroid)
    plane_normal: np.ndarray  # (3,) unit normal

    @property
    def size(self) -> int:
        return len(self.face_indices)


def face_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-face unit normals (F,3) and areas (F,)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    safe = np.where(norm > 0, norm, 1.0)
    normals = cross / safe[:, None]
    return normals, areas


def build_edge_adjacency(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge (sorted vertex pair) to incident face indices."""
    adjacency: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            adjacency.setdefault(key, []).append(fi)
    return adjacency


def _face_neighbors(
    faces: np.ndarray, adjacency: dict[tuple[int, int], list[int]]
) -> list[list[int]]:
    """For each face, the list of faces sharing an edge with it."""
    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for incident in adjacency.values():
        if len(incident) < 2:
            continue
        for i in incident:
            for j in incident:
                if i != j:
                    neighbors[i].append(j)
    return neighbors


def segment_planar(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig | None = None,
) -> list[Region]:
    """Grow planar regions of coplanar facets.

    A facet joins a region when (a) its normal is within ``angle_tol`` of the
    seed plane normal and (b) all its vertices lie within ``dist_tol`` of the
    seed plane. Seeds are taken largest-area first so dominant faces anchor
    their plane before noise can.
    """
    config = config or ConversionConfig()
    normals, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    neighbors = _face_neighbors(faces, adjacency)

    n_faces = len(faces)
    visited = np.zeros(n_faces, dtype=bool)
    order = np.argsort(-areas)  # largest first
    cos_tol = config.angle_tol_cos
    dist_tol = config.dist_tol

    regions: list[Region] = []
    for seed in order:
        if visited[seed]:
            continue
        seed_normal = normals[seed]
        seed_point = vertices[faces[seed]].mean(axis=0)

        members: list[int] = []
        stack = [int(seed)]
        visited[seed] = True
        while stack:
            fi = stack.pop()
            members.append(fi)
            for nb in neighbors[fi]:
                if visited[nb]:
                    continue
                # Coplanar test: normal alignment (allow flipped winding) ...
                if abs(float(np.dot(normals[nb], seed_normal))) < cos_tol:
                    continue
                # ... and every vertex close to the seed plane.
                rel = vertices[faces[nb]] - seed_point
                if np.max(np.abs(rel @ seed_normal)) > dist_tol:
                    continue
                visited[nb] = True
                stack.append(nb)

        centroid = vertices[faces[members].reshape(-1)].mean(axis=0)
        regions.append(
            Region(
                face_indices=members,
                plane_point=centroid,
                plane_normal=seed_normal,
            )
        )
    return regions
