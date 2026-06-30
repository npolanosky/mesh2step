"""Boundary-loop extraction for planar regions.

Given a planar region (a connected set of coplanar facets), recover the
polygonal boundary of the region: one outer loop plus zero or more hole loops,
each simplified by dropping collinear vertices. Pure numpy; no FreeCAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ConversionConfig
from .segmentation import Region


@dataclass
class FaceLoops:
    """Boundary of one planar region: an outer loop and any holes.

    Each loop is an (k, 3) array of 3D points in order (not repeating the first
    point at the end).
    """

    outer: np.ndarray
    holes: list[np.ndarray] = field(default_factory=list)
    normal: np.ndarray | None = None


def _directed_boundary_edges(faces: np.ndarray, members: list[int]) -> list[tuple[int, int]]:
    """Return oriented boundary half-edges of the region.

    A directed edge (u, v) from a triangle's winding is on the boundary when its
    reverse (v, u) is not also contributed by another triangle in the region.
    The resulting orientation follows the facet winding (outer loop CCW about
    the normal, holes CW).
    """
    directed: dict[tuple[int, int], int] = {}
    for fi in members:
        a, b, c = (int(x) for x in faces[fi])
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1

    boundary: list[tuple[int, int]] = []
    for (u, v), count in directed.items():
        reverse = directed.get((v, u), 0)
        # net (u,v) edges beyond cancelled interior pairs are boundary edges
        for _ in range(count - min(count, reverse)):
            boundary.append((u, v))
    return boundary


def _chain_loops(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Chain directed boundary edges into ordered vertex loops."""
    from collections import defaultdict, deque

    out: dict[int, deque[int]] = defaultdict(deque)
    for u, v in edges:
        out[u].append(v)

    loops: list[list[int]] = []
    for start in list(out.keys()):
        while out[start]:
            loop = [start]
            cur = out[start].popleft()
            while cur != start:
                loop.append(cur)
                if not out[cur]:
                    break  # open chain (non-manifold); abandon
                cur = out[cur].popleft()
            if cur == start and len(loop) >= 3:
                loops.append(loop)
    return loops


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes for a given unit normal."""
    n = normal / (np.linalg.norm(normal) or 1.0)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(n, u)
    return u, v


def _signed_area(points2d: np.ndarray) -> float:
    """Shoelace signed area of a 2D polygon."""
    x, y = points2d[:, 0], points2d[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _simplify_collinear(points3d: np.ndarray, tol: float) -> np.ndarray:
    """Drop vertices whose removal moves the polygon edge by less than ``tol``."""
    pts = list(points3d)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        i = 0
        while i < len(pts) and len(pts) > 3:
            prev = pts[i - 1]
            cur = pts[i]
            nxt = pts[(i + 1) % len(pts)]
            chord = nxt - prev
            chord_len = np.linalg.norm(chord)
            if chord_len > 0:
                # perpendicular distance of cur from the prev->nxt line
                dist = np.linalg.norm(np.cross(chord, cur - prev)) / chord_len
                if dist < tol:
                    pts.pop(i)
                    changed = True
                    continue
            i += 1
    return np.asarray(pts)


def extract_face_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    region: Region,
    config: ConversionConfig | None = None,
) -> FaceLoops | None:
    """Extract the outer boundary and holes of a planar region.

    Returns ``None`` if no closed boundary loop could be recovered (the region
    should then fall back to faceted output).
    """
    config = config or ConversionConfig()
    edges = _directed_boundary_edges(faces, region.face_indices)
    if not edges:
        return None
    loops = _chain_loops(edges)
    if not loops:
        return None

    u, v = _plane_basis(region.plane_normal)
    origin = region.plane_point

    scored: list[tuple[float, np.ndarray]] = []
    for loop in loops:
        pts3d = vertices[loop]
        rel = pts3d - origin
        pts2d = np.column_stack((rel @ u, rel @ v))
        area = _signed_area(pts2d)
        pts3d = _simplify_collinear(pts3d, config.collinear_tol)
        if len(pts3d) >= 3:
            scored.append((area, pts3d))

    if not scored:
        return None

    # Outer loop = largest absolute area; the rest are holes.
    scored.sort(key=lambda s: abs(s[0]), reverse=True)
    outer = scored[0][1]
    holes = [pts for _, pts in scored[1:]]
    return FaceLoops(outer=outer, holes=holes, normal=region.plane_normal)
