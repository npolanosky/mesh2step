"""Cylinder detection and best-fit for clean hole/bore reconstruction.

Faceted cylinders (holes, bores, bosses, fillets) are the single biggest pain
when importing meshes into CAD: hundreds of tiny triangles that tools choke on
and that can't be used as a real hole. This module finds those regions and fits
a best-fit cylinder (axis + radius + axial extent) so the builder can rebuild
them as a single analytic cylindrical face with true circular edges.

Pure numpy; no FreeCAD. The builder consumes the :class:`Cylinder` results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConversionConfig
from .segmentation import build_edge_adjacency, face_normals_and_areas


@dataclass
class Cylinder:
    """A best-fit cylinder for a set of facets."""

    axis_point: np.ndarray   # (3,) a point on the axis (projected circle centre)
    axis_dir: np.ndarray     # (3,) unit axis direction
    radius: float
    axial_min: float         # min of (vertex-axis_point) . axis_dir over inliers
    axial_max: float
    rms: float               # RMS radial residual (mm)
    face_indices: list[int]  # facets assigned to this cylinder
    outward: bool = True     # True for a boss (material inside), False for a hole

    @property
    def height(self) -> float:
        return self.axial_max - self.axial_min

    @property
    def role(self) -> str:
        return "boss" if self.outward else "hole"

    def base_point(self) -> np.ndarray:
        """A point on the axis at the lower axial extent (cylinder base)."""
        return self.axis_point + self.axis_dir * self.axial_min

    def as_dict(self) -> dict:
        return {
            "radius": float(self.radius),
            "axis_dir": [float(x) for x in self.axis_dir],
            "axis_point": [float(x) for x in self.axis_point],
            "height": float(self.height),
            "rms": float(self.rms),
            "facets": len(self.face_indices),
            "role": self.role,
        }


def _connected_components(
    pool: list[int], neighbors: list[list[int]]
) -> list[list[int]]:
    """Connected components of ``pool`` faces using edge adjacency."""
    pool_set = set(pool)
    seen: set[int] = set()
    out: list[list[int]] = []
    for start in pool:
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in neighbors[f]:
                if nb in pool_set and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def _fit_circle_2d(pts2d: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa algebraic circle fit. Returns (center(2,), radius, rms_residual)."""
    x, y = pts2d[:, 0], pts2d[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy, c = sol
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    rms = float(np.sqrt(np.mean(resid**2))) if len(resid) else float("inf")
    return np.array([cx, cy]), radius, rms


def _plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(axis, u)
    return u, v


def _candidate_axes(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, areas: np.ndarray
) -> list[np.ndarray]:
    """Directions to try as cylinder axes.

    Holes/bores are almost always perpendicular to a flat face, so the normals
    of the largest flat faces are excellent axis candidates. We collect the
    dominant face-normal directions (by total area) plus the mesh principal
    axes as a fallback, de-duplicated up to sign.
    """
    # Bin facets by quantized normal direction, weight by area.
    weight: dict[tuple, float] = {}
    rep: dict[tuple, np.ndarray] = {}
    for n, a in zip(normals, areas):
        key = tuple(np.round(n, 2))
        # Fold antipodal directions together so +z and -z share a bin.
        if key < tuple(-x for x in key):
            key = tuple(-x for x in key)
            n = -n
        weight[key] = weight.get(key, 0.0) + float(a)
        rep.setdefault(key, n)
    ranked = sorted(weight, key=weight.get, reverse=True)
    axes = [rep[k] / (np.linalg.norm(rep[k]) or 1.0) for k in ranked[:6]]

    # Principal axes of the vertex cloud as a fallback for parts with no flat
    # face perpendicular to the cylinder (rare).
    centered = vertices - vertices.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    axes.extend(eigvecs.T)

    # De-duplicate up to sign.
    unique: list[np.ndarray] = []
    for ax in axes:
        if not any(abs(float(ax @ u)) > 0.999 for u in unique):
            unique.append(ax)
    return unique


def _fit_circle_for_facets(
    vertices: np.ndarray,
    faces: np.ndarray,
    facet_ids: list[int],
    axis: np.ndarray,
    normals: np.ndarray,
    config: ConversionConfig,
) -> Cylinder | None:
    """Fit + validate a cylinder for wall facets known to share ``axis``."""
    if len(facet_ids) < 4:
        return None
    vert_ids = np.unique(faces[facet_ids].reshape(-1))
    pts = vertices[vert_ids]
    centroid = pts.mean(axis=0)
    u, v = _plane_basis(axis)
    rel = pts - centroid
    pts2d = np.column_stack((rel @ u, rel @ v))
    center2d, radius, rms = _fit_circle_2d(pts2d)
    if radius <= 0 or rms > config.cylinder_tol:
        return None

    axis_point = centroid + center2d[0] * u + center2d[1] * v

    # Discriminator: facet *centroids* must sit at the fitted radius. This is
    # what separates a real cylinder from e.g. a square's corners (which can be
    # equidistant from a centre yet have centroids well inside that radius).
    fcent = vertices[faces[facet_ids]].mean(axis=1)
    radial_vec = (fcent - axis_point) - ((fcent - axis_point) @ axis)[:, None] * axis
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    if abs(float(radial_dist.mean()) - radius) > 0.05 * radius + 0.05:
        return None
    if float(radial_dist.std()) > 0.05 * radius + 0.05:
        return None

    # Boss vs hole: do the mesh's outward normals point away from the axis
    # (boss, material inside) or toward it (hole, material outside)?
    radial_unit = radial_vec / (np.linalg.norm(radial_vec, axis=1, keepdims=True) + 1e-12)
    outward = bool(np.mean(np.sum(normals[facet_ids] * radial_unit, axis=1)) > 0)

    axial = (pts - axis_point) @ axis
    return Cylinder(
        axis_point=axis_point,
        axis_dir=axis,
        radius=radius,
        axial_min=float(axial.min()),
        axial_max=float(axial.max()),
        rms=rms,
        face_indices=list(facet_ids),
        outward=outward,
    )


def detect_cylinders(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig | None = None,
) -> list[Cylinder]:
    """Find all best-fit cylinders in the mesh.

    For each candidate axis we take the facets whose normal is perpendicular to
    it (the potential wall), cluster them by edge-adjacency into separate
    surfaces, and fit + validate a circle per cluster. Restricting to
    perpendicular facets stops flat faces from bridging unrelated holes.
    """
    config = config or ConversionConfig()
    if not config.detect_cylinders or len(faces) < 8:
        return []

    normals, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for incident in adjacency.values():
        for i in incident:
            for j in incident:
                if i != j:
                    neighbors[i].append(j)

    found: list[Cylinder] = []
    claimed: set[int] = set()
    for axis in _candidate_axes(vertices, faces, normals, areas):
        wall = [
            fi
            for fi in range(len(faces))
            if fi not in claimed and abs(float(normals[fi] @ axis)) < 0.25
        ]
        if len(wall) < 4:
            continue
        for cluster in _connected_components(wall, neighbors):
            cyl = _fit_circle_for_facets(vertices, faces, cluster, axis, normals, config)
            if cyl is None:
                continue
            found.append(cyl)
            claimed.update(cyl.face_indices)
    return found
