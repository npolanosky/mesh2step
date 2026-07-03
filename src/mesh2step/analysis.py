"""Mesh measurement helpers: bounding boxes for import-time inspection.

Pure numpy; no FreeCAD. Used by the worker's ``inspect`` mode so the GUI can
show the user the part's size and offer unit-scaling presets before converting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    """An oriented or axis-aligned box described by its three side lengths."""

    dimensions: np.ndarray  # (3,) side lengths, sorted descending
    volume: float
    # For the oriented box: the rotation whose columns are the box axes, and the
    # box centre (both None for the axis-aligned box, which uses world axes).
    axes: np.ndarray | None = None
    center: np.ndarray | None = None
    # World X/Y/Z extents (axis-aligned box only; None for the oriented box).
    extents_xyz: np.ndarray | None = None

    def as_dict(self) -> dict:
        d = {
            "dimensions": [float(x) for x in self.dimensions],
            "volume": float(self.volume),
        }
        if self.center is not None:
            d["center"] = [float(x) for x in self.center]
        if self.axes is not None:
            d["axes"] = self.axes.tolist()
        if self.extents_xyz is not None:
            d["extents_xyz"] = [float(x) for x in self.extents_xyz]
        return d


def axis_aligned_bbox(vertices: np.ndarray) -> BoundingBox:
    """Axis-aligned bounding box (AABB) in world coordinates."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    dims = hi - lo
    return BoundingBox(
        dimensions=np.sort(dims)[::-1],
        volume=float(np.prod(dims)),
        center=(lo + hi) / 2.0,
        extents_xyz=dims,  # world X, Y, Z extents (unsorted)
    )


def oriented_bbox(vertices: np.ndarray) -> BoundingBox:
    """Approximate minimum oriented bounding box (OBB) via PCA.

    Uses the principal axes of the vertex covariance as the box orientation.
    This is the standard fast approximation — it is exact for box-like parts and
    close to optimal for most others, but is not guaranteed to be the true
    minimum-volume box (that needs rotating-calipers over the convex hull, a
    roadmap item). Good enough to report dimensions and suggest scaling.

    PCA's eigenvectors are only defined up to an arbitrary rotation whenever
    two or more covariance eigenvalues are (numerically) equal — e.g. for a
    cube, or any part symmetric enough to have degenerate principal axes. In
    that case PCA can pick a frame that is rotated 45 degrees relative to the
    part's true faces, giving a needlessly larger box. As a safety net we also
    compute the plain axis-aligned box and fall back to it (reported as an
    oriented box with identity axes) whenever it is no bigger than the
    PCA box. This guarantees oriented_bbox() is never worse than the AABB;
    a true rotating-calipers minimum OBB is still a roadmap item.
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # Principal axes = eigenvectors of the covariance matrix.
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    # Project points onto the principal axes and measure the extent.
    projected = centered @ eigvecs
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    dims = hi - lo
    box_center = centroid + eigvecs @ ((lo + hi) / 2.0)

    order = np.argsort(dims)[::-1]
    pca_dims = dims[order]
    pca_volume = float(np.prod(dims))

    # Safety net: never return a box worse than the plain AABB (see docstring).
    aabb_lo = vertices.min(axis=0)
    aabb_hi = vertices.max(axis=0)
    aabb_dims = aabb_hi - aabb_lo
    aabb_volume = float(np.prod(aabb_dims))

    rel_tol = 1e-9
    if aabb_volume <= pca_volume * (1.0 + rel_tol):
        aabb_order = np.argsort(aabb_dims)[::-1]
        return BoundingBox(
            dimensions=aabb_dims[aabb_order],
            volume=aabb_volume,
            axes=np.eye(3),
            center=(aabb_lo + aabb_hi) / 2.0,
        )

    return BoundingBox(
        dimensions=pca_dims,
        volume=pca_volume,
        axes=eigvecs[:, order],
        center=box_center,
    )


def measure(vertices: np.ndarray) -> dict:
    """Return both bounding boxes plus vertex/triangle independent stats."""
    aabb = axis_aligned_bbox(vertices)
    obb = oriented_bbox(vertices)
    return {
        "aabb": aabb.as_dict(),
        "obb": obb.as_dict(),
        "vertex_count": int(len(vertices)),
    }
