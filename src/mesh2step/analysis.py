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

    def as_dict(self) -> dict:
        d = {
            "dimensions": [float(x) for x in self.dimensions],
            "volume": float(self.volume),
        }
        if self.center is not None:
            d["center"] = [float(x) for x in self.center]
        if self.axes is not None:
            d["axes"] = self.axes.tolist()
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
    )


def oriented_bbox(vertices: np.ndarray) -> BoundingBox:
    """Approximate minimum oriented bounding box (OBB) via PCA.

    Uses the principal axes of the vertex covariance as the box orientation.
    This is the standard fast approximation — it is exact for box-like parts and
    close to optimal for most others, but is not guaranteed to be the true
    minimum-volume box (that needs rotating-calipers over the convex hull, a
    roadmap item). Good enough to report dimensions and suggest scaling.
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
    return BoundingBox(
        dimensions=dims[order],
        volume=float(np.prod(dims)),
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
