"""Analytic surface fitting — ROADMAP / not yet implemented.

The planar pipeline rebuilds flat regions as single STEP faces. The next step
toward true reverse-engineering is detecting curved regions (cylinders, cones,
spheres) and rebuilding them as analytic OCC surfaces instead of facets.

Planned approach
----------------
1. Flag non-planar regions left over after :func:`segment_planar`.
2. For each, estimate the surface type from the distribution of facet normals:
   - normals through a common axis  -> cylinder / cone
   - normals through a common point -> sphere
3. RANSAC-fit the candidate primitive; accept if the inlier ratio and residual
   pass thresholds, else keep the region faceted (or fit a B-spline patch).
4. In ``builder``, emit the analytic surface (``Part.makeCylinder`` etc.,
   trimmed to the region boundary) rather than a polygonal face.

Nothing here is wired into the pipeline yet; these are typed stubs so the shape
of the API is fixed and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import Region


@dataclass
class CylinderFit:
    axis_point: np.ndarray
    axis_dir: np.ndarray
    radius: float
    residual: float


@dataclass
class SphereFit:
    center: np.ndarray
    radius: float
    residual: float


def fit_cylinder(vertices: np.ndarray, faces: np.ndarray, region: Region) -> CylinderFit | None:
    """Fit a cylinder to a region's facets. Not yet implemented."""
    raise NotImplementedError("cylinder fitting is on the roadmap (see DESIGN.md)")


def fit_sphere(vertices: np.ndarray, faces: np.ndarray, region: Region) -> SphereFit | None:
    """Fit a sphere to a region's facets. Not yet implemented."""
    raise NotImplementedError("sphere fitting is on the roadmap (see DESIGN.md)")
