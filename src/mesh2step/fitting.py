"""Cylinder detection and best-fit for clean hole/bore reconstruction.

Faceted cylinders (holes, bores, bosses, fillets) are the single biggest pain
when importing meshes into CAD: hundreds of tiny triangles that tools choke on
and that can't be used as a real hole. This module finds those regions and fits
a best-fit cylinder (axis + radius + axial extent) so the builder can rebuild
them as a single analytic cylindrical face with true circular edges.

Pure numpy; no FreeCAD. The builder consumes the :class:`Cylinder` results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import ConversionConfig
from .boundary import _chain_loops, _directed_boundary_edges
from .segmentation import (
    MeshResolution,
    Region,
    SweptRegion,
    build_edge_adjacency,
    face_normals_and_areas,
    mesh_resolution,
)


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
    coverage: float = 1.0    # fraction of the full circle the facets span (0..1)
    is_fillet: bool = False  # True for a partial-arc straight-edge fillet section
    tangent: bool = False    # True when snapped tangent to its adjacent flats
    radius_source: str = "fit"  # "fit" | "tangency"
    u_start: float = 0.0     # start angle (rad) of the arc, from axis_point basis
    u_span: float = 2.0 * np.pi  # arc span (rad); 2*pi for a full cylinder

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
            "coverage": float(self.coverage),
            "is_fillet": bool(self.is_fillet),
            "tangent": bool(self.tangent),
            "radius_source": self.radius_source,
        }


@dataclass
class Cone:
    """A best-fit (right circular) cone, e.g. a countersink coaxial with a hole."""

    axis_point: np.ndarray   # a point on the axis (the paired cylinder's centre)
    axis_dir: np.ndarray
    r_small: float
    r_large: float
    half_angle_deg: float
    axial_min: float
    axial_max: float
    rms: float
    face_indices: list[int]
    r_base: float = 0.0      # radius at axial_min (for building makeCone)
    r_top: float = 0.0       # radius at axial_max
    outward: bool = False

    def as_dict(self) -> dict:
        return {
            "r_small": float(self.r_small),
            "r_large": float(self.r_large),
            "half_angle_deg": float(self.half_angle_deg),
            "axis_dir": [float(x) for x in self.axis_dir],
            "rms": float(self.rms),
            "facets": len(self.face_indices),
            "role": "countersink",
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
    vertices: np.ndarray, normals: np.ndarray, areas: np.ndarray, max_axes: int = 12
) -> list[np.ndarray]:
    """Directions to try as cylinder axes.

    Holes/bores are usually perpendicular to a flat face, so the normals of the
    largest flat faces (by total area) are strong axis candidates. We add the
    mesh principal axes as a fallback and de-duplicate up to sign.
    """
    weight: dict[tuple, float] = {}
    rep: dict[tuple, np.ndarray] = {}
    for n, a in zip(normals, areas):
        key = tuple(np.round(n, 2))
        if key < tuple(-x for x in key):  # fold antipodal directions together
            key = tuple(-x for x in key)
            n = -n
        weight[key] = weight.get(key, 0.0) + float(a)
        rep.setdefault(key, n)
    ranked = sorted(weight, key=weight.get, reverse=True)
    axes = [rep[k] / (np.linalg.norm(rep[k]) or 1.0) for k in ranked[:max_axes]]

    centered = vertices - vertices.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    axes.extend(eigvecs.T)

    unique: list[np.ndarray] = []
    for ax in axes:
        if not any(abs(float(ax @ u)) > 0.999 for u in unique):
            unique.append(ax)
    return unique


def _region_axis(
    component: list[int], comp_set: set[int], normals: np.ndarray,
    neighbors: list[list[int]], config: ConversionConfig,
) -> np.ndarray | None:
    """Estimate an isolated curved region's axis from its own facet normals.

    On a cylinder/cone wall the surface normals are (near) perpendicular to the
    axis and rotate around it, so the cross product of two *adjacent* normals is
    parallel to the axis. Summing their outer products, the axis is therefore
    the largest-eigenvalue eigenvector, and it must clearly dominate the other
    two (else the region is a sphere/blob whose normals — and their cross
    products — point every which way, with no single axis). Used to seed
    candidate axes for holes drilled at an arbitrary angle.
    """
    # Gate on the *angle* between normals, not the cross-product magnitude: a
    # finely tessellated cylinder steps only a couple of degrees per facet, so a
    # magnitude cut-off would discard every real curvature step. The smooth band
    # (coplanar tol .. curve_max) keeps genuine curvature and drops both flat
    # noise and sharp feature edges; the tiny cross product is then normalised.
    cos_flat = config.angle_tol_cos
    cos_sharp = float(np.cos(np.radians(config.curve_max_deg)))
    M = np.zeros((3, 3))
    count = 0
    for i in component:
        ni = normals[i]
        for j in neighbors[i]:
            if j > i and j in comp_set:
                d = float(ni @ normals[j])
                if cos_sharp < d < cos_flat:  # smooth curvature step
                    c = np.cross(ni, normals[j])
                    nc = float(np.linalg.norm(c))
                    if nc > 1e-9:
                        c /= nc
                        M += np.outer(c, c)
                        count += 1
    if count < 3:
        return None
    eigvals, eigvecs = np.linalg.eigh(M)  # ascending
    # The axis direction must dominate: the second-largest eigenvalue should be
    # well below the largest, otherwise the cross products fill a plane/sphere.
    if eigvals[1] > 0.35 * eigvals[2] + 1e-12:
        return None
    axis = eigvecs[:, 2]
    return axis / (np.linalg.norm(axis) or 1.0)


def _angled_axis_candidates(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
    neighbors: list[list[int]], config: ConversionConfig,
) -> list[np.ndarray]:
    """Extra axes from isolated curved regions (holes at arbitrary angles).

    A facet is 'curved' when at least one edge-neighbour's normal differs by a
    small-to-moderate angle (a smooth surface transition) rather than being
    coplanar (same flat face) or meeting at a sharp feature edge (a flat-face
    boundary). This distinguishes a hole wall from the flat faces around it
    regardless of how few facets each flat face has. The curved facets are then
    grouped into connected regions and each contributes its estimated axis;
    _region_axis returns None for regions with no single axis (organic blobs).
    """
    cos_flat = config.angle_tol_cos
    cos_sharp = float(np.cos(np.radians(config.curve_max_deg)))
    curved: list[int] = []
    for i in range(len(faces)):
        ni = normals[i]
        for j in neighbors[i]:
            d = float(ni @ normals[j])
            if cos_sharp < d < cos_flat:  # smooth curvature, not flat/sharp
                curved.append(i)
                break
    if not curved:
        return []

    # Each connected curved region contributes its axis. _region_axis returns
    # None for regions with no single axis (a free-form/organic blob), so no
    # fraction gate is needed — a plate dominated by one big angled hole is fine.
    axes: list[np.ndarray] = []
    curved_set = set(curved)
    for comp in _connected_components(curved, neighbors):
        if len(comp) < config.min_cylinder_facets:
            continue
        ax = _region_axis(comp, curved_set, normals, neighbors, config)
        if ax is not None:
            axes.append(ax)
    return axes


def _angular_coverage(
    centroids: np.ndarray, axis: np.ndarray, center: np.ndarray, bins: int = 24
) -> float:
    """Fraction of the circle (0..1) the facet centroids span about the axis."""
    u, v = _plane_basis(axis)
    rel = centroids - center
    ang = np.arctan2(rel @ v, rel @ u)
    idx = np.floor((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    return len(set(idx.tolist())) / bins


def _local_tol(config: ConversionConfig, local_edge: float) -> float:
    """Resolution-scaled RMS acceptance: max(abs floor, rel * local edge).

    Chord error on a tessellated curved surface scales with the edge length, so
    a fixed mm tolerance rejects correct fits on coarse meshes. This is the
    accepted RMS-about-fit (roughly half the one-sided chordal sagitta), floored
    by ``curve_fit_tol_abs`` so fine meshes don't over-tighten. Never tighter
    than the legacy ``cylinder_tol`` so exact-radius recovery is unchanged where
    it already worked.
    """
    scaled = max(config.curve_fit_tol_abs, config.curve_fit_tol_rel * float(local_edge))
    return max(scaled, config.cylinder_tol)


def _fit_circle_for_facets(
    vertices: np.ndarray,
    faces: np.ndarray,
    facet_ids: list[int],
    axis: np.ndarray,
    normals: np.ndarray,
    max_radius: float,
    config: ConversionConfig,
    resolution: MeshResolution | None = None,
) -> Cylinder | None:
    """Fit + validate a cylinder for wall facets known to share ``axis``.

    Tolerances scale with the band's local edge length (``resolution``) so a
    coarse-mesh cylinder whose chordal sagitta exceeds the absolute ``cylinder_tol``
    is still accepted, while the coverage/centroid/RMS guards keep false
    positives out. With ``resolution=None`` the legacy absolute tolerances apply.
    """
    if len(facet_ids) < config.min_cylinder_facets:
        return None
    vert_ids = np.unique(faces[facet_ids].reshape(-1))
    pts = vertices[vert_ids]
    centroid = pts.mean(axis=0)
    u, v = _plane_basis(axis)
    rel = pts - centroid
    pts2d = np.column_stack((rel @ u, rel @ v))
    center2d, radius, rms = _fit_circle_2d(pts2d)

    if resolution is not None:
        local_edge = resolution.edge_for(facet_ids)
        tol = _local_tol(config, local_edge)
        # The centroid of a chordal facet sits inside the true circle by the
        # sagitta ~edge^2/(8R); scale the centroid-radius guard by that (NOT
        # linearly in edge, which would be far too loose on big flat faces and
        # let a square's corners pass as a giant circle) plus a small relative
        # term for fit/rounding noise. Cap the sagitta contribution at a strict
        # fraction of the radius: a genuine coarse fit has sagitta/R well under
        # this (edge<~0.7R), while a square's four corners fitting a huge circle
        # miss by ~29% and must still be rejected.
        sagitta = (local_edge * local_edge) / (8.0 * radius) if radius > 1e-9 else 0.0
        guard = 0.05 * radius + min(max(0.05, 1.5 * sagitta), 0.08 * radius)
    else:
        local_edge = 0.0
        tol = config.cylinder_tol
        guard = 0.05 * radius + 0.05

    if radius <= 0 or rms > tol:
        return None
    # Reject sub-millimetre "cylinders": tiny curved facet clusters on an organic
    # surface algebraically fit a near-zero-radius circle, producing dozens of
    # bogus micro-holes. Real holes/bosses are well above this.
    if radius < config.min_cylinder_radius:
        return None
    # A hole/boss can't be larger than the part (kills shallow-arc mega-circles).
    if radius > max_radius:
        return None

    axis_point = centroid + center2d[0] * u + center2d[1] * v

    # Discriminator: facet *centroids* must sit at the fitted radius. This is
    # what separates a real cylinder from e.g. a square's corners (which can be
    # equidistant from a centre yet have centroids well inside that radius). The
    # centroid of a chordal facet sits at radius*cos(half-arc) < radius, so on a
    # coarse mesh the mean centroid radius reads low by ~edge^2/(8R); the guard is
    # edge-scaled to admit that without letting a corner cluster through.
    fcent = vertices[faces[facet_ids]].mean(axis=1)
    radial_vec = (fcent - axis_point) - ((fcent - axis_point) @ axis)[:, None] * axis
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    if abs(float(radial_dist.mean()) - radius) > guard:
        return None
    if float(radial_dist.std()) > guard:
        return None

    # The facets must wrap a meaningful arc; a sliver spanning a few degrees can
    # still pass an algebraic circle fit but is not a real cylinder.
    coverage = _angular_coverage(fcent, axis, axis_point)
    if coverage < config.min_cylinder_coverage:
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
        coverage=coverage,
    )


def detect_cylinders(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig | None = None,
) -> list[Cylinder]:
    """Find all best-fit cylinders in the mesh.

    Curved facets (those with a non-coplanar edge-neighbour) are grouped into
    connected regions. Each region estimates its *own* axis from its facet
    normals — so holes at any angle are found, not just those perpendicular to a
    flat face — then the wall facets (normals perpendicular to that axis) are
    fitted to a best-fit circle and validated.
    """
    config = config or ConversionConfig()
    if not config.detect_cylinders or len(faces) < config.min_cylinder_facets:
        return []

    resolution = mesh_resolution(vertices, faces, config)
    normals, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for incident in adjacency.values():
        for i in incident:
            for j in incident:
                if i != j:
                    neighbors[i].append(j)

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    max_radius = config.max_cylinder_radius or float(extent.max())

    # For each candidate axis, take the facets whose normal is perpendicular to
    # it (the potential wall), split those into separate surfaces by
    # connectivity, and fit + validate a circle per surface. Restricting to
    # perpendicular facets keeps flat faces from bridging unrelated holes, and
    # works even on organic meshes where nearly every facet is "curved".
    axes = _candidate_axes(vertices, normals, areas, config.max_candidate_axes)
    if config.detect_angled:
        axes += _angled_axis_candidates(vertices, faces, normals, neighbors, config)
    # De-duplicate up to sign so an angled axis that coincides with a flat-normal
    # one isn't fitted twice.
    unique_axes: list[np.ndarray] = []
    for ax in axes:
        if not any(abs(float(ax @ u)) > 0.999 for u in unique_axes):
            unique_axes.append(ax)

    found: list[Cylinder] = []
    claimed: set[int] = set()
    for axis in unique_axes:
        wall = [
            fi for fi in range(len(faces))
            if fi not in claimed and abs(float(normals[fi] @ axis)) < 0.25
        ]
        if len(wall) < config.min_cylinder_facets:
            continue
        for cluster in _connected_components(wall, neighbors):
            cyl = _fit_circle_for_facets(
                vertices, faces, cluster, axis, normals, max_radius, config,
                resolution=resolution,
            )
            if cyl is not None:
                found.append(cyl)
                claimed.update(cyl.face_indices)

    if config.harmonize_radii:
        _harmonize_radii(found, config)
    return found


def detect_cones(
    vertices: np.ndarray,
    faces: np.ndarray,
    cylinders: list[Cylinder],
    config: ConversionConfig | None = None,
) -> list[Cone]:
    """Detect cones coaxial with detected cylinders (countersinks/chamfers).

    A cone's surface normals make a constant non-zero angle with the axis
    (|n·axis| = sin(half-angle)), and its radius grows linearly along the axis.
    We look, around each cylinder's axis, for a connected ring of such facets
    and fit radius-vs-axial to recover the half-angle and end radii.
    """
    config = config or ConversionConfig()
    if not config.detect_cylinders or not cylinders:
        return []

    normals, _ = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for incident in adjacency.values():
        for i in incident:
            for j in incident:
                if i != j:
                    neighbors[i].append(j)

    centroids = vertices[faces].mean(axis=1)
    ndot_axis = np.empty(len(faces))
    claimed: set[int] = set(i for c in cylinders for i in c.face_indices)
    cones: list[Cone] = []

    for cyl in cylinders:
        axis, p = cyl.axis_dir, cyl.axis_point
        rel = centroids - p
        axial = rel @ axis
        rho = np.linalg.norm(rel - axial[:, None] * axis, axis=1)
        np.abs(normals @ axis, out=ndot_axis)
        # Cone facets: normal tilted (not perpendicular, not flat), coaxial ring
        # a bit wider than the bore, close to it axially.
        mask = (
            (ndot_axis > 0.2) & (ndot_axis < 0.98)
            & (rho > 0.6 * cyl.radius) & (rho < 6.0 * cyl.radius)
            & (np.abs(axial - cyl.axial_min) < 4 * cyl.radius)
        ) | (
            (ndot_axis > 0.2) & (ndot_axis < 0.98)
            & (rho > 0.6 * cyl.radius) & (rho < 6.0 * cyl.radius)
            & (np.abs(axial - cyl.axial_max) < 4 * cyl.radius)
        )
        candidates = [i for i in np.where(mask)[0] if i not in claimed]
        if len(candidates) < config.min_cylinder_facets:
            continue
        for cluster in _connected_components(candidates, neighbors):
            cone = _fit_cone(cluster, cyl, axis, p, vertices, faces, centroids, config)
            if cone is not None:
                cones.append(cone)
                claimed.update(cone.face_indices)
    return cones


def _fit_cone(cluster, cyl, axis, p, vertices, faces, centroids, config) -> Cone | None:
    if len(cluster) < config.min_cylinder_facets:
        return None
    # Fit on the cluster's *vertices* (not facet centroids) so the cone reaches
    # the true edges (mouth at the surface, junction at the bore).
    vert_ids = np.unique(faces[cluster].reshape(-1))
    pts = vertices[vert_ids]
    rel = pts - p
    z = rel @ axis
    r = np.linalg.norm(rel - z[:, None] * axis, axis=1)
    # Linear radius-vs-axial fit: r = slope*z + intercept; slope = tan(half-angle).
    A = np.column_stack((z, np.ones_like(z)))
    (slope, intercept), *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = r - (slope * z + intercept)
    rms = float(np.sqrt(np.mean(resid**2)))
    if abs(slope) < 0.15 or rms > 3 * config.cylinder_tol:
        return None  # too flat to be a cone, or not a clean cone
    if _angular_coverage(centroids[cluster], axis, p) < config.min_cylinder_coverage:
        return None

    zmin, zmax = float(z.min()), float(z.max())
    # Boss vs hole, exactly as for cylinders: outward normals pointing away from
    # the axis mean material inside (a tapered boss/neck); pointing toward it
    # mean a countersink-style hole. Getting this wrong turns the boolean
    # clean-up destructive — cutting a boss cone carves the boss off the part.
    tri = vertices[faces[cluster]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    crel = centroids[cluster] - p
    radial = crel - (crel @ axis)[:, None] * axis
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    outward = bool(np.mean(np.sum(n * radial, axis=1)) > 0)

    # Snap the cone's near end exactly to the paired cylinder's end circle, so
    # the two analytic faces share an identical junction edge and sew closed.
    junction_z = min((cyl.axial_min, cyl.axial_max),
                     key=lambda ze: min(abs(zmin - ze), abs(zmax - ze)))
    far_z = zmax if abs(zmax - junction_z) >= abs(zmin - junction_z) else zmin
    intercept = cyl.radius - slope * junction_z  # force line through the junction
    r_far = slope * far_z + intercept
    if r_far <= 0:
        return None
    axial_min, axial_max = sorted((junction_z, far_z))
    r_base = slope * axial_min + intercept
    r_top = slope * axial_max + intercept
    return Cone(
        axis_point=p, axis_dir=axis,
        r_small=min(cyl.radius, r_far), r_large=max(cyl.radius, r_far),
        half_angle_deg=float(np.degrees(np.arctan(abs(slope)))),
        axial_min=axial_min, axial_max=axial_max, rms=rms,
        face_indices=list(cluster),
        r_base=float(r_base), r_top=float(r_top),
        outward=outward,
    )


def _harmonize_radii(cylinders: list[Cylinder], config: ConversionConfig) -> None:
    """Snap near-equal radii to a shared, rounded value (in place).

    Groups radii that are within ``harmonize_rel_tol`` of each other, then sets
    every member to the group's facet-count-weighted mean rounded to the
    ``harmonize_round`` grid — so triangulation noise (6.04/6.05/6.06) collapses
    to one clean radius.
    """
    if not cylinders:
        return
    order = sorted(range(len(cylinders)), key=lambda i: cylinders[i].radius)
    grid = config.harmonize_round
    groups: list[list[int]] = []
    for i in order:
        r = cylinders[i].radius
        if groups and abs(r - cylinders[groups[-1][-1]].radius) <= config.harmonize_rel_tol * r:
            groups[-1].append(i)
        else:
            groups.append([i])
    for group in groups:
        weights = np.array([len(cylinders[i].face_indices) for i in group], dtype=float)
        radii = np.array([cylinders[i].radius for i in group])
        mean_r = float(np.average(radii, weights=weights))
        snapped = round(mean_r / grid) * grid if grid > 0 else mean_r
        for i in group:
            cylinders[i].radius = snapped


# --------------------------------------------------------------------------- #
# Straight-edge fillets (plane-plane blends) — design §1.2, §1.3, §2, §3.
# --------------------------------------------------------------------------- #


def _plane_intersection_line(
    p1: np.ndarray, n1: np.ndarray, p2: np.ndarray, n2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Line where two planes meet: returns (point_on_line, unit_direction).

    ``p*``/``n*`` are a point and unit normal of each plane. Returns None when
    the planes are (near) parallel.
    """
    d = np.cross(n1, n2)
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        return None
    d = d / dn
    # Solve for a point on both planes: [n1; n2; d] x = [n1.p1; n2.p2; 0].
    A = np.vstack([n1, n2, d])
    rhs = np.array([float(n1 @ p1), float(n2 @ p2), 0.0])
    try:
        pt = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    return pt, d


def _tangent_axis_for_radius(
    line_pt: np.ndarray, bisector: np.ndarray, half_theta: float, radius: float,
) -> np.ndarray:
    """Axis point of a cylinder of ``radius`` tangent to both planes.

    The axis lies on the bisector plane at distance ``R / sin(theta/2)`` from the
    edge line (design §1.3).
    """
    s = float(np.sin(half_theta))
    if s < 1e-6:
        return line_pt.copy()
    return line_pt + bisector * (radius / s)


def measure_tangency_defect(
    axis: np.ndarray,
    axis_point: np.ndarray,
    radius: float,
    plane_normals: list[np.ndarray],
    plane_points: list[np.ndarray],
) -> float:
    """Tangency defect (deg) of a cylinder against adjacent planes (design §1.2).

    For a cylinder truly tangent to a plane, the point on the plane closest to
    the axis lies at exactly ``radius`` from the axis, and the surface normal
    there is parallel to the plane normal. We measure how far the plane sits from
    that ideal: the angle whose sine is ``(dist_axis_to_plane - radius) /
    radius`` — 0 deg at a true tangent blend. Returns the max defect over the
    adjacent planes (the worst tangency).
    """
    axis = axis / (np.linalg.norm(axis) or 1.0)
    worst = 0.0
    for pn, pp in zip(plane_normals, plane_points):
        pn = pn / (np.linalg.norm(pn) or 1.0)
        # Signed distance from the axis point to the plane, along the plane normal.
        dist = abs(float((axis_point - pp) @ pn))
        if radius <= 1e-9:
            continue
        # A tangent cylinder has dist == radius. Convert the shortfall/excess to
        # an angular defect relative to the radius.
        frac = min(1.0, abs(dist - radius) / radius)
        defect = float(np.degrees(np.arcsin(frac)))
        worst = max(worst, defect)
    return worst


def tangency_threshold_deg(config: ConversionConfig, resolution: MeshResolution) -> float:
    """Resolution-scaled near/far tangency threshold (design §1.2)."""
    return max(config.tangency_floor_deg,
               config.tangency_k * resolution.median_dihedral_deg)


def _fit_fillet_between_planes(
    vertices: np.ndarray,
    faces: np.ndarray,
    band_faces: list[int],
    r1: Region,
    r2: Region,
    normals: np.ndarray,
    config: ConversionConfig,
    resolution: MeshResolution,
) -> Cylinder | None:
    """Fit a partial-arc cylinder to a fillet band between two planar regions.

    The candidate axis is the two planes' intersection-edge direction (which
    ``_candidate_axes`` never generates). A free circle fit in the plane
    perpendicular to that edge gives a rough radius; if the band is near-tangent
    to both flats the radius is re-derived from the tangency constraint (exact
    planes, chord-bias-free), otherwise the free fit radius is kept.
    """
    if len(band_faces) < config.min_cylinder_facets:
        return None
    n1 = r1.plane_normal / (np.linalg.norm(r1.plane_normal) or 1.0)
    n2 = r2.plane_normal / (np.linalg.norm(r2.plane_normal) or 1.0)
    line = _plane_intersection_line(r1.plane_point, n1, r2.plane_point, n2)
    if line is None:
        return None
    line_pt, axis = line

    vert_ids = np.unique(faces[band_faces].reshape(-1))
    pts = vertices[vert_ids]

    # Free circle fit in the plane perpendicular to the edge direction.
    u, v = _plane_basis(axis)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    pts2d = np.column_stack((rel @ u, rel @ v))
    center2d, r_fit, rms_free = _fit_circle_2d(pts2d)
    if r_fit <= 0:
        return None

    max_radius = config.max_cylinder_radius or float(
        (vertices.max(axis=0) - vertices.min(axis=0)).max())
    if r_fit > max_radius:
        return None

    # Interior dihedral angle theta between the two flats. The fillet fills the
    # interior corner, so its axis is offset along the interior bisector.
    cos_between = float(np.clip(n1 @ n2, -1.0, 1.0))
    # theta is the interior dihedral (angle of the material wedge). For outward
    # normals, the exterior angle between normals is (pi - theta), so
    # theta = pi - angle(n1, n2).
    theta = np.pi - float(np.arccos(cos_between))
    half_theta = theta / 2.0
    if half_theta < np.radians(5) or half_theta > np.radians(85):
        return None

    # Bisector candidates (into the material): try both signs and both radius
    # sources, keep whichever gives the lowest vertex residual to the tangent
    # cylinder. The 1-D radius search is bracketed around the free-fit radius.
    line_pt0 = line_pt + axis * float((centroid - line_pt) @ axis)  # nearest to band

    def residual_for(radius: float, sign: float) -> tuple[float, np.ndarray]:
        bis = sign * (-(n1 + n2))
        bn = float(np.linalg.norm(bis))
        if bn < 1e-9:
            return float("inf"), line_pt0
        bis /= bn
        ap = _tangent_axis_for_radius(line_pt0, bis, half_theta, radius)
        d = pts - ap
        radial = d - np.outer(d @ axis, axis)
        rr = np.linalg.norm(radial, axis=1)
        resid = rr - radius
        return float(np.sqrt(np.mean(resid ** 2))), ap

    best = None  # (rms, radius, axis_point, sign)
    for sign in (1.0, -1.0):
        # 1-D search over radius near the free fit (design: minimize vertex
        # residual to the tangency-constrained cylinder).
        lo, hi = 0.4 * r_fit, 1.8 * r_fit
        grid = np.linspace(lo, hi, 33)
        rmss = [residual_for(float(rg), sign)[0] for rg in grid]
        k = int(np.argmin(rmss))
        # Refine around the best grid point.
        r_lo = grid[max(0, k - 1)]
        r_hi = grid[min(len(grid) - 1, k + 1)]
        fine = np.linspace(r_lo, r_hi, 21)
        for rg in fine:
            rms, ap = residual_for(float(rg), sign)
            if best is None or rms < best[0]:
                best = (rms, float(rg), ap, sign)
    if best is None:
        return None
    rms_tan, r_tan, ap_tan, sign = best

    local_edge = resolution.edge_for(band_faces)
    tol = _local_tol(config, local_edge)

    # Free-fit axis point (in 3D) for defect measurement and the non-tangent case.
    free_axis_point = centroid + center2d[0] * u + center2d[1] * v

    # Near/far tangency decision (design §1.2): measure the free fit's tangency
    # defect against the two adjacent flats; if within the resolution-scaled
    # threshold, tangency is design intent — snap and derive R from the exact
    # planes (chord-bias free). Otherwise keep the best-effort free fit radius.
    defect = measure_tangency_defect(
        axis, free_axis_point, r_fit,
        [n1, n2], [r1.plane_point, r2.plane_point])
    thresh_deg = tangency_threshold_deg(config, resolution)

    # Coverage: a straight fillet is a partial arc. Compute it about the chosen
    # axis so we can gate on the fillet-specific coverage window.
    fcent = vertices[faces[band_faces]].mean(axis=1)

    tangent = defect <= thresh_deg and rms_tan <= tol
    if tangent:
        radius, rms, radius_source = r_tan, rms_tan, "tangency"
        axis_point = ap_tan
    else:
        # Far from tangent: best-effort free fit, no snap.
        if rms_free > tol:
            return None
        radius = r_fit
        rms = rms_free
        radius_source = "fit"
        axis_point = free_axis_point
    coverage = _angular_coverage(fcent, axis, axis_point)

    # Resolution-scaled minimum radius: sub-facet fillets can't be built cleanly.
    min_r = max(config.min_fillet_radius, config.min_fillet_radius_edges * local_edge)
    if radius < min_r or radius > max_radius:
        return None

    # Coverage window: a fillet is a genuine partial arc. Too little => sliver
    # noise; too much => it's a real cylinder detect_cylinders should own.
    if coverage < config.min_fillet_coverage or coverage > config.max_fillet_coverage:
        return None

    # Boss vs hole (convex outer edge vs concave inner corner): outward normals
    # pointing away from the axis => convex/boss (fuse); toward => concave (cut).
    d = pts - axis_point
    radial = d - np.outer(d @ axis, axis)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    # Use facet normals for orientation.
    fn = normals[band_faces]
    fc = fcent - axis_point
    fr = fc - np.outer(fc @ axis, axis)
    fr /= np.linalg.norm(fr, axis=1, keepdims=True) + 1e-12
    outward = bool(np.mean(np.sum(fn * fr, axis=1)) > 0)

    # Arc extent (u_start, u_span) about the axis for a trimmed cylinder face.
    rel_c = fcent - axis_point
    ang = np.arctan2(rel_c @ v, rel_c @ u)
    u_start, u_span = _arc_span(ang)

    axial = (pts - axis_point) @ axis
    return Cylinder(
        axis_point=axis_point,
        axis_dir=axis,
        radius=float(radius),
        axial_min=float(axial.min()),
        axial_max=float(axial.max()),
        rms=float(rms),
        face_indices=list(band_faces),
        outward=outward,
        coverage=coverage,
        is_fillet=True,
        tangent=tangent,
        radius_source=radius_source,
        u_start=float(u_start),
        u_span=float(u_span),
    )


def _arc_span(ang: np.ndarray) -> tuple[float, float]:
    """Smallest arc (start, span) in radians covering all angles in ``ang``.

    Finds the largest angular gap between consecutive sorted angles; the arc is
    the complement of that gap.
    """
    a = np.sort(np.mod(ang, 2 * np.pi))
    if a.size < 2:
        return 0.0, 2 * np.pi
    gaps = np.diff(a)
    wrap = (a[0] + 2 * np.pi) - a[-1]
    all_gaps = np.append(gaps, wrap)
    k = int(np.argmax(all_gaps))
    span = 2 * np.pi - float(all_gaps[k])
    # The arc starts just after the largest gap.
    start = float(a[(k + 1) % a.size])
    return start, span


def detect_fillets_straight(
    vertices: np.ndarray,
    faces: np.ndarray,
    smooth_bands,
    regions: list[Region],
    claimed: set[int],
    config: ConversionConfig | None = None,
    resolution: MeshResolution | None = None,
) -> list[Cylinder]:
    """Detect straight-edge fillets as partial-arc cylinders (design §2, §3).

    Driven from the ``band``-classed smooth regions (each bordering exactly two
    planar regions): the candidate axis is the two planes' intersection-edge
    direction, and the radius is tangency-derived when the band is near-tangent
    to both flats. Returns fillet ``Cylinder`` objects tagged ``is_fillet``.
    """
    config = config or ConversionConfig()
    if not config.detect_fillets:
        return []
    if resolution is None:
        resolution = mesh_resolution(vertices, faces, config)
    normals, _ = face_normals_and_areas(vertices, faces)

    fillets: list[Cylinder] = []
    for band in smooth_bands:
        if band.class_hint != "band" or len(band.border_regions) != 2:
            continue
        if any(fi in claimed for fi in band.face_indices):
            band_faces = [fi for fi in band.face_indices if fi not in claimed]
        else:
            band_faces = list(band.face_indices)
        if len(band_faces) < config.min_cylinder_facets:
            continue
        r1 = regions[band.border_regions[0]]
        r2 = regions[band.border_regions[1]]
        cyl = _fit_fillet_between_planes(
            vertices, faces, band_faces, r1, r2, normals, config, resolution)
        if cyl is not None:
            fillets.append(cyl)
            claimed.update(cyl.face_indices)
    return fillets


# --------------------------------------------------------------------------- #
# Swept / extruded curved walls (Milestone 4). Given a SweptRegion (a set of
# planar strips whose normals are all perpendicular to a common extrusion axis),
# recover the 2D profile curve so the builder can extrude it into a single
# analytic/B-spline face instead of a fan of thin planar strips.
# --------------------------------------------------------------------------- #


@dataclass
class ProfileSegment:
    """One fitted piece of a swept profile, in 2D profile-plane coordinates.

    ``kind`` is ``"line"``, ``"arc"``, or ``"spline"``.
    Lines: ``p0``, ``p1`` endpoints (2,).
    Arcs: ``p0``, ``p1`` endpoints plus ``center`` (2,) and ``radius``, with
    ``ccw`` giving the sweep sense.
    Splines: ``points`` (k,2) interpolation points.
    """

    kind: str
    p0: np.ndarray | None = None
    p1: np.ndarray | None = None
    center: np.ndarray | None = None
    radius: float = 0.0
    ccw: bool = True
    points: np.ndarray | None = None
    tangent_start: bool = False   # snapped tangent to the previous segment
    tangent_end: bool = False
    # Builder annotations (set by the builder from the mesh, arcs only):
    # ``outward``: facet normals point away from the arc centre (convex wall —
    # facets inscribed, fuse the sliver) vs toward it (concave — overshoot, cut).
    # ``covered``: the segment's parametric rectangle (arc span x axial extent)
    # is fully covered by facets — False when cutouts pierce the wall, which
    # makes a fuse unsafe (it would bridge the holes).
    outward: bool | None = None
    covered: bool = True


@dataclass
class SweptProfile:
    """A fitted swept wall: an ordered list of 2D profile segments in the plane
    perpendicular to ``axis``, plus the plane basis and axial extent so the
    builder can lift each 2D point back to 3D and extrude.
    """

    axis: np.ndarray            # (3,) unit extrusion direction
    origin: np.ndarray          # (3,) plane origin (a point on the min-axial rail)
    e1: np.ndarray              # (3,) profile-plane basis vector 1
    e2: np.ndarray              # (3,) profile-plane basis vector 2
    segments: list[ProfileSegment]
    axial_min: float
    axial_max: float
    closed: bool
    rms: float                  # RMS of the fit to the rail points (mm)
    face_indices: list[int]
    n_arcs: int = 0
    n_lines: int = 0
    n_splines: int = 0
    tangency_snaps: int = 0
    member_regions: list[int] = field(default_factory=list)

    def point3d(self, p2: np.ndarray) -> np.ndarray:
        return self.origin + float(p2[0]) * self.e1 + float(p2[1]) * self.e2

    def as_dict(self) -> dict:
        return {
            "axis": [float(x) for x in self.axis],
            "extent": float(self.axial_max - self.axial_min),
            "segments": len(self.segments),
            "lines": self.n_lines, "arcs": self.n_arcs, "splines": self.n_splines,
            "tangency_snaps": self.tangency_snaps,
            "closed": self.closed, "rms": float(self.rms),
            "facets": len(self.face_indices),
        }


def _swept_plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _plane_basis(axis)


def _extract_rails(
    vertices: np.ndarray, faces: np.ndarray, face_indices: list[int], axis: np.ndarray,
) -> tuple[list[tuple[np.ndarray, bool]], float, float] | None:
    """Extract the swept region's profile rails as ordered 3D polylines.

    Every boundary loop of the region lies ON the swept surface, so any
    contiguous run of *profile-like* boundary edges (edge direction
    perpendicular to the sweep axis) projects onto the true profile curve —
    regardless of the run's height along the axis, because the profile is
    height-invariant on a sweep. Classifying by edge direction (rather than
    keeping only the lower-axial half of each loop) matters on real parts:
    a wall's boundary weaves up and over crossing features (e.g. a shelf
    joining mid-height), and a height cut-off would fragment or drop exactly
    the corner-blend arcs those junctions carry. Side edges (running along the
    axis) separate the runs.

    A loop consisting entirely of profile-like edges is a closed rail (a tube
    profile / a wall end at constant height). Rails describing the same curve
    at two heights fit identical segments; the builder de-duplicates ops.

    Returns ``([(rail_pts (k,3), closed), ...], axial_min, axial_max)`` over
    the region, or ``None`` when no boundary exists.
    """
    edges = _directed_boundary_edges(faces, list(face_indices))
    if not edges:
        return None
    loops = _chain_loops(edges)
    if not loops:
        return None
    vids = np.unique(faces[np.asarray(list(face_indices))].reshape(-1))
    ax_all = vertices[vids] @ axis
    amin, amax = float(ax_all.min()), float(ax_all.max())
    if amax - amin < 1e-9:
        return None

    rails: list[tuple[np.ndarray, bool]] = []
    for loop in loops:
        n = len(loop)
        if n < 3:
            continue
        P = vertices[np.asarray(loop)]
        E = P[(np.arange(n) + 1) % n] - P
        elen = np.linalg.norm(E, axis=1)
        ok = elen > 1e-12
        along = np.zeros(n)
        along[ok] = np.abs(E[ok] @ axis) / elen[ok]
        prof_like = ok & (along < 0.35)
        if prof_like.all():
            rails.append((P, True))
            continue
        if not prof_like.any():
            continue
        # Maximal cyclic runs of consecutive profile-like edges; edge i joins
        # vertex i to i+1, so a run of edges s..e yields vertices s..e+1.
        starts = [i for i in range(n) if prof_like[i] and not prof_like[(i - 1) % n]]
        for s in starts:
            run_v = [loop[s]]
            i = s
            while prof_like[i % n] and len(run_v) <= n:
                run_v.append(loop[(i + 1) % n])
                i += 1
            if len(run_v) >= 3:
                rails.append((vertices[np.asarray(run_v)], False))
    if not rails:
        return None
    return rails, amin, amax


def _resample_polyline(pts2d: np.ndarray, closed: bool) -> np.ndarray:
    """Drop near-duplicate consecutive points from a 2D polyline.

    A rail run can include a stub of the wall's vertical side edge; its points
    all project onto (nearly) the same 2D spot, so consecutive points closer
    than a micron-to-milli scale epsilon collapse to one.
    """
    out = [pts2d[0]]
    for p in pts2d[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-3:
            out.append(p)
    arr = np.asarray(out)
    if closed and len(arr) > 1 and np.linalg.norm(arr[0] - arr[-1]) < 1e-3:
        arr = arr[:-1]
    return arr


def _line_reach(pts2d: np.ndarray, i0: int, line_tol: float) -> int:
    """Furthest index a straight run from ``i0`` stays within ``line_tol``."""
    n = len(pts2d)
    j = i0 + 1
    while j + 1 < n:
        seg = pts2d[j + 1] - pts2d[i0]
        L = np.linalg.norm(seg)
        if L < 1e-9:
            j += 1
            continue
        u = seg / L
        rel = pts2d[i0:j + 2] - pts2d[i0]
        perp = np.abs(rel[:, 0] * (-u[1]) + rel[:, 1] * u[0])
        if perp.max() > line_tol:
            break
        j += 1
    return j


def _segment_profile(
    pts2d: np.ndarray, config: ConversionConfig, local_edge: float,
) -> list[tuple[str, int, int]]:
    """Partition an ordered 2D polyline into runs of ``line`` / ``arc`` / ``spline``.

    Line-first, curvature-gated: at each position we take the longest straight
    run that stays within ``line_tol``. Only when a straight run cannot even
    reach the next point (the profile genuinely bends there) do we try to fit a
    circular arc — and accept it only if it has a bounded radius (real curvature,
    not a near-line mega-circle) and covers >= 3 points. Everything else becomes a
    short ``spline`` run. Returns inclusive ``(kind, i0, i1)`` spans.

    A line is preferred over an arc because a straight segment is algebraically a
    circle of huge radius; a greedy arc-first walk would otherwise swallow the
    whole profile as one bogus arc (seen on the ground-truth line+arc+line wall).
    """
    n = len(pts2d)
    runs: list[tuple[str, int, int]] = []
    # Rail vertices sit ON the true profile curve (they are mesh vertices, not
    # chord midpoints), so the collinearity tolerance is weld/decimation-noise
    # scale — NOT the chord-error scale used for surface fits. A loose (edge-
    # scaled) tolerance would let a straight run swallow the start of an
    # adjacent arc and mis-place the junction (seen on the ground-truth wall,
    # whose coarse strips pushed the edge-scaled tolerance to millimetres).
    line_tol = float(config.swept_profile_tol_abs)
    # Cap arc radius so a nearly-straight run is not accepted as a giant arc.
    span = float(np.linalg.norm(pts2d.max(axis=0) - pts2d.min(axis=0)))
    max_arc_radius = 4.0 * span + 1.0
    i = 0
    while i < n - 1:
        jline = _line_reach(pts2d, i, line_tol)
        # Always evaluate the arc too: on a tessellated arc every 2-3 consecutive
        # chords also pass the line test (their sagitta is below tolerance), so a
        # line-only-first rule would shred the arc into mini-lines. The arc wins
        # when it explains materially more of the polyline than the line.
        ka, radius = _extend_arc(pts2d, i, config, max_arc_radius)
        if ka - i >= 3 and radius <= max_arc_radius and ka >= jline + 2:
            runs.append(("arc", i, ka))
            i = ka
            continue
        if jline > i + 1 or (jline == i + 1
                             and np.linalg.norm(pts2d[jline] - pts2d[i]) > line_tol):
            runs.append(("line", i, jline))
            i = jline
            continue
        # Neither a clean line nor arc: accumulate a spline run until a straight
        # or arc run becomes possible again.
        k = i + 1
        while k < n - 1:
            if _line_reach(pts2d, k, line_tol) > k + 1:
                break
            ka2, _ = _extend_arc(pts2d, k, config, max_arc_radius)
            if ka2 - k >= 3:
                break
            k += 1
        runs.append(("spline", i, max(k, i + 1)))
        i = max(k, i + 1)
    return runs


def _extend_arc(
    pts2d: np.ndarray, i0: int, config: ConversionConfig, max_radius: float,
) -> tuple[int, float]:
    """Greedily extend a circular arc from ``i0``; return ``(i_end, radius)``.

    Grows the run while the circle fit RMS stays within tolerance AND the radius
    stays below ``max_radius`` (a near-straight run fits a huge circle and must be
    rejected in favour of a line). Rail points sit ON the true circle, so the
    accepted RMS is noise-scale — ``swept_profile_tol_abs`` plus a *small*
    relative term (``swept_arc_tol_rel`` of the radius); a chord-error-sized
    relative tolerance would let one giant circle "fit" a whole line+arc+line
    profile. ``radius`` is the last accepted fit's radius (``inf`` if none).
    """
    n = len(pts2d)
    best_end = i0
    best_radius = float("inf")
    j = i0 + 3
    while j < n:
        seg = pts2d[i0:j + 1]
        center, radius, rms = _fit_circle_2d(seg)
        tol = config.swept_profile_tol_abs + config.swept_arc_tol_rel * radius
        if radius <= 1e-6 or radius > max_radius or rms > tol:
            break
        # Chord-uniformity guard: a genuine tessellated arc has roughly evenly
        # spaced vertices; a run that swallowed a long straight segment has one
        # huge chord among small ones (a circle passes near-exactly through a
        # handful of sparse points, so RMS alone cannot catch this).
        chords = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        med = float(np.median(chords))
        if med > 1e-9 and float(chords.max()) > 4.0 * med:
            break
        best_end = j
        best_radius = float(radius)
        j += 1
    return best_end, best_radius


def _fit_line(seg: np.ndarray) -> ProfileSegment:
    return ProfileSegment(kind="line", p0=seg[0].copy(), p1=seg[-1].copy())


def _fit_arc(seg: np.ndarray) -> ProfileSegment | None:
    center, radius, _ = _fit_circle_2d(seg)
    if radius <= 1e-6:
        return None
    p0, p1 = seg[0], seg[-1]
    mid = seg[len(seg) // 2]
    # Sweep sense from the cross product of (p0-center) x (mid-center).
    v0 = p0 - center
    vm = mid - center
    ccw = bool((v0[0] * vm[1] - v0[1] * vm[0]) > 0)
    return ProfileSegment(kind="arc", p0=p0.copy(), p1=p1.copy(),
                          center=center.copy(), radius=float(radius), ccw=ccw)


def _apply_tangency(
    segments: list[ProfileSegment], resolution: MeshResolution, config: ConversionConfig,
) -> int:
    """Snap arc<->line joins to exact tangency in 2D (design §1.2, product-owner
    rule applied to the profile plane).

    At a join where a line meets an arc, tangency means the line is perpendicular
    to the arc's centre->join radius. If the measured angle defect is within the
    resolution-scaled threshold, the join is moved to the arc's *true tangent
    point with the line* — the foot of the perpendicular from the arc centre onto
    the line, radially projected onto the circle. Besides making the join exactly
    tangent, this *extends the arc back to where the blend really starts*: the
    greedy line run steals the first chord or two of a tessellated arc (their
    sagitta is under the line tolerance), and without the extension those strips
    would sit outside the fitted arc's span and stay faceted.
    Returns the number of joins snapped.
    """
    thresh = max(config.tangency_floor_deg,
                 config.tangency_k * resolution.median_dihedral_deg)
    snaps = 0
    m = len(segments)
    for k in range(m):
        a = segments[k]
        b = segments[k + 1] if k + 1 < m else None
        if b is None:
            break
        # Line-arc or arc-line join.
        line, arc, line_first = None, None, None
        if a.kind == "line" and b.kind == "arc":
            line, arc, line_first = a, b, True
        elif a.kind == "arc" and b.kind == "line":
            line, arc, line_first = b, a, False
        else:
            continue
        join = a.p1  # shared point
        radial = join - arc.center
        rn = np.linalg.norm(radial)
        if rn < 1e-9:
            continue
        radial_u = radial / rn
        ldir = (line.p1 - line.p0)
        ln = np.linalg.norm(ldir)
        if ln < 1e-9:
            continue
        ldir_u = ldir / ln
        # Tangent condition: line direction perpendicular to the radius.
        dotv = abs(float(ldir_u @ radial_u))
        defect = float(np.degrees(np.arcsin(min(1.0, dotv))))
        if defect > thresh:
            continue
        # True tangent point: foot of the perpendicular from the arc centre onto
        # the line (kept ON the line), projected radially onto the circle (kept
        # ON the arc). The two coincide up to the tangency defect distance.
        foot = line.p0 + float((arc.center - line.p0) @ ldir_u) * ldir_u
        f_radial = foot - arc.center
        fn = float(np.linalg.norm(f_radial))
        if fn < 1e-9:
            continue
        tangent_pt = arc.center + f_radial / fn * arc.radius
        if line_first:
            line.p1 = foot.copy()
            arc.p0 = tangent_pt.copy()
            a.p1 = foot.copy()
            b.p0 = tangent_pt.copy()
        else:
            arc.p1 = tangent_pt.copy()
            line.p0 = foot.copy()
            a.p1 = tangent_pt.copy()
            b.p0 = foot.copy()
        a.tangent_end = True
        b.tangent_start = True
        snaps += 1
    return snaps


def fit_swept_profile(
    vertices: np.ndarray,
    faces: np.ndarray,
    swept: SweptRegion,
    config: ConversionConfig,
    resolution: MeshResolution,
) -> SweptProfile | None:
    """Fit the 2D profile of a swept region (design §2, §3, M4).

    Extracts the profile rails (one per boundary-loop run — a merged multi-wall
    sweep or a wall with cutouts yields several), projects each into the plane
    perpendicular to the sweep axis, partitions it into line + arc runs
    (B-spline for the rest), and snaps near-tangent line<->arc joins to exact
    tangency in 2D. Returns a :class:`SweptProfile` whose segments the builder
    turns into extruded surfaces / boolean lens tools, or ``None`` when the
    region is too small / no rail can be recovered / the fit RMS is too high.
    """
    if swept.size < config.swept_min_facets:
        return None
    extracted = _extract_rails(vertices, faces, swept.face_indices, swept.axis)
    if extracted is None:
        return None
    rails, amin, amax = extracted
    if amax - amin < config.swept_min_extent:
        return None

    axis = swept.axis / (np.linalg.norm(swept.axis) or 1.0)
    e1, e2 = _swept_plane_basis(axis)
    origin = rails[0][0].mean(axis=0)
    origin = origin - float(origin @ axis) * axis + amin * axis
    local_edge = resolution.edge_for(swept.face_indices)
    tol = max(config.swept_profile_tol_abs, config.curve_fit_tol_rel * local_edge)

    segments: list[ProfileSegment] = []
    snaps = 0
    devs_sq_sum = 0.0
    devs_n = 0
    any_closed = False
    for rail_pts, closed in rails:
        rel = rail_pts - origin
        pts2d = _resample_polyline(np.column_stack((rel @ e1, rel @ e2)), closed)
        if len(pts2d) < 3:
            continue
        if closed:
            # Close the polyline so the last->first stretch is fitted too.
            pts2d = np.vstack([pts2d, pts2d[:1]])
            any_closed = True
        runs = _segment_profile(pts2d, config, local_edge)
        rail_segments: list[ProfileSegment] = []
        for kind, i0, i1 in runs:
            seg = pts2d[i0:i1 + 1]
            if len(seg) < 2:
                continue
            if kind == "line" or len(seg) == 2:
                # A 2-point "spline" is just a segment; keep the wire simple.
                rail_segments.append(_fit_line(seg))
            elif kind == "arc":
                a = _fit_arc(seg)
                rail_segments.append(a if a is not None else _fit_line(seg))
            else:
                rail_segments.append(ProfileSegment(kind="spline", points=seg.copy(),
                                                    p0=seg[0].copy(), p1=seg[-1].copy()))
        if not rail_segments:
            continue
        snaps += _apply_tangency(rail_segments, resolution, config)
        # Per-rail RMS gate: a rail whose fit is poor contributes no segments
        # (its strips stay faceted) without sinking the whole sweep.
        rms_rail = _profile_rms(pts2d, rail_segments)
        if rms_rail > 3.0 * tol:
            continue
        devs_sq_sum += rms_rail * rms_rail * len(pts2d)
        devs_n += len(pts2d)
        segments.extend(rail_segments)
    if not segments:
        return None
    segments, snaps = _dedupe_segments(segments)
    rms = math.sqrt(devs_sq_sum / devs_n) if devs_n else float("inf")
    closed = any_closed

    n_lines = sum(1 for s in segments if s.kind == "line")
    n_arcs = sum(1 for s in segments if s.kind == "arc")
    n_splines = sum(1 for s in segments if s.kind == "spline")
    return SweptProfile(
        axis=axis, origin=origin, e1=e1, e2=e2, segments=segments,
        axial_min=amin, axial_max=amax, closed=closed, rms=float(rms),
        face_indices=list(swept.face_indices),
        n_arcs=n_arcs, n_lines=n_lines, n_splines=n_splines, tangency_snaps=snaps,
        member_regions=list(swept.member_regions),
    )


def _dedupe_segments(
    segments: list[ProfileSegment],
) -> tuple[list[ProfileSegment], int]:
    """Drop duplicate profile segments and recount tangency snaps.

    The min- and max-axial rails of a sweep describe the *same* profile curve,
    so both contribute identical segments (traversed in opposite directions).
    One copy suffices; keys fold endpoint order (and arc geometry) so reversed
    duplicates collapse. Snap count = joins marked tangent on kept segments.
    """
    def same_pt(a, b, tol=0.05) -> bool:
        return a is not None and b is not None and float(np.linalg.norm(a - b)) <= tol

    def same_seg(s, t) -> bool:
        if s.kind != t.kind:
            return False
        ends_match = ((same_pt(s.p0, t.p0) and same_pt(s.p1, t.p1))
                      or (same_pt(s.p0, t.p1) and same_pt(s.p1, t.p0)))
        if not ends_match:
            return False
        if s.kind == "arc":
            return (same_pt(s.center, t.center)
                    and abs(s.radius - t.radius) <= 0.02 + 0.01 * s.radius)
        return True

    kept: list[ProfileSegment] = []
    for s in segments:
        if any(same_seg(s, t) for t in kept):
            continue
        kept.append(s)
    snaps = sum(1 for s in kept if s.tangent_start)
    return kept, snaps


def _profile_rms(pts2d: np.ndarray, segments: list[ProfileSegment]) -> float:
    """RMS distance of the rail points to the fitted profile segments."""
    devs: list[float] = []
    for p in pts2d:
        best = float("inf")
        for s in segments:
            if s.kind == "line":
                best = min(best, _dist_point_segment(p, s.p0, s.p1))
            elif s.kind == "arc":
                d = abs(float(np.linalg.norm(p - s.center) - s.radius))
                best = min(best, d)
            elif s.kind == "spline" and s.points is not None:
                for q0, q1 in zip(s.points[:-1], s.points[1:]):
                    best = min(best, _dist_point_segment(p, q0, q1))
        devs.append(best)
    return float(np.sqrt(np.mean(np.square(devs)))) if devs else float("inf")


def _dist_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def detect_swept_walls(
    vertices: np.ndarray,
    faces: np.ndarray,
    swept_regions: list[SweptRegion],
    config: ConversionConfig | None = None,
    resolution: MeshResolution | None = None,
) -> list[SweptProfile]:
    """Fit swept profiles for every swept region (design §2, §3, M4)."""
    config = config or ConversionConfig()
    if not config.detect_swept_walls:
        return []
    if resolution is None:
        resolution = mesh_resolution(vertices, faces, config)
    out: list[SweptProfile] = []
    for sw in swept_regions:
        prof = fit_swept_profile(vertices, faces, sw, config, resolution)
        if prof is not None:
            out.append(prof)
    return out
