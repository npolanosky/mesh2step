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
    coverage: float = 1.0    # fraction of the full circle the facets span (0..1)

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


def _fit_circle_for_facets(
    vertices: np.ndarray,
    faces: np.ndarray,
    facet_ids: list[int],
    axis: np.ndarray,
    normals: np.ndarray,
    max_radius: float,
    config: ConversionConfig,
) -> Cylinder | None:
    """Fit + validate a cylinder for wall facets known to share ``axis``."""
    if len(facet_ids) < config.min_cylinder_facets:
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
    # equidistant from a centre yet have centroids well inside that radius).
    fcent = vertices[faces[facet_ids]].mean(axis=1)
    radial_vec = (fcent - axis_point) - ((fcent - axis_point) @ axis)[:, None] * axis
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    if abs(float(radial_dist.mean()) - radius) > 0.05 * radius + 0.05:
        return None
    if float(radial_dist.std()) > 0.05 * radius + 0.05:
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
                vertices, faces, cluster, axis, normals, max_radius, config
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
