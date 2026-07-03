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
from .segmentation import (
    MeshResolution,
    Region,
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
