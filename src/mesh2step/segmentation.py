"""Planar region growing over a welded triangle mesh.

Groups facets that share a common plane into regions, so that downstream code
can rebuild each region as a single planar STEP face instead of many triangles.
Pure numpy; no FreeCAD dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class MeshResolution:
    """Per-mesh resolution descriptor used to scale fit tolerances.

    Fit gates in the original code are absolute mm constants. Chord error on a
    tessellated curved surface scales with ``edge_length**2 / (8R)``, so on a
    coarse STL the sagitta of a correct fit can be many times the absolute
    tolerance and the surface stays faceted. Scaling the accepted residual with
    the *local* edge length recovers those coarse fits while keeping the guards
    tight where the mesh is fine.
    """

    median_edge: float              # median edge length over the whole mesh (mm)
    local_edge: np.ndarray          # (F,) robust local edge length per face (mm)
    median_dihedral_deg: float      # median dihedral step across smooth edges

    def edge_for(self, face_ids) -> float:
        """Representative local edge length (mm) for a set of faces.

        The median per-face edge length, but clamped to at most 3x the global
        median edge: a curved band's sampling can't be an order of magnitude
        coarser than the whole mesh, and a few oversized triangles (e.g. a
        fan-triangulated end cap misclassified into the cluster) must not inflate
        the fit tolerance into admitting garbage. This keeps the resolution
        scaling honest — it loosens for genuinely coarse bands, not for outliers.
        """
        ids = np.asarray(list(face_ids), dtype=int)
        if ids.size == 0:
            return float(self.median_edge)
        med = float(np.median(self.local_edge[ids]))
        return min(med, 3.0 * self.median_edge)


def mesh_resolution(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig | None = None,
) -> MeshResolution:
    """Compute a :class:`MeshResolution` descriptor. Pure numpy, no FreeCAD.

    ``local_edge`` is each face's mean edge length (a robust per-face scale);
    ``median_edge`` the global median; ``median_dihedral_deg`` the median
    dihedral angle across *smooth* (non-sharp, non-coplanar) shared edges, which
    sets the tangency-defect floor a coarse mesh reads even at a true tangent.
    """
    config = config or ConversionConfig()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    local_edge = (e0 + e1 + e2) / 3.0
    all_edges = np.concatenate([e0, e1, e2])
    median_edge = float(np.median(all_edges)) if all_edges.size else 1.0

    normals, _ = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    cos_flat = config.angle_tol_cos
    cos_sharp = float(np.cos(np.radians(config.curve_max_deg)))
    smooth_steps: list[float] = []
    for incident in adjacency.values():
        if len(incident) != 2:
            continue
        i, j = incident[0], incident[1]
        d = float(np.clip(abs(normals[i] @ normals[j]), -1.0, 1.0))
        if cos_sharp < d < cos_flat:  # smooth curvature step, not flat/sharp
            smooth_steps.append(float(np.degrees(np.arccos(d))))
    median_dihedral_deg = float(np.median(smooth_steps)) if smooth_steps else 0.0

    return MeshResolution(
        median_edge=median_edge,
        local_edge=local_edge,
        median_dihedral_deg=median_dihedral_deg,
    )


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


def planar_merge_tols(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig,
) -> tuple[float, float]:
    """Effective (cos_tol, dist_tol) for planar merging, resolution-scaled.

    Mirrors the curved detectors' ``_local_tol``: the absolute ``angle_tol_deg`` /
    ``dist_tol`` are floors, and the effective tolerance grows with the mesh's own
    tessellation noise (median smooth-dihedral step and median edge length) so a
    coarse or decimated export of a genuinely flat face merges into one region
    instead of shattering — while a fine, clean mesh is unaffected. Both are
    clamped by conservative caps so a genuinely curved wall's arc rows are never
    swallowed into one flat (which would starve the swept-wall / freeform / sphere
    detectors). Returns the strict absolute tolerances when the rel factors are 0
    (legacy behaviour). Cheap pure-numpy call; safe to compute per segmentation.
    """
    angle_deg = float(config.angle_tol_deg)
    dist_tol = float(config.dist_tol)
    if config.planar_angle_tol_rel > 0.0 or config.planar_dist_tol_rel > 0.0:
        res = mesh_resolution(vertices, faces, config)
        if config.planar_angle_tol_rel > 0.0 and res.median_dihedral_deg > 0.0:
            scaled = config.planar_angle_tol_rel * res.median_dihedral_deg
            angle_deg = min(config.planar_angle_tol_cap_deg, max(angle_deg, scaled))
        if config.planar_dist_tol_rel > 0.0 and res.median_edge > 0.0:
            scaled = config.planar_dist_tol_rel * res.median_edge
            dist_tol = min(config.planar_dist_tol_cap, max(dist_tol, scaled))
    cos_tol = float(np.cos(np.radians(angle_deg)))
    return cos_tol, dist_tol


def segment_planar(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig | None = None,
) -> list[Region]:
    """Grow planar regions of coplanar facets.

    A facet joins a region when (a) its normal is within ``angle_tol`` of the
    seed plane normal and (b) all its vertices lie within ``dist_tol`` of the
    seed plane. Seeds are taken largest-area first so dominant faces anchor
    their plane before noise can. Both tolerances are resolution-scaled (see
    :func:`planar_merge_tols`) so coarse/decimated exports of flat faces merge
    into single regions rather than shipping as a faceted-looking fan.
    """
    config = config or ConversionConfig()
    normals, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    neighbors = _face_neighbors(faces, adjacency)

    n_faces = len(faces)
    visited = np.zeros(n_faces, dtype=bool)
    order = np.argsort(-areas)  # largest first
    cos_tol, dist_tol = planar_merge_tols(vertices, faces, config)

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


def planar_coverage(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConversionConfig | None = None,
    min_region_facets: int = 8,
) -> dict:
    """Area-weighted planar coverage: fraction of surface area in *large* flats.

    Segments the mesh with :func:`segment_planar` and reports how much of the
    total surface AREA lands in planar regions of at least ``min_region_facets``
    facets (a "real" flat, not a mesh-noise micro-region). This is the cheap
    planarity-damage metric: a mesh whose flats are genuinely planar has most of
    its flat area in a handful of large regions; decimation that warps those
    flats past the coplanar gate shatters them into thousands of sub-``min``
    micro-regions, dropping this coverage sharply even though the total face
    count only fell. Comparing ``coverage`` before/after decimation (a ratio)
    detects exactly the damage that makes a coarse organic scan ship "everything
    faceted" — without needing a reconstruction or a boolean run.

    Returns ``{coverage, big_area, total_area, n_regions, n_big_regions}``.
    Area-weighted (not facet-count-weighted) so one large warped flat counts for
    its true share, and pure numpy — same ~2 s cost as one ``segment_planar`` on
    a 200k-face mesh.
    """
    _, areas = face_normals_and_areas(vertices, faces)
    total = float(areas.sum())
    if total <= 0.0:
        return {"coverage": 0.0, "big_area": 0.0, "total_area": 0.0,
                "n_regions": 0, "n_big_regions": 0}
    regions = segment_planar(vertices, faces, config)
    big_area = 0.0
    n_big = 0
    for r in regions:
        if len(r.face_indices) >= min_region_facets:
            big_area += float(areas[r.face_indices].sum())
            n_big += 1
    return {
        "coverage": big_area / total,
        "big_area": big_area,
        "total_area": total,
        "n_regions": len(regions),
        "n_big_regions": n_big,
    }


@dataclass
class SmoothRegion:
    """A smoothly-curved band/cap/blend built from a chain of thin planar
    sub-regions whose normals rotate across a curved surface (e.g. a fillet's
    tessellated arc rows), tagged with the flat regions it borders.

    ``class_hint`` is a cheap topological signature: ``band`` (a strip bordering
    exactly two flat regions — the fillet/chamfer case), ``blend`` (bordering
    >=3 flats — a corner), or ``cap`` (bordering <=1 flat — an end cap / dome).
    ``border_regions`` are indices into the ``regions`` list.
    ``member_regions`` are the sub-region indices forming the band.
    """

    face_indices: list[int]
    border_regions: list[int]
    class_hint: str          # "band" | "cap" | "blend"
    aspect: float            # perimeter^2 / area  (long-and-thin => large)
    member_regions: list[int] = field(default_factory=list)


def _region_adjacency(
    faces: np.ndarray, region_of: np.ndarray, n_regions: int,
    adjacency: dict[tuple[int, int], list[int]],
) -> dict[int, set[int]]:
    """Which planar regions share a mesh edge (region graph)."""
    graph: dict[int, set[int]] = {r: set() for r in range(n_regions)}
    for incident in adjacency.values():
        if len(incident) != 2:
            continue
        ra, rb = int(region_of[incident[0]]), int(region_of[incident[1]])
        if ra != rb and ra >= 0 and rb >= 0:
            graph[ra].add(rb)
            graph[rb].add(ra)
    return graph


def segment_smooth_bands(
    vertices: np.ndarray,
    faces: np.ndarray,
    claimed: set[int],
    regions: list[Region],
    config: ConversionConfig | None = None,
) -> list[SmoothRegion]:
    """Group thin curved sub-regions into fillet/chamfer bands (design §2, §3).

    A tessellated fillet arrives as a chain of thin planar strips whose normals
    rotate monotonically between two large flats (segment_planar splits each arc
    row into its own coplanar region). This groups regions connected across
    *smooth* dihedral steps into a band, records the large flat regions the band
    borders, and classifies band/cap/blend by that border count. Pure numpy.
    """
    config = config or ConversionConfig()
    n_faces = len(faces)
    if n_faces == 0 or not regions:
        return []

    _, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)

    region_of = np.full(n_faces, -1, dtype=int)
    for ri, region in enumerate(regions):
        for fi in region.face_indices:
            region_of[fi] = ri
    n_regions = len(regions)
    region_area = np.array([float(areas[r.face_indices].sum()) for r in regions])

    graph = _region_adjacency(faces, region_of, n_regions, adjacency)
    cos_flat = config.angle_tol_cos
    cos_sharp = float(np.cos(np.radians(config.curve_max_deg)))

    # Two regions take a "smooth step" when their normals differ by a curved
    # (non-coplanar, non-sharp) angle — the dihedral-chain predicate that links
    # a fillet's rotating arc rows.
    def smooth_step(a: int, b: int) -> bool:
        na = regions[a].plane_normal
        nb = regions[b].plane_normal
        d = float(np.clip(abs(na @ nb), -1.0, 1.0))
        return cos_sharp < d < cos_flat

    # A curved-band member (a rotating arc row) sits *between* other rows, so it
    # has >=2 smooth-step neighbours AND is small (a thin strip). A bounding flat
    # is either tangent on only one side (one smooth-step neighbour) or, if it is
    # tangent to fillets on two of its edges, still a large dominant face — so
    # requiring a strip to be BOTH multiply-smooth AND non-dominant separates the
    # arc rows from their flats reliably even when a flat blends two edges.
    smooth_count = np.zeros(n_regions, dtype=int)
    for a in range(n_regions):
        for b in graph[a]:
            if smooth_step(a, b):
                smooth_count[a] += 1

    # "Dominant" = a face much larger than a typical region (a bounding flat).
    # Arc-row strips cluster at a small area; the flats they blend are many times
    # larger, so a multiple of the median region area cleanly separates them (a
    # percentile fails when most regions are the small arc rows themselves).
    med_area = float(np.median(region_area)) if region_area.size else 0.0
    big = 5.0 * med_area if med_area > 0 else float("inf")

    strip_pool = [r for r in range(n_regions)
                  if smooth_count[r] >= 2 and region_area[r] <= big]
    strip_set = set(strip_pool)

    # Connected components of strip regions joined by smooth steps.
    seen: set[int] = set()
    out: list[SmoothRegion] = []
    for start in strip_pool:
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        stack = [start]
        while stack:
            r = stack.pop()
            for nb in graph[r]:
                if nb in seen or nb not in strip_set:
                    continue
                if smooth_step(r, nb):
                    seen.add(nb)
                    comp.append(nb)
                    stack.append(nb)

        member_faces: list[int] = []
        for r in comp:
            member_faces.extend(regions[r].face_indices)
        member_faces = [fi for fi in member_faces if fi not in claimed]
        if not member_faces:
            continue

        # Border flats: regions adjacent to the band that are NOT band members.
        # The blend PARTNERS are the flats the band is *tangent* to (a smooth
        # dihedral step from a band member) — the two surfaces the fillet rounds
        # between. Borders met at a sharp edge (the end caps of a straight
        # fillet, perpendicular to the sweep) are NOT partners and are excluded
        # from the band/cap/blend classification.
        comp_set = set(comp)
        tangent_border: set[int] = set()
        for r in comp:
            for nb in graph[r]:
                if nb not in comp_set and smooth_step(r, nb):
                    tangent_border.add(nb)
        border = tangent_border

        area = float(areas[member_faces].sum())
        perimeter = _component_perimeter(faces, member_faces, adjacency, vertices)
        aspect = (perimeter * perimeter / area) if area > 0 else 0.0
        n_border = len(border)
        if n_border >= 3:
            class_hint = "blend"
        elif n_border <= 1:
            class_hint = "cap"
        else:
            class_hint = "band"
        out.append(
            SmoothRegion(
                face_indices=member_faces,
                border_regions=sorted(border),
                class_hint=class_hint,
                aspect=aspect,
                member_regions=sorted(comp),
            )
        )
    return out


def _component_perimeter(faces, member_faces, adjacency, vertices) -> float:
    comp_set = set(member_faces)
    perimeter = 0.0
    for fi in member_faces:
        a, b, c = (int(x) for x in faces[fi])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            incident = adjacency.get(key, [])
            if any(k not in comp_set for k in incident) or len(incident) < 2:
                perimeter += float(np.linalg.norm(vertices[u] - vertices[v]))
    return perimeter


# --------------------------------------------------------------------------- #
# Swept / extruded curved-wall regions (Milestone 4, design §2, CURVED_FEATURES
# §6a). A swept wall is a constant-cross-section extrusion: every facet normal is
# perpendicular to one common extrusion direction ``d`` and the profile (the
# region seen looking down ``d``) repeats along ``d``. Tessellated, it arrives as
# a fan of thin planar strips whose normals rotate about ``d`` — a smooth chain
# that ``segment_planar`` split into one region per arc row.
# --------------------------------------------------------------------------- #


@dataclass
class SweptRegion:
    """A constant-cross-section (extruded) curved wall built from a chain of thin
    planar strips whose normals all lie perpendicular to a common extrusion
    direction ``axis`` (the sweep direction ``d``).

    ``face_indices`` index the mesh the region was segmented from.
    ``member_regions`` are the planar sub-region indices forming the sweep.
    ``axis`` is the unit extrusion direction; the profile lives in the plane
    perpendicular to it. ``axial_min``/``axial_max`` bound the sweep along
    ``axis`` (from a chosen origin).
    """

    face_indices: list[int]
    axis: np.ndarray
    member_regions: list[int] = field(default_factory=list)
    axial_min: float = 0.0
    axial_max: float = 0.0
    normal_dot_axis_max: float = 0.0  # worst |n·axis| over member facets (QA)

    @property
    def size(self) -> int:
        return len(self.face_indices)

    @property
    def extent(self) -> float:
        return self.axial_max - self.axial_min


def segment_swept_walls(
    vertices: np.ndarray,
    faces: np.ndarray,
    claimed: set[int],
    regions: list[Region],
    config: ConversionConfig | None = None,
) -> list[SweptRegion]:
    """Group planar strips into swept (extruded constant-cross-section) walls.

    Each sweep is grown from a seed planar region: the extrusion direction ``d``
    is fixed from the cross product of the seed's normal with a smooth neighbour's
    (both perpendicular to ``d`` on a true sweep), then the region grows across
    *smooth* dihedral steps only to neighbours whose normal stays perpendicular to
    that fixed ``d``. Fixing ``d`` up front is what stops the chain drifting around
    corners onto the end caps (whose normals are *parallel* to ``d``) and merging
    the whole body into one blob — the failure mode of a plain smooth-chain walk.

    Pure numpy. Returns sweeps with >= ``swept_min_regions`` member strips and a
    consistent ``d``-extent; the builder fits a profile curve and extrudes it.
    """
    config = config or ConversionConfig()
    n_faces = len(faces)
    if n_faces == 0 or not regions:
        return []

    _, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)

    region_of = np.full(n_faces, -1, dtype=int)
    for ri, region in enumerate(regions):
        for fi in region.face_indices:
            region_of[fi] = ri
    n_regions = len(regions)
    graph = _region_adjacency(faces, region_of, n_regions, adjacency)
    rn = np.array([r.plane_normal / (np.linalg.norm(r.plane_normal) or 1.0)
                   for r in regions])
    rarea = np.array([float(areas[r.face_indices].sum()) for r in regions])

    cos_flat = config.angle_tol_cos
    cos_sharp = float(np.cos(np.radians(config.curve_max_deg)))

    def smooth_step(a: int, b: int) -> bool:
        d = float(np.clip(abs(rn[a] @ rn[b]), -1.0, 1.0))
        return cos_sharp < d < cos_flat

    # A region is unavailable to seed/grow a sweep if all its facets are already
    # claimed (by cylinders/cones/fillets).
    def region_free(ri: int) -> bool:
        return any(fi not in claimed for fi in regions[ri].face_indices)

    dtol = config.swept_axis_perp_tol
    order = np.argsort(-rarea)  # largest strips seed first
    assigned = np.full(n_regions, -1, dtype=int)
    sweeps: list[SweptRegion] = []

    for seed in order:
        if assigned[seed] >= 0 or not region_free(seed):
            continue
        # Fix the extrusion direction from the seed + a smooth neighbour: on a
        # true sweep both normals are perpendicular to d, so d = n_seed x n_nbr.
        dvec = None
        for nb in graph[seed]:
            if assigned[nb] >= 0 or not smooth_step(seed, nb) or not region_free(nb):
                continue
            c = np.cross(rn[seed], rn[nb])
            nc = float(np.linalg.norm(c))
            if nc > 1e-6:
                dvec = c / nc
                break
        if dvec is None:
            continue

        sid = len(sweeps)
        comp = [seed]
        assigned[seed] = sid
        stack = [seed]
        while stack:
            r = stack.pop()
            for nb in graph[r]:
                if assigned[nb] >= 0 or not region_free(nb):
                    continue
                if not smooth_step(r, nb):
                    continue
                # Grow only onto strips whose plane stays perpendicular to the
                # fixed sweep direction (rotating profile, constant section).
                if abs(float(rn[nb] @ dvec)) > dtol:
                    continue
                assigned[nb] = sid
                comp.append(nb)
                stack.append(nb)

        if len(comp) < config.swept_min_regions:
            for r in comp:
                assigned[r] = -1
            continue

        member_faces = [fi for r in comp for fi in regions[r].face_indices
                        if fi not in claimed]
        if len(member_faces) < config.swept_min_facets:
            for r in comp:
                assigned[r] = -1
            continue

        member_normals = rn[comp]
        ndotd = float(np.abs(member_normals @ dvec).max())
        vids = np.unique(faces[member_faces].reshape(-1))
        ax = vertices[vids] @ dvec
        sweeps.append(
            SweptRegion(
                face_indices=member_faces,
                axis=dvec,
                member_regions=sorted(comp),
                axial_min=float(ax.min()),
                axial_max=float(ax.max()),
                normal_dot_axis_max=ndotd,
            )
        )
    return sweeps


# --------------------------------------------------------------------------- #
# Freeform sheet regions (Candidate B, docs/ORGANIC_CONVERSION_RESEARCH.md). The
# residual after all analytic + swept + sphere detectors is dominated, on some
# parts, by genuinely doubly-curved regions (an ergonomic shell, a curved lid,
# a camera-adapter panel) that no analytic fit and no constant-cross-section
# sweep claims. Where such a region is a *height field* — injective under a
# projection axis (its facet normals all sit on one side of the axis, no
# foldover) — it can be resampled on a (u,v) grid and fitted with a single
# trimmed B-spline face.
#
# Region growth (the key to injectivity): fix a projection axis from a seed
# facet's normal and grow across smooth dihedral steps, admitting a neighbour
# only while its normal stays on the +axis side (n·axis > tol). The axis is
# refreshed to the region's running mean normal so the patch can follow gentle
# curvature further, but the +side test guarantees the grown region never wraps
# past its own silhouette (which would fold over under projection). Strongly
# doubly-curved surfaces that wrap past a silhouette (a closed organic blob)
# simply fragment into several injective patches or are rejected by the final
# foldover gate — never mis-fit as one multivalued sheet.
# --------------------------------------------------------------------------- #


@dataclass
class FreeformRegion:
    """A residual doubly-curved height-field region to fit as one B-spline sheet.

    ``axis`` is the injective projection direction (facet normals sit on its
    +side); the profile grid lives in the plane perpendicular to it. ``e1``,
    ``e2`` are the in-plane basis, ``origin`` the grid origin. ``curvature`` is
    the region's peak-to-peak height variation about its mean plane (mm) — the
    doubly-curved signal that separates a genuine freeform sheet from a flat
    strip that happens to be residual. ``foldover`` is the injectivity defect
    (area fraction of facets facing away from ``axis``; ~0 for a clean field).
    """

    face_indices: list[int]
    axis: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    origin: np.ndarray
    area: float
    curvature: float
    foldover: float

    @property
    def size(self) -> int:
        return len(self.face_indices)


def _axis_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A right-handed in-plane basis (e1, e2) perpendicular to ``axis``."""
    t = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, t)
    e1 = e1 / (np.linalg.norm(e1) or 1.0)
    e2 = np.cross(axis, e1)
    return e1, e2


def segment_freeform_sheets(
    vertices: np.ndarray,
    faces: np.ndarray,
    claimed: set[int],
    config: ConversionConfig | None = None,
) -> list[FreeformRegion]:
    """Group unclaimed residual facets into doubly-curved height-field regions.

    Pure numpy. Grows injective (single-valued) regions via adaptive-axis
    height-field growth, then keeps only regions that are (a) large enough
    (``freeform_min_facets`` / ``freeform_min_area``), (b) genuinely
    doubly-curved (peak-to-peak height about the mean plane exceeds a
    resolution-scaled floor — a flat residual strip is left to the planar path),
    and (c) injective (foldover below ``freeform_max_foldover``). Everything
    else is left faceted (no regression).
    """
    config = config or ConversionConfig()
    n_faces = len(faces)
    if n_faces == 0:
        return []
    normals, areas = face_normals_and_areas(vertices, faces)
    adjacency = build_edge_adjacency(faces)
    nbr: list[list[int]] = [[] for _ in range(n_faces)]
    for incident in adjacency.values():
        if len(incident) == 2:
            nbr[incident[0]].append(incident[1])
            nbr[incident[1]].append(incident[0])

    cos_flat = config.angle_tol_cos
    cos_sharp = float(np.cos(np.radians(config.curve_max_deg)))
    ndot_tol = float(config.freeform_ndot_tol)
    unclaimed = [i for i in range(n_faces) if i not in claimed]

    # Only CURVED facets seed a region: a facet with at least one smooth-step
    # neighbour (a curved surface), not a flat-wall facet (all neighbours
    # coplanar). Seeding from a flat wall grows a 2-facet stub and fragments the
    # pool; seeding from the curved surface lets the height field span it. Among
    # curved seeds, largest area first for stable growth.
    def is_curved_seed(i: int) -> bool:
        ni = normals[i]
        for y in nbr[i]:
            if y in claimed:
                continue
            d = float(np.clip(abs(ni @ normals[y]), 0.0, 1.0))
            if cos_sharp < d < cos_flat:
                return True
        return False

    seeds_pool = [i for i in unclaimed if is_curved_seed(i)]
    order = sorted(seeds_pool, key=lambda i: -float(areas[i]))
    assigned: set[int] = set()
    out: list[FreeformRegion] = []

    res = mesh_resolution(vertices, faces, config)

    def grow_from(seed: int, axis0: np.ndarray, blocked: set[int]) -> list[int]:
        """Flood from ``seed`` across smooth steps, admitting a neighbour only
        while its normal stays on the +``axis0`` side (n·axis0 ≥ ndot_tol). The
        axis is FIXED for the pass (not drifting), so the region is a genuine
        height field about ``axis0`` and cannot wander onto a perpendicular
        face. ``blocked`` are facets already owned by other regions."""
        comp = [seed]
        local = {seed}
        stack = [seed]
        while stack:
            x = stack.pop()
            for y in nbr[x]:
                if y in local or y in blocked or y in claimed:
                    continue
                d = float(np.clip(abs(normals[x] @ normals[y]), 0.0, 1.0))
                if d <= cos_sharp:
                    continue
                if float(normals[y] @ axis0) < ndot_tol:
                    continue
                local.add(y)
                comp.append(y)
                stack.append(y)
        return comp

    for seed in order:
        if seed in assigned:
            continue
        # Iterate growth to convergence: seed the axis from the seed normal, grow
        # a height field, refresh the axis to the region's area-weighted mean
        # normal, and re-grow — twice. A fixed axis per pass keeps each region a
        # true height field (no wander), while the refresh lets the axis settle
        # on the surface's real facing (Z for a bump top) rather than the tilted
        # seed facet. Converges the region to the maximal field around the seed.
        axis = normals[seed].astype(float).copy()
        comp = grow_from(seed, axis, assigned)
        for _ in range(2):
            fa0 = np.array(comp, dtype=int)
            mn0 = (normals[fa0] * areas[fa0][:, None]).sum(axis=0)
            nmn = float(np.linalg.norm(mn0))
            if nmn < 1e-6:
                break
            new_axis = mn0 / nmn
            if float(new_axis @ axis) > 0.9995:
                break
            axis = new_axis
            comp = grow_from(seed, axis, assigned)
        for c in comp:
            assigned.add(c)

        if len(comp) < config.freeform_min_facets:
            # Release the small component's members (except the seed, to avoid
            # re-seeding it) so its facets can join a neighbouring region.
            for c in comp:
                if c != seed:
                    assigned.discard(c)
            continue

        fa = np.array(comp, dtype=int)
        area = float(areas[fa].sum())
        if area < config.freeform_min_area:
            for c in comp:
                if c != seed:
                    assigned.discard(c)
            continue

        # Validate + emit the region as a doubly-curved height field. Build-time
        # deviation-triggered splitting (task §1) happens in the builder, which
        # can re-run _freeform_subregions on a sheet whose true B-spline fit
        # misses the mesh — the honest trigger. A segmentation-time quadratic
        # residual over-fires on gentle single bumps, so it is not used here.
        reg = _finalize_freeform_region(
            fa, vertices, faces, normals, areas, res, ndot_tol, config)
        if reg is not None:
            out.append(reg)
    return out


def _finalize_freeform_region(
    fa: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    res: "MeshResolution",
    ndot_tol: float,
    config: ConversionConfig,
) -> FreeformRegion | None:
    """Validate one facet set as a doubly-curved height field and return the
    :class:`FreeformRegion`, or ``None`` if it fails the injectivity / curvature
    / double-curvature gates."""
    if len(fa) < config.freeform_min_facets:
        return None

    n = normals[fa]
    ar = areas[fa]
    tot = float(ar.sum())
    mn = (n * ar[:, None]).sum(axis=0)
    na = float(np.linalg.norm(mn))
    if na < 1e-6:
        return None
    axis = mn / na
    # Drop stragglers whose normal is near-perpendicular to the mean axis (a side
    # wall the growth reached over a smooth edge): they are not part of this
    # height field and would corrupt its curvature/extent. Recompute the axis
    # once from the trimmed set for stability.
    keep_mask = (n @ axis) >= ndot_tol
    if keep_mask.sum() < config.freeform_min_facets:
        return None
    fa = fa[keep_mask]
    n = normals[fa]
    ar = areas[fa]
    tot = float(ar.sum())
    mn = (n * ar[:, None]).sum(axis=0)
    na = float(np.linalg.norm(mn))
    if na < 1e-6:
        return None
    axis = mn / na
    dots = n @ axis
    best_fold = float(ar[dots < -0.05].sum()) / tot if tot > 0 else 1.0
    if best_fold > config.freeform_max_foldover:
        return None

    e1, e2 = _axis_basis(axis)
    pts = vertices[faces[fa]].reshape(-1, 3)
    origin = pts.mean(axis=0)
    # Height field h(u,v) about the mean plane.
    rel = pts - origin
    u = rel @ e1
    w = rel @ e2
    h = rel @ axis
    curvature = float(h.max() - h.min())
    edge = res.edge_for(fa.tolist())
    curv_floor = max(config.freeform_min_curvature,
                     config.freeform_min_curvature_edges * edge)
    if curvature < curv_floor:
        return None

    # Double-curvature gate: the surface must bend in BOTH in-plane directions,
    # else it is a single-curvature wall that the swept detector owns (and
    # de-facets more cheaply). Fit h ≈ a·u + b·w + c·u² + d·w² + e·uw by least
    # squares; the two second-order coefficients (c, d) are the principal
    # curvatures. A cylinder/sweep has one ≈ 0; a genuine freeform sheet has both
    # non-negligible.
    try:
        span_u = float(u.max() - u.min()) or 1.0
        span_w = float(w.max() - w.min()) or 1.0
        un = u / span_u
        wn = w / span_w
        A = np.column_stack([un, wn, un * un, wn * wn, un * wn,
                             np.ones_like(un)])
        coef, *_ = np.linalg.lstsq(A, h, rcond=None)
        bend_u = abs(float(coef[2]))  # curvature along e1 (over normalised u)
        bend_w = abs(float(coef[3]))  # curvature along e2
    except Exception:  # noqa: BLE001
        bend_u = bend_w = 0.0
    double = min(bend_u, bend_w)
    pk = max(curvature, 1e-6)
    if double < config.freeform_double_curve_frac * pk:
        return None

    return FreeformRegion(
        face_indices=sorted(int(i) for i in fa),
        axis=axis,
        e1=e1,
        e2=e2,
        origin=origin,
        area=float(ar.sum()),
        curvature=curvature,
        foldover=best_fold,
    )


def split_freeform_region(
    vertices: np.ndarray,
    faces: np.ndarray,
    region: FreeformRegion,
    config: ConversionConfig | None = None,
) -> list[FreeformRegion]:
    """Bisect a freeform region into two sub-regions along its dominant in-plane
    curvature ridge, re-validating each as its own height field (task §1).

    Used at BUILD time when a region's fitted B-spline misses the real mesh (its
    deviation gate fails): a large cast surface can be locally a height field but
    curve too much to be one clean field, so splitting along the ridge yields two
    flatter sub-fields the builder can each fit. Returns 0-2 valid sub-regions
    (an empty list means the split produced nothing usable — the caller then
    leaves the region faceted). Pure numpy."""
    config = config or ConversionConfig()
    fa = np.array(region.face_indices, dtype=int)
    if len(fa) < 2 * config.freeform_min_facets:
        return []
    normals, areas = face_normals_and_areas(vertices, faces)
    res = mesh_resolution(vertices, faces, config)
    ndot_tol = float(config.freeform_ndot_tol)

    # Bisect along the in-plane basis direction with the larger span (the ridge
    # the surface bends over): points either side of the median form two sub-
    # fields. Using the geometric long axis (not the quadratic curvature, which
    # is noisy) keeps the split stable.
    origin = region.origin
    cent = vertices[faces[fa]].mean(axis=1)
    pu = (cent - origin) @ region.e1
    pw = (cent - origin) @ region.e2
    split_dir = region.e1 if (pu.max() - pu.min()) >= (pw.max() - pw.min()) else region.e2
    proj = (cent - origin) @ split_dir
    med = float(np.median(proj))
    left = fa[proj <= med]
    right = fa[proj > med]
    out: list[FreeformRegion] = []
    for sub in (left, right):
        if len(sub) < config.freeform_min_facets:
            continue
        reg = _finalize_freeform_region(
            sub, vertices, faces, normals, areas, res, ndot_tol, config)
        if reg is not None:
            out.append(reg)
    return out


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the ``mask`` cells reachable from the grid border through
    other ``mask`` cells (4-connected flood fill from the edge).

    Used to tell a footprint's OPEN boundary frame (missing cells touching the
    grid edge) apart from an ENCLOSED interior hole (missing cells surrounded by
    covered cells). Pure numpy, iterated dilation intersected with ``mask``."""
    reach = np.zeros_like(mask, dtype=bool)
    reach[0, :] |= mask[0, :]
    reach[-1, :] |= mask[-1, :]
    reach[:, 0] |= mask[:, 0]
    reach[:, -1] |= mask[:, -1]
    while True:
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= mask
        if np.array_equal(grown, reach):
            return reach
        reach = grown


def _laplace_inpaint(height: np.ndarray, mask: np.ndarray,
                     iters: int = 400) -> np.ndarray:
    """Fill the ``~mask`` cells of a height grid by solving a discrete Laplace
    equation (∇²h = 0) with the covered cells as Dirichlet boundary conditions.

    Jacobi/Gauss-Seidel relaxation: each unknown cell is repeatedly replaced by
    the mean of its 4-neighbours (reflecting at the grid edge so a boundary hole
    extrapolates the interior gradient smoothly rather than clamping to a wall).
    This yields the smoothest (minimal-curvature) interpolant through the covered
    values — the natural extension of a height field into ragged / notched /
    boundary gaps, and exactly what an OCC B-spline approximation wants (no
    nearest-neighbour step discontinuities that manufacture folds). Pure numpy.
    """
    h = height.astype(float).copy()
    known = mask.astype(bool)
    if known.all() or not known.any():
        return h
    # Seed unknowns with the global mean of knowns so relaxation converges fast.
    h[~known] = float(h[known].mean())
    ng = h.shape[0]
    for _ in range(iters):
        up = np.empty_like(h)
        up[1:, :] = h[:-1, :]
        up[0, :] = h[1, :] if ng > 1 else h[0, :]
        dn = np.empty_like(h)
        dn[:-1, :] = h[1:, :]
        dn[-1, :] = h[-2, :] if ng > 1 else h[-1, :]
        lf = np.empty_like(h)
        lf[:, 1:] = h[:, :-1]
        lf[:, 0] = h[:, 1] if ng > 1 else h[:, 0]
        rt = np.empty_like(h)
        rt[:, :-1] = h[:, 1:]
        rt[:, -1] = h[:, -2] if ng > 1 else h[:, -1]
        avg = 0.25 * (up + dn + lf + rt)
        new = np.where(known, h, avg)
        if np.max(np.abs(new - h)) < 1e-6:
            h = new
            break
        h = new
    return h


def sample_freeform_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    region: FreeformRegion,
    ng: int,
    inpaint: bool = True,
    return_mask: bool = False,
):
    """Resample the mesh over ``region``'s (u,v) footprint into an ``ng``×``ng``
    grid of 3D points (height along ``axis`` from the outermost surface hit).

    Pure numpy: for each grid (u,v) the height is the max axis-projection over
    the region facets whose 2D barycentric coordinates contain (u,v) — the
    outermost sheet along the projection. Cells outside the footprint (a ragged
    boundary, an interior notch, an L-shaped corner) are *inpainted* by a
    discrete Laplace solve from the covered cells (``inpaint=True``, the default)
    — a smooth minimal-curvature extension rather than a nearest-centroid step
    that would manufacture a fold. The extrapolated skirt is oversized on
    purpose: the builder's boolean CUT trims whatever lands outside the solid, so
    a smoothly-extended grid is safe. Returns ``(grid, missing_fraction)`` (the
    covered-cell fraction is a quality signal the caller still gates on) or
    ``None`` if the footprint is degenerate. The grid feeds the OCC B-spline
    approximation in the builder (the only FreeCAD-touching step)."""
    fa = np.array(region.face_indices, dtype=int)
    axis, e1, e2, origin = region.axis, region.e1, region.e2, region.origin
    tri = vertices[faces[fa]]
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    Au = np.column_stack(((A - origin) @ e1, (A - origin) @ e2))
    Bu = np.column_stack(((B - origin) @ e1, (B - origin) @ e2))
    Cu = np.column_stack(((C - origin) @ e1, (C - origin) @ e2))
    Ah = (A - origin) @ axis
    Bh = (B - origin) @ axis
    Ch = (C - origin) @ axis

    allpts = tri.reshape(-1, 3)
    uv = np.column_stack(((allpts - origin) @ e1, (allpts - origin) @ e2))
    umin, umax = float(uv[:, 0].min()), float(uv[:, 0].max())
    vmin, vmax = float(uv[:, 1].min()), float(uv[:, 1].max())
    if umax - umin < 1e-6 or vmax - vmin < 1e-6:
        return None

    # Barycentric denominators, precomputed once.
    v0 = Bu - Au
    v1 = Cu - Au
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    cent = tri.mean(axis=1)
    cu = (cent - origin) @ e1
    cv = (cent - origin) @ e2

    us = np.linspace(umin, umax, ng)
    vs = np.linspace(vmin, vmax, ng)
    hgrid = np.zeros((ng, ng))
    covered = np.zeros((ng, ng), dtype=bool)
    for i, uu in enumerate(us):
        pp0 = uu - Au[:, 0]
        for j, vv in enumerate(vs):
            pp1 = vv - Au[:, 1]
            d20 = v0[:, 0] * pp0 + v0[:, 1] * pp1
            d21 = v1[:, 0] * pp0 + v1[:, 1] * pp1
            vb = (d11 * d20 - d01 * d21) / denom
            wb = (d00 * d21 - d01 * d20) / denom
            ub = 1.0 - vb - wb
            inside = (ub >= -0.03) & (vb >= -0.03) & (wb >= -0.03)
            if inside.any():
                hgrid[i, j] = float(np.max((ub * Ah + vb * Bh + wb * Ch)[inside]))
                covered[i, j] = True
            else:
                # Provisional nearest-centroid height; overwritten by the
                # Laplace inpaint below when enabled (kept as the fallback so a
                # disabled inpaint reproduces the historical behaviour).
                k = int(np.argmin((cu - uu) ** 2 + (cv - vv) ** 2))
                hgrid[i, j] = float((cent[k] - origin) @ axis)

    missing = float((~covered).sum()) / float(ng * ng)
    if inpaint and not covered.all() and covered.any():
        # Inpaint INTERIOR holes (missing cells enclosed by covered cells) with a
        # smooth Laplace solve; leave BOUNDARY-connected missing cells (the thin
        # frame the (u,v) bounding box adds around a nearly-rectangular footprint,
        # or an open L-shaped edge) at their nearest-centroid height. Reason: a
        # harmonic extension of a domed field past its rim flat-extends the edge
        # UPWARD, and the extruded cut then leaves that raised skirt — growing the
        # part's bbox and getting the whole (correct) sheet rejected (freeform_bump
        # regression). Interior holes have covered cells on all sides, so their
        # harmonic fill stays inside the surface and is exactly where inpainting
        # earns its keep (port_cover's notched / cored regions). Boundary-open
        # regions still benefit: the raggedness that used to blow up the missing
        # fraction is tolerated, and the real overshoot is trimmed by the cut.
        border_missing = _border_connected(~covered)
        interior_fill = ~covered & ~border_missing
        if interior_fill.any():
            hgrid = _laplace_inpaint(hgrid, covered | border_missing)

    grid = np.zeros((ng, ng, 3))
    for i, uu in enumerate(us):
        for j, vv in enumerate(vs):
            grid[i, j] = origin + uu * e1 + vv * e2 + hgrid[i, j] * axis
    if return_mask:
        return grid, missing, covered
    return grid, missing
