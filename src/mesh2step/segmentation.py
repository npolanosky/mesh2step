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
