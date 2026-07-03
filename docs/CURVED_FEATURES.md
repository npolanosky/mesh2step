# Design: Curved-Feature Reconstruction (Fillets, Chamfers, Spheres, Tori)

Status: approved design, pre-implementation. Goal: minimize residual faceted
features on watertight models. Scoreboard: community-sweep `skipped_facets`
(worst: cross_brace 11,780 / 22%; labstack_fan_panel_2u 10,732 / 41%;
labstack_drive_bay_525 9,510 / 66%; USB-holder 48%; insert-cable holder 50%)
plus user-flagged parts in `tests/data/community/failures/faceted_improvable/`
(first benchmark: "3x1 Tweezer Mount.stl").

## 0. Scope and grounding

Today `builder.build_reconstructed_solid` reconstructs three analytic
families: planar regions (`segmentation.segment_planar` →
`boundary.extract_face_loops` → `_planar_face`), cylinders and cones
(`fitting.detect_cylinders` / `detect_cones` → `_cylinder_face` /
`_cone_face`). Everything a cylinder/cone doesn't claim and that isn't
coplanar falls through as `skipped_facets` and is emitted as
merged-but-still-faceted patches in `_faceted_faces` (the gap-fill path).
The scoreboard's "unreconstructed facets" is exactly this residual.

The residual is dominated by three geometric families, all analytically
reconstructible:

- **Straight-edge fillets/chamfers** — a fillet along a straight edge is a
  *cylinder section* (partial arc, <0.5 coverage); a chamfer is a *plane*.
  These almost-fit the existing cylinder/plane machinery but are rejected by
  `min_cylinder_coverage` (0.33) and by `_fit_circle_for_facets`
  centroid/coverage guards, or never get a candidate axis because the axis is
  the edge direction, not a face normal.
- **Corner blends / domes** — spherical caps where three fillets meet, or
  rounded bosses. No fitter exists.
- **Curved walls** — swept profiles (constant cross-section) that are neither
  full cylinders nor planes; today only full-coverage cylinders survive.

Key architectural fact: **the boolean-clean tier
(`build_boolean_clean_solid`) is the path that actually ships watertight
analytic geometry on real parts.** The sew-based path drops faces when
analytic and mesh edges disagree. New curved surfaces that must interoperate
with faceted neighbors should follow the boolean cut/fuse-back pattern
(DESIGN.md lines 110–131, `_boolean_clean_cylinder`), not naive sewing. New
surface-building primitives (`_torus_face`, `_sphere_face`) are needed in
both tiers, but the **integration** center of gravity is the boolean tier.

## 1. Tangency prior and resolution-scaled tolerances

Foundational; belongs in `fitting.py` + `config.py`, feeding every new
fitter. Product-owner requirements (Nick):

- If a fitted fillet/cylinder/sphere is *nearly* tangent to adjacent flats,
  tangency is design intent: snap to exact tangency and derive the radius
  from the constraint (coarse meshes chord-cut the surface and under-read
  radii). If *far* from tangent, it's intentional geometry: best-effort fit,
  no snapping, purely to eliminate facets.
- Fit tolerances must scale with local mesh resolution so low-res STLs with
  big facets on curved surfaces still get analytic fits.

### 1.1 Resolution-scaled tolerances

Current fit gates are absolute mm constants (`cylinder_tol=0.05`,
`dist_tol=0.01`, `min_cylinder_radius=0.4`). Chord error scales with
`edge_length²/(8R)`: a 2 mm-edge facet on an R=1 fillet has ~0.5 mm sagitta —
10× over `cylinder_tol` — so coarse bands stay faceted (why labstack panels
score 41–66% unreconstructed).

Add a per-mesh resolution descriptor, computed once (new helper in
`segmentation.py`, pure numpy):

```
def mesh_resolution(vertices, faces) -> MeshResolution:
    # median edge length, robust local edge length per face,
    # median dihedral step across smooth (non-sharp) edges
```

Config additions:

```
curve_fit_tol_rel: float = 0.15    # accept RMS up to 15% of local edge length
curve_fit_tol_abs: float = 0.02    # floor (fine meshes shouldn't over-tighten)
```

Fitters use `tol = max(curve_fit_tol_abs, curve_fit_tol_rel * local_edge)`.
Subtlety: the chordal sagitta of a correctly fit surface is one-sided (facets
bulge inward), so use RMS-about-fit (~half the sagitta) and recentre the
surface radius by the sagitta bias. The coverage/centroid guards in
`_fit_circle_for_facets` (lines ~300–312) must likewise scale with edge
length, not the current `0.05*radius + 0.05`.

This alone recovers coarse-mesh cylinder fits currently rejected — a
prerequisite, not a milestone.

### 1.2 Tangency prior — near/far decision

Tangency defect = angle between the fitted surface's normal at the shared
boundary edge and the adjacent plane's normal (0° at a true tangent blend).

Threshold is resolution-scaled, not fixed: even a true tangent blend reads a
defect of order `median_dihedral_deg/2` on a coarse mesh.

```
tangency_threshold_deg = max(tangency_floor_deg, k * median_dihedral_deg)
# tangency_floor_deg ~ 3°, k ~ 1.0
```

defect ≤ threshold → tangent (design intent): snap, derive radius (1.3).
defect > threshold → intentional: keep best-effort fit radius, no snap.
Store `tangent: bool`, `radius_source: "tangency"|"fit"` on the fit object
for the builder and QA reporting.

### 1.3 Radius-from-tangency math

**Fillet between two planes** (straight-edge fillet → cylinder section).
Planes with unit outward normals n₁, n₂ meeting at interior dihedral θ. A
tangent cylinder of radius R has its axis on the bisector plane at distance
`d = R / sin(θ/2)` from the edge line, offset into the material.

1. Solve the edge line L = P₁∩P₂ from the analytic planar regions (exact,
   immune to facet noise).
2. The fitted arc's inliers give rough R and bisector b = normalize(−n₁−n₂).
3. Snap: `axis_pt = L_point + b·R/sin(θ/2)`, direction = L's direction.
   Choose R by a 1-D search minimizing vertex residual to the tangent
   cylinder whose axis is fixed by R via `d = R/sin(θ/2)` — cheap and
   chord-bias-corrected (near and far vertices both constrained).

The derived R uses the exact analytic planes, so it does not under-read the
way the free Kasa fit does.

**Fillet between a plane and a cylinder** (blend of a boss/bore into a flat):
the exact fillet is a **torus**. Tangent-to-plane fixes the torus center
plane at height `r` above the flat; tangent-to-cylinder fixes
`R_major = R_cyl ± r` (− concave into a bore, + convex around a boss). Solve
r by the same 1-D fit with axis/coaxiality locked to the already-detected
`Cylinder` — a strong reason to run fillet detection *after* cylinder
detection.

## 2. Segmentation: detecting bands

Options evaluated:

- Principal-curvature clustering: most general, but per-vertex curvature
  tensors are noisy on coarse STLs (our target regime). Reject for v1.
- Constant-cross-section sweep detection: powerful for curved walls; defer
  to Milestone 4.
- **Dihedral-angle chains: chosen.** `_angled_axis_candidates` (fitting.py)
  already classifies facets as "curved" (edge-neighbor normal differs by
  more than coplanar tol but less than `curve_max_deg`) and groups them with
  `_connected_components`.

After `segment_planar` claims flats and `detect_cylinders`/`detect_cones`
claim full round features, group the remaining curved facets into connected
smooth regions and classify by cheap signatures:

- **Band vs. cap topology:** a fillet/chamfer band is a strip bordering
  exactly two other regions, long-and-thin (high perimeter²/area). A corner
  blend is a small patch bordering ≥3 regions. Computable from existing edge
  adjacency + region labels.
- **Chamfer vs. fillet:** a chamfer is planar (residual chamfers are those
  whose 1–2-facet width falls below `min_region_facets` or fails loop
  extraction); a fillet's normals rotate monotonically across the strip.
- **Sphere signature:** `_region_axis` (fitting.py:167) returns `None` when
  normals fan out in all directions — a positive sphere signal on a compact
  region. Reuse directly.

New: `segmentation.segment_smooth_bands(vertices, faces, claimed, config)
-> list[SmoothRegion]` returning curved components tagged with bordering
region labels and a class hint (`band|cap|blend`). Pure numpy (keeps the
FreeCAD-free rule, DESIGN.md line 36).

**Coarse-mesh guard (1–2-facet-wide bands):** (a) fit using the band's
boundary vertices against the two adjacent analytic planes — the tangency
constraint needs only the two planes + strip extent, not interior rows;
(b) planar decimation preserves band density (§5).

## 3. Fitting: new surface types

Follow the `Cylinder`/`Cone` dataclass + `detect_*` + `_fit_*` pattern in
`fitting.py`:

- **`Torus`**: `center, axis_dir, major_radius, minor_radius, u_range,
  v_range, face_indices, tangent, outward`. `detect_fillets_torus(...)` runs
  after `detect_cylinders`: for each cylinder, find adjacent smooth bands
  coaxial with its axis, fit r via the tangency 1-D solve (plane-cylinder
  case). Trim from the band's angular + host's axial extent.
- **`FilletCylinder`**: reuse `Cylinder` with `coverage < 0.5` allowed,
  tagged `tangent`/`radius_source`. `detect_fillets_straight(...)` handles
  the plane-plane case. Critical: the candidate axis is the **edge
  direction** between two planes, which `_candidate_axes` does not generate
  (face normals + PCA only) — drive this fitter from adjacent-plane pairs
  rather than the axis-sweep loop.
- **`Sphere`**: `center, radius, trim, face_indices, tangent`.
  `detect_spheres(...)` on compact smooth regions where `_region_axis is
  None`; algebraic sphere fit (4-parameter linear analogue of
  `_fit_circle_2d`: solve `2x·cx+2y·cy+2z·cz+c = x²+y²+z²`), sagitta-bias
  correction, resolution-scaled tolerance. Corner blends: snap radius to
  tangency with the three adjacent fillets/planes if near-tangent.

All three share the §1 tolerance/tangency helper.

## 4. Building: trimmed STEP faces

Mirror `_cylinder_face`/`_cone_face` style in `builder.py`:

- **Sphere:** `Part.Sphere()` → `.Center`, `.Radius` →
  `surf.toShape(uMin,uMax,vMin,vMax)`; reverse per `outward`. Boolean tool:
  `Part.makeSphere`.
- **Torus:** `Part.Toroid()` (verify exact class name per FreeCAD version)
  with `.Center`, `.Axis`, `.MajorRadius`, `.MinorRadius` → `toShape(...)`.
  Boolean tool: `Part.makeTorus(R, r, center, axis, a1, a2, a3)`. A
  straight-edge fillet stays `_cylinder_face` with a partial `[0, arc]`
  u-range.

**Boundary/loop handling (`boundary.py`):** generalize `_analytic_circles` +
`_match_loop_to_circle` (builder.py:41–78) to non-circular analytic edges. A
torus fillet's trim edges are two circles (tangent lines to plane and host
cylinder) — extends cleanly; fillet end caps and sphere/torus seam edges need
`_match_loop_to_arc`/`_match_loop_to_analytic_edge`. Keep the "match by
centroid + radius + axis-coincidence" strategy.

**Watertightness — do NOT rely on sewing.** Follow the boolean pattern:

- Concave fillet (bore / inner corner): **cut** a torus/cylinder tool of the
  derived radius plus `_clean_cut_eps` clearance, padded past the ends
  (`_cut_pad`); the analytic fillet surface is the cut wall.
- Convex fillet (boss / outer edge): **fuse** the exact rounded solid via
  `_guarded_fuse` (added-volume guard).
- Sphere caps: same cut/fuse dichotomy via `Part.makeSphere`.

Each step goes through `_try_boolean_step` (reverts on invalidity) so one bad
fillet never breaks the solid. Extend the rogue-radius artifact check
(builder.py:680) to torus/sphere faces.

**Torus self-intersection risk:** reject/clamp when `R_major <= r + margin`
(spindle torus); reject fillets whose derived R is below a resolution-scaled
floor (sub-facet fillets can't be built cleanly).

## 5. Pipeline integration and decimation

Detector order (dependencies matter), in both builders:

1. `detect_cylinders` (unchanged) — fillets attach to cylinders.
2. `detect_cones` (unchanged).
3. `detect_fillets_torus` (needs cylinders) and `detect_fillets_straight`
   (needs planar regions).
4. `detect_spheres` on remaining compact smooth regions.
5. `segment_planar` on the rest, then gap-fill.

Boolean tier: apply cuts/fuses in the existing feature-ordering discipline;
fillets after their host cylinder's cut so the fillet trims the
already-analytic wall.

**Decimation interaction:** `decimate_planar` (pymeshlab `planarquadric`,
`preserveboundary`) favors coplanar collapses — it collapses flats and leaves
fillet/curve facets dense. Good for band detection (sharpens the
dihedral-chain contrast). Verify at implementation that `preserveboundary`
doesn't over-smooth the fillet-to-flat crease; if it does, the tangency prior
compensates (radius re-derived from exact flats). Fitters run on the
decimated mesh in the boolean tier (`build_boolean_clean_solid` receives
`dv, df`).

**Config additions** (default on, conservative): `detect_fillets: bool`,
`detect_spheres: bool`, `curve_fit_tol_rel`, `curve_fit_tol_abs`,
`tangency_floor_deg`, `tangency_k`, `min_fillet_radius`,
`torus_min_major_over_minor`.

## 6a. Post-M1 diagnosis update (2026-07-03) — REVISED ORDER

Facet-level classification of the actual residual on the four worst parts
(20,455 skipped facets) after M1 landed:

| class → milestone | facets | share |
|---|---|---|
| swept/extruded curved walls → M4 | 12,365 | 60.4% |
| planar faces failing to build (`OCCError: Not planar`) → bug fix | 3,988 | 19.5% |
| <10-facet tail (mostly M4-family micro-walls) | 2,284 | 11.2% |
| sphere-ish → M3 | 1,128 | 5.5% |
| torus/fillet-around-hole → M2 | 468 | 2.3% |
| missed cylinder fits → tuning | 222 | 1.1% |

Consequences:
- **Revised order: planar-build bug fix → M4 (curved walls) → M3 (spheres,
  incl. fan_panel's R≈74.7 grille dome) → M2 (torus, smallest payoff).**
- The `Not planar` bug: `segment_planar` admits facets within
  `dist_tol=0.01`, but loop vertices exceed OCC's face planarity tolerance;
  fix by projecting loop points onto the fitted plane before `Part.Face`.
- **Metric: RTAF (Residual Tessellation Area Fraction)** = area of planar
  faces in smooth chains (≥3 faces, normal steps 0.5°–30°) / total area of
  the output solid. Matches human ranking (drive_bay 0.911, fan_panel 0.493,
  tweezer 0.311, USB-holder 0.057, cross_brace 0.009, rack_mount 0.004,
  hexwall 0.000, plate_with_holes 0.000). This is the primary scoreboard
  going forward; `skipped_facets` undercounts (tessellation chains rebuilt
  as thin planar strips don't count as skipped but read as faceted).
  Prototype: scratchpad diagnosis/step_strips.py.

## 6. Original milestones (payoff ranking superseded by §6a)

1. **M1 — Resolution-scaled tolerances + tangency prior + straight-edge
   fillets/chamfers.** No new OCC surface type (partial-arc `_cylinder_face`
   + chamfer planes). Highest payoff-per-effort: dominates the rack/panel
   parts — labstack_fan_panel_2u (10,732/41%), labstack_drive_bay_525
   (9,510/66%), rack mounts, cross_brace (11,780/22%). Metric: total corpus
   `skipped_facets` + per-file before/after. Landable alone.
2. **M2 — Torus fillets (plane↔cylinder blends).** `Torus` fitter +
   `_torus_face` + boolean cut/fuse. Targets USB-holder (48%), insert-cable
   holder (50%), topper-cable, countersunk/boss parts. Depends on M1's
   tangency helper. Metric: torus faces validate at derived radii, bbox
   drift <1%, artifact-free rate.
3. **M3 — Sphere caps / corner blends.** `Sphere` fitter + `_sphere_face`.
   Targets honeycomb-shelve (39%) and the user-flagged "3x1 Tweezer Mount"
   benchmark. Metric: benchmark `skipped_facets` drops materially; sphere
   radius matches ground truth on a new synthetic sample.
4. **M4 — Curved walls (swept/constant-cross-section).** Hardest; may land
   as a B-spline-face fallback (DESIGN.md roadmap) rather than analytic.
   Targets remaining USB-holder / cross_brace curved-wall residual. Metric:
   watertight rate must not regress.

Each milestone lands behind its config flag with the corpus `skipped_facets`
sweep as the shared regression metric.

## 7. Risk register

- OCC sewing tolerance between analytic patches → prefer boolean cut/fuse;
  `_try_boolean_step` reverts; extend rogue-radius check.
- Torus self-intersection at small r → reject `R_major <= r + margin`;
  resolution-scaled `min_fillet_radius` floor; fall back to faceted.
- 1–2-facet-wide bands → tangency prior needs only the adjacent planes +
  strip extent; decimation preserves band density.
- Tangency defect blurred by decimation crease smoothing → radius re-derived
  from exact flats; verify `preserveboundary` behavior.
- False-positive fillets on organic meshes → reuse proven guards (coverage,
  centroid-radius, RMS, `_region_axis is None`, `_guarded_fuse` volume
  guard); conservative defaults.
- Ordering dependency → explicit detector ladder in both builders.

## Critical files

`fitting.py`, `builder.py`, `segmentation.py`, `config.py`, `boundary.py`;
pipeline wiring in `pipeline.py` (new detectors slot into the existing tier
ladder without structural change).
