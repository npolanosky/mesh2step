# Milestone 5 Design: Helical & Patterned Features (threads, gears, knurling)

Status: approved design, pre-implementation. Targets the remaining big
unmodeled classes: parametric_bottle_cap (threads, RTAF 0.63),
gear_box_gear_v2 (involute teeth, RTAF 0.73), knurled_knob (diamond knurl).

## Governing decision

**Threads and knurling are suppressed to their nominal cylinder with
pitch/pattern captured as metadata — not rebuilt as true helical B-reps.**
This is standard reverse-engineering practice (Verisurf, Core77). The OCC
true-helix path (Part.makeHelix + BRepOffsetAPI_MakePipe) exists but a
helical sweep booleaned against a faceted base is OCC's worst
self-intersection case (G1-spine requirement, coincident-edge pitfalls —
the community workaround is a 0.01 mm profile offset). Unacceptable vs the
watertight-non-negotiable prior. True helical B-rep = deferred M6 stretch
behind `build_true_threads=False`.

All three ship-paths funnel into existing machinery
(`_boolean_clean_cylinder`, closed-wire extrude generalizing
`_swept_arc_lens_tool`), through `_try_boolean_step` with per-feature revert
(graceful degradation prior).

## 1. Threads — `detect_threads` in fitting.py (M5.2)

Seeded from each detected `Cylinder` (threads are coaxial — reuse the axis).
Candidate band: unclaimed facets with radial distance within R±~0.6·pitch of
the wall, axially inside the cylinder extent.

**Helix invariant fit** (the one new numpy fitter): per-facet angle
φ = atan2(rel·v, rel·u) unwrapped along the connected band, axial z = rel·axis;
single-start thread satisfies z = (pitch/2π)·φ + z₀. Least-squares fit
z = a·φ + b. Accept when: residual RMS ≤ `_local_tol` (resolution-scaled);
pitch = 2π·a with pitch/radius in `thread_min_pitch_rel..thread_max_pitch_rel`
(~0.05..1.5); **angular coverage ≥ ~1.5 turns** (primary false-positive
guard — one turn is a chamfer/ramp, not a thread). Multi-start count from
residual clustering modulo pitch; handedness from sign(a) — metadata only.

Dataclass `Thread`: axis_point, axis_dir, nominal_radius, axial_min/max,
pitch, starts, handedness, crest_radius, root_radius, rms, face_indices,
is_internal (from the existing boss/hole outward-normal test).

**Reconstruction (ship):** suppress to a cylinder at pitch diameter
(crest+root)/2 via the proven `_boolean_clean_cylinder` (external → fuse,
internal → cut). Pitch/diameter/starts/hand into `stats["threads"]` +
metadata channel. Fallback: leave faceted (status quo).

Risks: spiral/knurl false positives → coverage+RMS+pitch gates; pitch
under-read on coarse meshes → optional snap (harmless — metadata only);
ladder order threads-after-host-cylinder (fillet discipline).

## 2. Gear teeth — whole-outline extrusion (M5.3)

The gear IS an extrusion; M4 already recognizes the shape but
`_is_repeated_arc_pattern` deliberately drops it (per-arc lens ops are
O(arcs×base) — never converges). **Ship: claim it wholesale** — route
repeated-arc *closed* profiles to new `_boolean_clean_gear`: assemble the
full 2D cross-section from `SweptProfile.segments` (lines + arcs + B-spline
spans; involute flanks naturally become splines) into ONE closed Part.Wire →
Part.Face → extrude along axis → ONE `_guarded_fuse`/cut. O(base) once, not
per arc; same trusted primitive chain as swept walls.

New `SweptProfile.whole_extrusion` flag; classifier = `_is_repeated_arc_pattern`
fires AND profile closed AND outline roughly centered on the axis. Loose
guard — a splined shaft is also fine to extrude whole.

Rejected: one-tooth-patterned-N-times (N seam risks, tooth-pitch
segmentation complexity, no real benefit). Cost: single ~seconds boolean at
12k base; `gear_max_profile_segments` (~2000) sanity ceiling.

Risks: noisy outline → wire/face isValid precheck + revert; jagged short
arcs → `gear_flanks_as_spline=True` forces spline spans; central bore —
ladder discipline: bore cut after gear fuse.

## 3. Knurling — `detect_knurling` (M5.1, cheapest — implement first)

Signature: high-frequency micro-roughness on a cylindrical band — small mean
radial deviation from the wall, **bimodal high-variance normal tilt** (two
crossing helix families = diamond; discriminates from thread's single
family), very high facet density. Eigen-decomposition of tilt directions à
la `_region_axis`. Dataclass `KnurlBand`: axis, nominal_radius, extent,
pattern (diamond/straight), pitch_estimate, face_indices.

**Reconstruction (ship):** suppress to median-ρ mid-surface cylinder via
`_boolean_clean_cylinder`; pattern metadata in `stats["knurling"]`.
Reject: bump reconstruction (absurd cost). Thread/knurl misclassification is
harmless — both suppress to cylinder; only the metadata label differs.

## Ladder placement (both builders)

cylinders → cones → **threads** → **knurling** → fillets → dome spheres →
freeform sheets → swept (repeated-arc+closed → **gear whole-extrusion**,
else lens ops) → blend spheres. Threads/knurl claim early so their bands
never mis-fit as swept walls or domes.

## Ordering & metrics

- **M5.1 knurling** (band classifier + median-radius fuse; proves the
  suppress+metadata pattern). Metric: knurled_knob RTAF.
- **M5.2 threads** (helix fit; reuses M5.1 path). Metric: bottle_cap RTAF
  0.63 → ~0 band; pitch/diameter in stats.
- **M5.3 gear** (closed-wire builder + routing). Metric: gear RTAF 0.73 →
  low; bore survives; cost under budget.
- **M6 (deferred, off):** true helical B-rep.

Config flags: detect_threads, detect_knurling, reconstruct_gears,
build_true_threads(False), thread_min_turns, thread_min/max_pitch_rel,
knurl_min_normal_bimodality, gear_max_profile_segments,
gear_flanks_as_spline. Shared regression metric: corpus RTAF sweep +
watertight rate, per-feature revert throughout.

## Metadata channel (cross-cutting)

v1: sidecar `<name>_features.json` next to the STEP carrying
{threads, knurling, gears} (pitch, diameters, starts, hand, pattern, tooth
count). STEP face-name annotation later if `export_step` supports it.

Sources: Verisurf scan-to-CAD practice; Core77 thread RE; cadnauseam thread
offset trick; FreeCAD Part API (makeHelix/MakePipe); OCC MakePipe G1-spine
requirement.
