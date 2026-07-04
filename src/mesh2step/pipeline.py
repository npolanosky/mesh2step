"""End-to-end orchestration: STL path in, STEP path out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ConversionConfig
from .mesh_io import load_stl

# Target face count for the boolean clean-up base. Each cut is O(base faces);
# ~12k keeps holes accurate while making dozens of cuts complete in ~1–2 min.
DEFAULT_BOOLEAN_TARGET = 12000


@dataclass
class ConversionResult:
    output_path: Path          # primary file written
    method: str                # e.g. "reconstructed", "boolean-clean", "faceted"
    stats: dict
    outputs: list | None = None  # all files written (>1 when dual output)


def _suffixed(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def _write_features_sidecar(output_path: Path, stats: dict, progress) -> None:
    """Write the ``<name>_features.json`` metadata sidecar (M5 design §metadata).

    Carries the helical/patterned feature metadata — threads (pitch/starts/hand/
    crest/root), knurling (pattern/diameter/pitch), gears (segments/extent) — next
    to the STEP so downstream tools have the parametric intent even though the
    geometry ships as suppressed cylinders / whole extrusions. Best-effort: a
    write failure never breaks the conversion. Only written when there is at least
    one such feature."""
    import json

    features = {
        "threads": stats.get("threads") or [],
        "knurling": stats.get("knurling") or [],
        "gears": stats.get("gears") or [],
    }
    if not any(features.values()):
        return
    sidecar = output_path.with_name(output_path.stem + "_features.json")
    try:
        sidecar.write_text(json.dumps(features, indent=2))
        stats["features_sidecar"] = str(sidecar)
        n = sum(len(v) for v in features.values())
        progress(f"Wrote {sidecar.name} ({n} helical/patterned feature(s))")
    except OSError as exc:
        progress(f"Feature sidecar not written ({exc})")


def convert(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: ConversionConfig | None = None,
    on_progress=None,
) -> ConversionResult:
    """Convert an STL file to a STEP solid.

    Tries surface reconstruction first; falls back to a faceted solid if
    reconstruction can't produce geometry. Requires FreeCAD at runtime.

    ``on_progress`` is an optional ``callable(str)`` invoked with human-readable
    stage messages (used by the GUI to stream progress).
    """
    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    config = config or ConversionConfig()
    # Fully-closed relies on the boolean base being small; decimate the whole
    # pipeline by default so reconstruction is fast (and may even close) and the
    # boolean base is tractable. Users can override with an explicit target.
    if config.full_closed and config.decimate_target_faces is None:
        import dataclasses

        config = dataclasses.replace(config, decimate_target_faces=DEFAULT_BOOLEAN_TARGET)
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".step")
    output_path = Path(output_path)

    # Make FreeCAD importable, then pull in the builder (which needs it).
    progress("Locating FreeCAD")
    from .freecad_env import ensure_freecad

    ensure_freecad(config.freecad_bin)
    from . import builder

    # Multi-body dispatch: a mesh with several disjoint bodies (a print-in-place
    # hinge, a snap-fit lid+base). The connected components are split up front —
    # cheaply, via union-find — so every heavy step (repair, self-intersection
    # union, decimation, reconstruction, boolean clean-up) runs per body on the
    # SMALLER meshes instead of once on the whole combined mesh. ``multibody_mode``
    # then decides what to do with the split:
    #
    #   "separate": convert each body independently -> a STEP compound of N solids.
    #   "combine":  union the bodies into ONE solid -> the ordinary single-body path.
    #   "auto":     combine only when the bodies actually TOUCH — their bboxes
    #               overlap AND they share near-coincident vertices along a seam
    #               (one part exported as multiple shells). Bodies with a clearance
    #               gap (a print-in-place hinge pin inside its knuckle: boxes
    #               interpenetrate but no coincident faces) stay SEPARATE so a
    #               functional assembly is never fused. Falls back to separate on
    #               any combine failure, so it never regresses.
    #
    # A single-body mesh falls straight through to the ordinary path below.
    if config.multi_body:
        raw_v, raw_f = load_stl(input_path, weld_tol=config.weld_tol)
        from .mesh_io import (
            bodies_bbox_overlap,
            bodies_share_coincident_vertices,
            filter_junk_bodies,
            split_components,
        )

        components = split_components(raw_v, raw_f)
        # Drop degenerate junk components (tiny slivers, stray shells) before the
        # dispatch: they flip "auto" to "separate" on a real single-body part and
        # can hard-abort the sew later. If everything but junk remains, the file
        # is single-body and takes the ordinary path below.
        if len(components) > 1 and config.min_body_facets > 0:
            components, dropped = filter_junk_bodies(
                components, min_facets=config.min_body_facets,
                min_area_frac=config.min_body_area_frac)
            if dropped:
                progress(f"Ignored {dropped} junk body/bodies "
                         f"(< {config.min_body_facets} facets or negligible area); "
                         f"{len(components)} real body/bodies remain")
        if len(components) > 1:
            mode = config.multibody_mode
            if mode == "auto":
                # Conservative: combine only shells that genuinely touch (bbox
                # overlap is a cheap pre-filter; a shared coincident seam is the
                # deciding signal). A clearance-gap assembly stays separate.
                touching = (bodies_bbox_overlap(components)
                            and bodies_share_coincident_vertices(
                                components, tol=config.multibody_combine_weld))
                mode = "combine" if touching else "separate"
                progress(f"Multi-body ({len(components)} bodies): auto -> {mode} "
                         f"(bodies {'share a coincident seam' if touching else 'are disjoint / gap-separated'})")
            else:
                progress(f"Multi-body ({len(components)} bodies): mode={mode}")

            if mode == "combine":
                combined = _combine_bodies_to_stl(
                    raw_v, raw_f, input_path, config, progress)
                if combined is not None:
                    # Union succeeded: run the ordinary single-body pipeline on the
                    # fused mesh (multi_body off so it can't re-split).
                    import dataclasses

                    combined_stl, combine_report = combined
                    cfg1 = dataclasses.replace(config, multi_body=False)
                    try:
                        res = convert(combined_stl, output_path, cfg1, on_progress=progress)
                    finally:
                        import shutil

                        shutil.rmtree(combined_stl.parent, ignore_errors=True)
                    res.stats["multibody_mode"] = "combine"
                    res.stats["multibody_combine"] = combine_report
                    return res
                progress("Multi-body combine unavailable/failed — "
                         "falling back to separate per-body conversion")

            return _convert_multibody(
                components, input_path, output_path, config, builder, progress)

    # Weld in raw units, then scale to millimetres (STEP is always mm). Welding
    # first keeps weld_tol meaningful regardless of the source unit scale.
    prep_report = None
    if config.repair_mesh:
        progress("Preparing mesh (repair)")
        from . import meshprep

        vertices, faces, prep_report = meshprep.load_and_prepare(str(input_path), config)
        if prep_report.get("before_facets") != prep_report.get("after_facets"):
            progress(f"Mesh: {prep_report['before_facets']:,} -> "
                     f"{prep_report['after_facets']:,} facets")
    else:
        progress("Loading and welding mesh")
        vertices, faces = load_stl(input_path, weld_tol=config.weld_tol)
    if faces.size == 0:
        raise ValueError(f"{input_path} contained no usable triangles")
    progress(f"{len(faces):,} triangles, {len(vertices):,} vertices")
    scale = config.scale_to_mm
    if scale != 1.0:
        progress(f"Scaling {config.source_units} -> mm (x{scale:g})")
        vertices = vertices * scale

    # Planar-preserving decimation: collapses over-tessellated flats while
    # keeping holes/curves dense — shrinks the file and can improve detection.
    dec_report = None
    if config.decimate_target_faces:
        from . import meshprep

        target = int(config.decimate_target_faces)
        if len(faces) > target * 1.2:
            progress(f"Decimating {len(faces):,} -> ~{target:,} faces (planar-preserving)")
            vertices, faces, dec_report = meshprep.decimate_planar(vertices, faces, target)
            progress(f"Decimated to {len(faces):,} faces")

    input_dims = sorted((vertices.max(axis=0) - vertices.min(axis=0)).tolist(), reverse=True)

    method = "faceted"
    stats: dict = {"faces_in": int(len(faces))}
    if prep_report is not None:
        stats["mesh_prep"] = prep_report
    if dec_report is not None:
        stats["decimation"] = dec_report

    if not config.faceted:
        try:
            shape, built = builder.build_reconstructed_solid(
                vertices, faces, config, on_progress=progress
            )
            stats.update(built)  # merge (don't drop mesh_prep / faces_in)
            method = "reconstructed"
        except Exception as exc:  # noqa: BLE001 - reconstruction is best-effort
            progress(f"Reconstruction failed ({exc}); using faceted fallback")
            stats["reconstruction_error"] = str(exc)
            shape = builder.build_faceted_solid(vertices, faces)
    else:
        progress("Building faceted solid")
        shape = builder.build_faceted_solid(vertices, faces)

    # Organic multi-patch (Candidate A): when the after-analytic reconstruction is
    # mostly residual tessellation (a genuinely organic body — a sculpted cat, an
    # ergonomic handle), attempt a whole-body quad-patch B-spline network that
    # de-facets the whole surface. STRICT rollback: adopt only when it produces a
    # watertight solid that LOWERS the RTAF and stays bbox-stable; on any failure
    # keep the existing reconstruction (never regress). Behind organic_multipatch;
    # requires the optional remesher (declines gracefully when absent).
    if method == "reconstructed" and not config.faceted and config.organic_multipatch:
        try:
            from . import organic

            # RTAF of the after-analytic reconstruction is the routing signal, and
            # isn't computed until quality assessment — measure it here so the gate
            # (and the improvement check) can use it. Computed even on an OPEN
            # reconstruction: a mostly-faceted open shell is exactly the organic
            # case (the analytic tiers couldn't claim the body).
            rtaf_before = stats.get("rtaf")
            if rtaf_before is None:
                try:
                    rtaf_before = builder.compute_rtaf(shape, config).get("rtaf")
                    stats["rtaf"] = rtaf_before
                except Exception:  # noqa: BLE001
                    rtaf_before = None
            if organic.should_attempt(stats, faces, config):
                progress("Organic multi-patch: mostly-organic body — attempting "
                         "whole-body quad-patch reconstruction")
                org_shape, org_stats = organic.build_organic_shell(
                    vertices, faces, config, on_progress=progress)
                stats["organic"] = org_stats
                if org_shape is not None and _is_solid(org_shape):
                    rtaf_after = builder.compute_rtaf(org_shape, config).get("rtaf")
                    org_stats["organic_rtaf"] = rtaf_after
                    improves = (rtaf_before is None or rtaf_after is None
                                or rtaf_after < rtaf_before - 1e-4)
                    bbox_ok = _bbox_stable(shape, org_shape, config)
                    if improves and bbox_ok:
                        progress(f"Organic multi-patch: adopted (RTAF "
                                 f"{rtaf_before} -> {rtaf_after}, "
                                 f"{org_stats.get('organic_patches')} patches)")
                        shape = org_shape
                        method = "organic-multipatch"
                        stats["rtaf"] = rtaf_after
                    else:
                        progress("Organic multi-patch: result did not improve/"
                                 "stayed bbox-stable — keeping existing output")
                else:
                    progress(f"Organic multi-patch: declined "
                             f"({org_stats.get('organic_reason', 'no solid')}) — "
                             f"keeping existing output")
        except Exception as exc:  # noqa: BLE001 - organic tier is best-effort
            progress(f"Organic multi-patch skipped ({exc})")
            stats["organic_error"] = str(exc)

    # Keep the clean (possibly open) reconstruction so we can also emit it when
    # the watertight version can only be produced with artifacts.
    clean_shape = shape if method in ("reconstructed", "organic-multipatch") else None
    dual = False
    # Set when the boolean back-off ladder already wrote (and re-validated) the
    # adopted solid to ``output_path`` — the final single-file export is skipped.
    boolean_exported = False

    # Fully-closed toggle: upgrade the (possibly open) reconstruction to a
    # watertight solid via boolean clean-up. Priority: (1) watertight AND
    # artifact-free is ideal; (2) watertight with some artifacts — still emit it,
    # but ALSO emit the clean (open) reconstruction so the user has both and can
    # pick; (3) an open (but clean) reconstruction is the last resort.
    if config.full_closed and not _is_solid(shape):
        progress("Fully-closed: boolean clean-up (cut + fuse-back analytic holes)")
        # Use the ORIGINAL welded mesh (repair can leave it non-watertight, which
        # breaks the base solid), then decimate: each boolean cut costs O(base
        # faces), so a dense base makes this minutes. Decimation keeps holes
        # dense (planar-preserving) so detection/accuracy hold while the base
        # shrinks enough for the cuts to be fast.
        bverts, bfaces = load_stl(input_path, weld_tol=config.weld_tol)
        if scale != 1.0:
            bverts = bverts * scale
        from . import meshprep

        # Overlapping-body resolution MUST come before decimation: manifold3d
        # needs the raw (still-manifold) topology, and decimating a self-
        # intersecting mesh produces defects it can't digest. Without this, a
        # mesh exported without a final boolean union (a clip modelled through
        # its panel, tabs interpenetrating a base) can never form a valid base.
        if meshprep.has_self_intersections(bverts, bfaces):
            progress("Base mesh self-intersects; unioning overlapping bodies (manifold)")
            resolved = meshprep.resolve_self_intersections(
                bverts, bfaces, on_progress=progress)
            if resolved is not None:
                bverts, bfaces, resolve_report = resolved
                stats["self_intersection_resolve"] = resolve_report
                progress(f"  unioned {resolve_report['bodies']} bodies "
                         f"({resolve_report['faces_in']:,} -> "
                         f"{resolve_report['faces_out']:,} facets)")
            else:
                progress("  union unavailable/failed; continuing with raw mesh")
        target = config.decimate_target_faces or DEFAULT_BOOLEAN_TARGET
        # Decimation back-off ladder: quadric collapse can break watertightness
        # on some meshes, so if the boolean base won't validate at the target,
        # retry at twice the target and finally with the undecimated mesh —
        # slower cuts beat losing the watertight result altogether.
        rungs = [t for t in (target, 2 * target) if len(bfaces) > t * 1.2] + [None]
        # Planarity-damage back-off (task §1): on a coarse organic scan, quadric
        # decimation warps genuinely-flat regions past the coplanar gate, so the
        # 12k base ships "everything faceted" even though it is watertight. Measure
        # the RAW mesh's area-weighted planar coverage once here; after each rung's
        # decimation we compare the rung's coverage to it, and a rung that dropped
        # the flat coverage below the ratio threshold is treated as FAILED (backed
        # off to a gentler rung) exactly like the export-revalidation criterion.
        # Only meaningful when there is an actual decimation rung to gate (a rung
        # list of just [None] means the mesh is already small enough — nothing to
        # damage, so skip the ~2 s raw-coverage segmentation entirely).
        raw_planar_coverage = None
        if (config.planarity_damage_check
                and config.planarity_damage_min_ratio is not None
                and any(t is not None for t in rungs)):
            try:
                from .segmentation import planar_coverage

                info = planar_coverage(
                    bverts, bfaces, config, config.planarity_min_region_facets)
                raw_planar_coverage = info["coverage"]
                stats["planarity_raw_coverage"] = round(raw_planar_coverage, 4)
                progress(f"Planar coverage (raw base): {raw_planar_coverage:.2f} "
                         f"of area in {info['n_big_regions']} large flats")
            except Exception as exc:  # noqa: BLE001 - metric must not break convert
                progress(f"Planar-coverage metric skipped ({exc})")
                raw_planar_coverage = None
        # Cost ceiling: the fully-closed tier lifts boolean_max_base_faces to let
        # the cuts run on dense meshes, but only up to this many faces. Above it
        # (decimation unavailable, or a base still huge after decimation) an
        # unbounded boolean run grinds for many minutes — so we cap the base and
        # fall through to the faceted watertight solid (tier 3) instead.
        ceiling = config.fully_closed_boolean_ceiling_faces
        try:
            import dataclasses

            bcfg = dataclasses.replace(config, boolean_max_base_faces=ceiling)
            bshape, built2 = None, None
            for i, tgt in enumerate(rungs):
                if tgt is not None:
                    progress(f"Decimating boolean base {len(bfaces):,} -> ~{tgt:,} faces")
                    dv, df, _ = meshprep.decimate_planar(bverts, bfaces, tgt)
                    progress(f"Boolean base: {len(df):,} faces")
                else:
                    dv, df = bverts, bfaces
                    if i > 0:
                        progress(f"Using undecimated base ({len(df):,} faces)")
                if ceiling is not None and len(df) > ceiling:
                    progress(f"  base {len(df):,} faces exceeds boolean cost ceiling "
                             f"({ceiling:,}); skipping boolean clean-up "
                             f"(decimation ineffective/unavailable), "
                             f"falling through to faceted watertight solid")
                    stats["boolean_ceiling_hit"] = {
                        "base_faces": int(len(df)), "ceiling": int(ceiling)}
                    if tgt is None:
                        break
                    continue
                # Planarity-damage gate: if this decimation rung warped the flats
                # (coverage ratio below the threshold) AND a gentler rung remains,
                # skip the (expensive) boolean build and back off — the check is a
                # cheap segmentation, far cheaper than a doomed boolean run whose
                # output would ship faceted. The last rung (undecimated / None)
                # never trips it: it is the gentlest available, so we always try it.
                if (raw_planar_coverage is not None and raw_planar_coverage > 0
                        and tgt is not None and i < len(rungs) - 1):
                    try:
                        from .segmentation import planar_coverage

                        cov = planar_coverage(
                            dv, df, config, config.planarity_min_region_facets
                        )["coverage"]
                    except Exception:  # noqa: BLE001 - never block on the metric
                        cov = None
                    if cov is not None:
                        ratio = cov / raw_planar_coverage
                        rung_label = f"~{tgt:,}"
                        # Only back off to a rung that can still BUILD a boolean-clean
                        # solid. Every gentler rung ahead that is above the boolean
                        # cost ceiling gets skipped to a plain faceted solid — worse
                        # than THIS rung's watertight boolean-clean output even with
                        # its warped flats. So if no reachable gentler rung stays
                        # under the ceiling, keep this one (a faceted-flat boolean
                        # solid beats a fully-faceted one).
                        gentler_buildable = True
                        if ceiling is not None:
                            gentler_buildable = any(
                                (len(bfaces) if nxt is None else min(nxt, len(bfaces)))
                                <= ceiling
                                for nxt in rungs[i + 1:])
                        if ratio < config.planarity_damage_min_ratio and gentler_buildable:
                            progress(
                                f"  rung {rung_label}: decimation warped flats "
                                f"(planar coverage {cov:.2f} vs raw "
                                f"{raw_planar_coverage:.2f}, ratio {ratio:.2f} < "
                                f"{config.planarity_damage_min_ratio}); backing off "
                                f"to a gentler rung to preserve flats")
                            stats.setdefault(
                                "planarity_damaged_rungs", []).append(
                                {"rung": rung_label, "base_faces": int(len(df)),
                                 "coverage": round(cov, 4),
                                 "ratio": round(ratio, 4)})
                            continue
                        if (ratio < config.planarity_damage_min_ratio
                                and not gentler_buildable):
                            progress(
                                f"  rung {rung_label}: flats warped (ratio {ratio:.2f}) "
                                f"but every gentler rung exceeds the boolean cost "
                                f"ceiling ({ceiling:,}) and would ship fully faceted; "
                                f"keeping this rung's boolean-clean solid")
                            stats["planarity_kept_damaged_rung"] = {
                                "rung": rung_label, "ratio": round(ratio, 4),
                                "reason": "gentler rungs exceed boolean ceiling"}
                        stats["planarity_winning_rung"] = rung_label
                        stats["planarity_winning_coverage"] = round(cov, 4)
                        stats["planarity_winning_ratio"] = round(ratio, 4)
                try:
                    cand_shape, cand_built = builder.build_boolean_clean_solid(
                        dv, df, bcfg, on_progress=progress
                    )
                except Exception:
                    if tgt is None:
                        raise
                    progress(f"  base not watertight at ~{tgt:,} faces; "
                             f"backing off decimation")
                    continue
                # Export round-trip re-validation: a rung can pass isValid() in
                # memory yet re-read invalid (self-intersecting wires from sliver
                # triangles surface only through the STEP write/read). Treat such
                # a rung exactly like an in-memory failure and back off. We write
                # to the real output path here; on success we keep it (no second
                # export at the end), on failure we overwrite at the next rung.
                if _is_solid(cand_shape) and config.revalidate_export:
                    rung_label = f"~{tgt:,}" if tgt is not None else "undecimated"
                    reval = _export_and_revalidate(
                        builder, cand_shape, output_path,
                        expected_solids=1, config=config, progress=progress)
                    if not reval.get("valid", False):
                        stats.setdefault("export_revalidation_failed_rungs", []).append(
                            {"rung": rung_label, "base_faces": int(len(df)),
                             "reason": reval.get("reason", "unknown")})
                        progress(f"  export re-read INVALID at rung {rung_label} "
                                 f"({reval.get('reason')}); backing off decimation")
                        if tgt is None:
                            # Undecimated also fails the round-trip: nothing gentler
                            # left. Keep it as the boolean candidate anyway (the
                            # quality report will flag it) rather than losing the
                            # watertight result entirely.
                            bshape, built2 = cand_shape, cand_built
                            stats["export_revalidated"] = False
                            break
                        continue
                    stats["export_revalidated"] = True
                    stats["export_revalidation_winning_rung"] = rung_label
                    boolean_exported = True
                bshape, built2 = cand_shape, cand_built
                break
            if _is_solid(bshape):
                shape, method = bshape, "boolean-clean"
                stats.update(built2)
                if built2.get("artifact_free", False):
                    progress("Boolean clean-up: watertight + artifact-free — adopted")
                elif clean_shape is not None:
                    # Watertight but with artifacts, and we have a clean open
                    # version too — emit both, named so it's obvious which is which.
                    dual = True
                    progress(f"Boolean clean-up: watertight but with artifacts at "
                             f"Ø~{built2.get('rogue_radii')}; will emit BOTH a "
                             f"_watertight and a _clean (open) file")
                    stats.setdefault("warnings_extra", []).append(
                        "Watertight version has partial-radius artifacts on intersecting "
                        f"holes (extra cylinder faces near {built2.get('rogue_radii')} mm "
                        "radius); a separate artifact-free open version is also written.")
                else:
                    progress(f"Boolean clean-up: watertight (adopted); residual "
                             f"artifacts at Ø~{built2.get('rogue_radii')}")
                    stats.setdefault("warnings_extra", []).append(
                        "Watertight, but some intersecting holes left partial-radius "
                        f"artifacts near {built2.get('rogue_radii')} mm radius.")
            else:
                stats.setdefault("warnings_extra", []).append(
                    "Boolean clean-up did not produce a watertight solid; kept the "
                    "artifact-free open reconstruction as a last resort.")
        except Exception as exc:  # noqa: BLE001
            progress(f"Boolean clean-up failed ({exc})")
            stats.setdefault("warnings_extra", []).append(f"Boolean clean-up failed: {exc}")

    # Tier 3 (plain faceted solid) is a last resort. Normally we only reach it
    # when there is no usable analytic reconstruction at all (faces_out == 0) —
    # never prefer faceted holes over clean analytic ones just to be closed. But
    # when the boolean tier was skipped by the cost ceiling, the (open) analytic
    # reconstruction cannot be closed the cheap way, so a watertight faceted
    # solid is the right fallback even though reconstruction produced faces.
    if (config.full_closed and not _is_solid(shape)
            and (stats.get("faces_out", 0) == 0 or "boolean_ceiling_hit" in stats)):
        progress("Fully-closed: building watertight faceted solid (last resort)")
        fverts, ffaces = load_stl(input_path, weld_tol=config.weld_tol)
        if scale != 1.0:
            fverts = fverts * scale
        fshape = builder.build_faceted_solid(fverts, ffaces)
        if _is_solid(fshape):
            shape, method = fshape, "faceted-closed"
            stats["closed_fallback"] = True
        else:
            stats.setdefault("warnings_extra", []).append(
                "Fully-closed fallback could not produce a watertight solid.")

    # Hard bounding-box ceiling gate (P0-1): a tier can ship a watertight, valid-
    # on-reread solid whose dimensions are catastrophically wrong (a degenerate
    # boolean collapsing a plate to a cube). The per-op guards catch this at the
    # source, but this is the last-line net: if the adopted output's bbox differs
    # from the input mesh by more than the ceiling on any axis, REJECT that tier's
    # result and fall back to a dimensionally-faithful watertight faceted solid.
    if config.bbox_reject_delta is not None and _is_solid(shape):
        delta = _bbox_delta(shape, input_dims)
        if delta is not None and delta > config.bbox_reject_delta:
            progress(f"Bounding-box GATE: output differs from input by "
                     f"{delta * 100:.1f}% (> {config.bbox_reject_delta * 100:.0f}% "
                     f"ceiling) — REJECTING the '{method}' result as dimensionally "
                     f"destroyed; falling back to a watertight faceted solid")
            stats["bbox_gate_rejected"] = {
                "rejected_method": method,
                "delta_pct": round(delta * 100, 2),
                "ceiling_pct": round(config.bbox_reject_delta * 100, 2),
            }
            gverts, gfaces = load_stl(input_path, weld_tol=config.weld_tol)
            if scale != 1.0:
                gverts = gverts * scale
            gshape = builder.build_faceted_solid(gverts, gfaces)
            # A self-intersecting mesh (base_lid) won't form a valid faceted solid
            # directly; resolve overlapping bodies first (same path the boolean
            # tier uses) so the fallback actually produces a watertight solid.
            if not _is_solid(gshape):
                from . import meshprep as _mp

                if _mp.has_self_intersections(gverts, gfaces):
                    resolved = _mp.resolve_self_intersections(
                        gverts, gfaces, on_progress=progress)
                    if resolved is not None:
                        rv, rf, _ = resolved
                        gshape = builder.build_faceted_solid(rv, rf)
            if _is_solid(gshape):
                gdelta = _bbox_delta(gshape, input_dims)
                if gdelta is None or gdelta <= config.bbox_reject_delta:
                    shape, method = gshape, "faceted-closed"
                    clean_shape, dual, boolean_exported = None, False, False
                    stats["bbox_gate_fallback"] = "faceted-closed"
                else:
                    # The faithful fallback should always be dimensionally sane;
                    # if even it is off, keep it (it is at worst the raw mesh) but
                    # record that the gate could not recover a clean result.
                    shape, method = gshape, "faceted-closed"
                    clean_shape, dual, boolean_exported = None, False, False
                    stats["bbox_gate_fallback"] = "faceted-closed (still off-dimension)"
            else:
                stats["bbox_gate_fallback"] = "failed (no watertight faceted solid)"

    _assess_quality(shape, input_dims, method, stats, config, builder)

    # Narrate a material bounding-box distortion so a user watching live progress
    # (CLI/GUI) sees it — previously it only appeared in the returned stats dict,
    # so an "artifact-free — adopted" success could hide a 15% oversize.
    bdelta = stats.get("bbox_delta_pct")
    if isinstance(bdelta, (int, float)) and bdelta > 2.0:
        sev = "PROBLEM" if bdelta > 5.0 else "warning"
        progress(f"Bounding-box {sev}: output differs from input by {bdelta:.1f}% "
                 f"(input {stats.get('bbox_input_mm')} vs "
                 f"output {stats.get('bbox_output_mm')} mm)")

    if dual and clean_shape is not None:
        # Two deliverables, clearly named: watertight (may contain artifacts) and
        # clean (artifact-free, but open shells — heal on import).
        wt_path = _suffixed(output_path, "_watertight")
        clean_path = _suffixed(output_path, "_clean")
        # The back-off ladder may have written a probe to output_path; the dual
        # deliverables use suffixed names, so drop the stale unsuffixed file.
        if boolean_exported and output_path.exists() and output_path not in (wt_path, clean_path):
            try:
                output_path.unlink()
            except OSError:
                pass
        progress(f"Exporting {wt_path.name} (watertight, has artifacts)")
        builder.export_step(shape, wt_path)
        if config.revalidate_export:
            reval = _export_and_revalidate(builder, shape, wt_path, expected_solids=None,
                                           config=config, progress=progress, already_written=True)
            stats["export_revalidation"] = reval
            if not reval.get("valid", False):
                stats.setdefault("warnings_extra", []).append(
                    f"Exported {wt_path.name} re-reads invalid ({reval.get('reason')}).")
                _finalize_quality_after_export(stats, reval)
        progress(f"Exporting {clean_path.name} (artifact-free, open)")
        builder.export_step(clean_shape, clean_path)
        stats["dual_output"] = {"watertight": str(wt_path), "clean": str(clean_path)}
        _write_features_sidecar(output_path, stats, progress)
        progress("Done")
        return ConversionResult(output_path=wt_path, method=method, stats=stats,
                                outputs=[wt_path, clean_path])

    if boolean_exported and output_path.exists():
        # The winning boolean rung was already written to output_path and
        # re-validated during the back-off ladder — no need to re-export.
        _write_features_sidecar(output_path, stats, progress)
        progress("Done")
        return ConversionResult(output_path=output_path, method=method, stats=stats,
                                outputs=[output_path])

    progress("Exporting STEP")
    builder.export_step(shape, output_path)
    if config.revalidate_export:
        reval = _export_and_revalidate(builder, shape, output_path, expected_solids=None,
                                       config=config, progress=progress, already_written=True)
        stats["export_revalidation"] = reval
        if not reval.get("valid", False):
            stats.setdefault("warnings_extra", []).append(
                f"Exported STEP re-reads invalid ({reval.get('reason')}).")
            _finalize_quality_after_export(stats, reval)
    _write_features_sidecar(output_path, stats, progress)
    progress("Done")
    return ConversionResult(output_path=output_path, method=method, stats=stats,
                            outputs=[output_path])


def _combine_bodies_to_stl(raw_v, raw_f, input_path, config, progress):
    """Union a multi-body mesh into one solid and write it to a temp STL.

    Returns ``(Path, report)`` on success (the caller runs the ordinary
    single-body pipeline on the fused STL), or ``None`` if the union is
    unavailable/failed (the caller falls back to the separate per-body path).
    The union runs on the raw welded mesh in source units — the single-body
    ``convert`` call scales it to mm afterwards, exactly as it would the original.
    """
    import tempfile

    from . import meshprep
    from .mesh_io import write_binary_stl

    progress("Multi-body combine: unioning bodies into one solid (manifold3d)")
    combined = meshprep.combine_bodies(
        raw_v, raw_f, weld=config.multibody_combine_weld, on_progress=progress)
    if combined is None:
        return None
    cv, cf, report = combined
    progress(f"  combined {report['bodies_in']} bodies "
             f"({report['faces_in']:,} -> {report['faces_out']:,} facets)")
    tmp = Path(tempfile.mkdtemp(prefix="mesh2step_combine_"))
    combined_stl = tmp / (Path(input_path).stem + "_combined.stl")
    write_binary_stl(cv, cf, combined_stl)
    return combined_stl, report


def _convert_multibody(components, input_path, output_path, config, builder,
                       progress) -> ConversionResult:
    """Convert a multi-body mesh: one STEP compound of N independent solids.

    Each disjoint body is written to a temp STL and run through the *ordinary*
    single-body ``convert`` (with multi-body dispatch disabled so it can't
    re-split), then the resulting solids are compounded into one STEP. The
    per-body stats are collected under ``stats["bodies"]`` and aggregated:
    ``stats["solids"]`` is N, ``stats["watertight"]`` / ``stats["is_solid"]`` are
    True only if EVERY body is a watertight solid, and ``stats["quality"]`` is the
    worst body verdict. Watertightness is thus required per body.
    """
    import dataclasses
    import tempfile

    import Part  # type: ignore

    from .mesh_io import write_binary_stl

    n = len(components)
    progress(f"Multi-body mesh: {n} disjoint bodies; converting each independently")
    body_cfg = dataclasses.replace(config, multi_body=False)

    solids: list = []
    body_stats: list[dict] = []
    outputs: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="mesh2step_bodies_") as tmp:
        tmpd = Path(tmp)
        for i, (bverts, bfaces) in enumerate(components):
            progress(f"Body {i + 1}/{n}: {len(bfaces):,} facets")
            body_stl = tmpd / f"body_{i}.stl"
            body_step = tmpd / f"body_{i}.step"
            write_binary_stl(bverts, bfaces, body_stl)

            def _body_progress(msg: str, _i=i) -> None:
                progress(f"  [body {_i + 1}] {msg}")

            res = convert(body_stl, body_step, body_cfg, on_progress=_body_progress)
            body_stats.append({"body": i, "method": res.method,
                               "is_solid": res.stats.get("is_solid"),
                               "watertight": res.stats.get("watertight"),
                               "quality": res.stats.get("quality"),
                               "rtaf": res.stats.get("rtaf"),
                               "bbox_delta_pct": res.stats.get("bbox_delta_pct")})
            # Read the body's solids back and collect them for the compound.
            for op in (res.outputs or [res.output_path]):
                # Prefer the primary (watertight) output of a dual-output body.
                if "_clean" in Path(op).name:
                    continue
                shp = Part.Shape()
                shp.read(str(op))
                solids.extend(getattr(shp, "Solids", []) or [shp])
                break

        compound = Part.makeCompound(solids)
        builder.export_step(compound, output_path)
        outputs.append(output_path)

    # Aggregate stats across bodies.
    all_solid = all(bool(b.get("is_solid")) for b in body_stats) and len(body_stats) == n
    order = {"problems": 0, "warnings": 1, "good": 2}
    worst = min((b.get("quality") or "problems" for b in body_stats),
                key=lambda q: order.get(q, 0))
    rtafs = [b["rtaf"] for b in body_stats if isinstance(b.get("rtaf"), (int, float))]
    deltas = [b["bbox_delta_pct"] for b in body_stats
              if isinstance(b.get("bbox_delta_pct"), (int, float))]
    stats: dict = {
        "solids": n,
        "bodies": body_stats,
        "is_solid": all_solid,
        "watertight": all_solid,
        "quality": worst,
        "multi_body": True,
        "multibody_mode": "separate",
    }
    if rtafs:
        stats["rtaf"] = round(max(rtafs), 4)  # worst (most-faceted) body
    if deltas:
        stats["bbox_delta_pct"] = max(deltas)
    warnings: list[str] = []
    if not all_solid:
        warnings.append(
            f"{sum(1 for b in body_stats if not b.get('is_solid'))} of {n} bodies "
            f"did not convert to a watertight solid.")
    stats["warnings"] = warnings

    progress(f"Multi-body: exported {n}-solid compound "
             f"({'all watertight' if all_solid else 'some bodies open'})")
    if config.revalidate_export:
        reval = builder.revalidate_step(output_path, expected_solids=n)
        stats["export_revalidation"] = reval
        if not reval.get("valid", False):
            warnings.append(f"Exported compound re-reads invalid ({reval.get('reason')}).")
            if all_solid:
                stats["quality"] = "problems"
    progress("Done")
    return ConversionResult(output_path=output_path, method="multi-body",
                            stats=stats, outputs=outputs)


def _is_solid(shape) -> bool:
    """True if the shape is a single valid (watertight) solid."""
    solids = getattr(shape, "Solids", [])
    return bool(solids) and solids[0].isValid()


def _bbox_delta(shape, input_dims) -> float | None:
    """Max relative bbox difference (any axis) between ``shape`` and the input.

    ``input_dims`` is the sorted (desc) input mesh side lengths. Returns the
    largest per-axis relative delta, or ``None`` if the box can't be read. Shared
    by the hard bbox-ceiling gate and mirrors the metric ``_assess_quality``
    reports, so the gate fires on exactly the number the user sees.
    """
    try:
        bb = shape.BoundBox
        out_dims = sorted([bb.XLength, bb.YLength, bb.ZLength], reverse=True)
        return max((abs(o - i) / i) for o, i in zip(out_dims, input_dims) if i > 1e-9)
    except Exception:  # noqa: BLE001
        return None


def _bbox_stable(before, after, config: ConversionConfig, rel: float = 0.03) -> bool:
    """True if ``after``'s bounding box matches ``before``'s within ``rel``.

    Guards the organic multi-patch adoption: a quad-remesh + limit surface must
    reproduce the part's silhouette, not shrink or bloat it. Uses the config's
    bbox growth guard when tighter."""
    try:
        b, a = before.BoundBox, after.BoundBox
        bd = sorted((b.XLength, b.YLength, b.ZLength), reverse=True)
        ad = sorted((a.XLength, a.YLength, a.ZLength), reverse=True)
        tol = rel
        if config.boolean_max_bbox_growth is not None:
            tol = max(tol, config.boolean_max_bbox_growth)
        return all(abs(x - y) <= tol * max(x, 1e-9) for x, y in zip(bd, ad))
    except Exception:  # noqa: BLE001
        return False


def _finalize_quality_after_export(stats: dict, reval: dict) -> None:
    """Downgrade the quality verdict when the final export re-reads invalid.

    ``_assess_quality`` runs before the final single-file export is written, so
    a re-read failure of that file isn't reflected in its verdict. This appends
    the warning (if not already present) and forces ``quality`` to "problems".
    """
    msg = f"Exported STEP re-reads invalid ({reval.get('reason')})."
    warnings = stats.setdefault("warnings", [])
    if msg not in warnings:
        warnings.append(msg)
    stats["quality"] = "problems"


def _export_and_revalidate(builder, shape, path, *, expected_solids, config, progress,
                           already_written: bool = False) -> dict:
    """Write ``shape`` to ``path`` (unless already written) and re-read to verify.

    Returns the re-validation dict from ``builder.revalidate_step`` (``valid``
    plus, on failure, ``reason``). Honours ``revalidate_export_max_bytes``: if
    the written file exceeds it, the re-read is skipped and ``valid`` is reported
    optimistically (with ``skipped="file too large"``) rather than paying a slow
    read on an enormous STEP.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    if not already_written:
        builder.export_step(shape, path)
    cap = config.revalidate_export_max_bytes
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if cap is not None and size > cap:
        progress(f"  export re-validation skipped ({size:,} bytes > {cap:,})")
        return {"valid": True, "skipped": "file too large", "bytes": int(size),
                "path": str(path)}
    return builder.revalidate_step(path, expected_solids=expected_solids)


def _assess_quality(shape, input_dims, method: str, stats: dict,
                    config: ConversionConfig | None = None, builder=None) -> None:
    """Populate ``stats`` with a quality verdict + human-readable warnings.

    Gives the user an at-a-glance sense of result quality: watertightness,
    faceted fallbacks, features that failed to build, and any bounding-box
    mismatch between the input mesh and the exported solid.
    """
    warnings: list[str] = []

    # Residual Tessellation Area Fraction: how much of the output surface area
    # still looks faceted (near-tangent planar-fan chains). Computed on the final
    # shape where Part is available. Surfaced as a percentage; a materially
    # faceted result (>= 5% of surface area) earns a warning line so the user
    # knows the geometry reads faceted even when it is a valid watertight solid.
    if config is not None and builder is not None:
        try:
            rtaf_info = builder.compute_rtaf(shape, config)
        except Exception as exc:  # noqa: BLE001 - metric must not break quality
            rtaf_info = {"rtaf": None, "skipped": f"error ({exc})"}
        stats["rtaf"] = rtaf_info.get("rtaf")
        stats["rtaf_detail"] = rtaf_info
        rtaf = rtaf_info.get("rtaf")
        if rtaf is not None and rtaf >= 0.05:
            warnings.append(
                f"Residual tessellation: {rtaf * 100:.0f}% of the output surface "
                f"area is faceted (near-tangent planar strips on curved features).")

    # Watertight solid? "watertight" mirrors is_solid semantics (a single valid
    # closed solid) and is populated as a stable, explicitly-named field so
    # consumers reading stats["watertight"] get a real bool instead of None.
    solids = getattr(shape, "Solids", [])
    is_solid = bool(solids) and solids[0].isValid()
    stats["is_solid"] = is_solid
    stats["watertight"] = is_solid
    if not is_solid:
        warnings.append("Result is not a single watertight solid (open shells present).")

    if method == "faceted":
        warnings.append("Surface reconstruction was skipped or failed; faceted solid produced.")
    if method == "faceted-closed":
        warnings.append("Fully-closed fallback: watertight faceted solid (holes not analytic).")
    if stats.get("reconstruction_error"):
        warnings.append(f"Reconstruction fell back to faceted: {stats['reconstruction_error']}")

    skipped = stats.get("skipped_facets", 0)
    if skipped:
        warnings.append(f"{skipped:,} facets could not be reconstructed and were left out/faceted.")

    # Export round-trip re-validation: surface rungs that produced an in-memory
    # valid solid but re-read invalid from the exported STEP (sliver-triangle
    # self-intersecting wires), and any final export that still re-reads invalid.
    failed_rungs = stats.get("export_revalidation_failed_rungs")
    if failed_rungs:
        rung_names = ", ".join(r.get("rung", "?") for r in failed_rungs)
        won = stats.get("export_revalidation_winning_rung")
        if won:
            warnings.append(
                f"Export re-read invalid at decimation rung(s) {rung_names}; "
                f"backed off and exported a valid solid at rung {won}.")
        else:
            warnings.append(
                f"Export re-read invalid at decimation rung(s) {rung_names}.")
    if stats.get("export_revalidated") is False:
        warnings.append(
            "Even the undecimated boolean base re-reads invalid from the "
            "exported STEP; shipped it as the best watertight result available.")

    # Planarity-damage back-off: surface rungs whose decimation warped the flats
    # (shattering large planar faces into micro-regions) and were backed off to a
    # gentler rung that preserved the flats.
    damaged = stats.get("planarity_damaged_rungs")
    if damaged:
        rung_names = ", ".join(r.get("rung", "?") for r in damaged)
        won = stats.get("planarity_winning_rung")
        if won:
            warnings.append(
                f"Decimation warped flats at rung(s) {rung_names}; backed off to "
                f"rung {won} to keep large planar faces (avoids a faceted result).")
        else:
            warnings.append(
                f"Decimation warped flats at rung(s) {rung_names}; used the "
                f"undecimated base to preserve flats.")
    final_reval = stats.get("export_revalidation")
    if isinstance(final_reval, dict) and final_reval.get("valid") is False:
        warnings.append(
            f"Exported STEP re-reads invalid ({final_reval.get('reason')}).")

    detected = stats.get("cylinders_detected")
    built = stats.get("cylinder_faces")
    if detected is not None and built is not None and built < detected:
        warnings.append(f"{detected - built} detected cylinder(s) could not be built as analytic faces.")

    # Bounding-box sanity: exported solid vs input mesh. A material distortion
    # (a part that shipped visibly over- or under-sized in some axis) is a
    # correctness problem, not a cosmetic one — a boolean fuse-back that extended
    # geometry past the original silhouette can grow the box 15-30%. Surface it
    # loudly: >2% is a warning line (and narrated via on_progress by the caller),
    # >5% forces the quality verdict down to "problems" (set below).
    bbox_delta = None
    try:
        bb = shape.BoundBox
        out_dims = sorted([bb.XLength, bb.YLength, bb.ZLength], reverse=True)
        stats["bbox_input_mm"] = [round(x, 3) for x in input_dims]
        stats["bbox_output_mm"] = [round(x, 3) for x in out_dims]
        rel = max((abs(o - i) / i) for o, i in zip(out_dims, input_dims) if i > 1e-9)
        bbox_delta = rel
        stats["bbox_delta_pct"] = round(rel * 100, 2)
        if rel > 0.05:
            warnings.append(
                f"Output bounding box differs from input by {rel * 100:.1f}% — the "
                f"exported solid is materially off-dimension (input "
                f"{stats['bbox_input_mm']} mm vs output {stats['bbox_output_mm']} mm).")
        elif rel > 0.02:
            warnings.append(f"Output bounding box differs from input by {rel * 100:.1f}%.")
    except Exception:  # noqa: BLE001
        pass

    # A final export that re-reads invalid, or an undecimated base that never
    # round-trips cleanly, is a hard problem — the shipped file is not a valid
    # solid on disk even though the in-memory shape passed isValid().
    final_reval = stats.get("export_revalidation")
    export_invalid = (isinstance(final_reval, dict) and final_reval.get("valid") is False) \
        or stats.get("export_revalidated") is False

    # A bounding box more than 5% off input is a correctness problem (the part
    # shipped materially wrong-sized), so it forces "problems" the same way a
    # non-solid or invalid export does — not the softer "warnings".
    bbox_problem = bbox_delta is not None and bbox_delta > 0.05

    stats["warnings"] = warnings
    if method == "faceted" or not is_solid or export_invalid or bbox_problem:
        stats["quality"] = "problems"
    elif warnings:
        stats["quality"] = "warnings"
    else:
        stats["quality"] = "good"
