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

    # Keep the clean (possibly open) reconstruction so we can also emit it when
    # the watertight version can only be produced with artifacts.
    clean_shape = shape if method == "reconstructed" else None
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
            resolved = meshprep.resolve_self_intersections(bverts, bfaces)
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

    _assess_quality(shape, input_dims, method, stats, config, builder)

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
        progress("Done")
        return ConversionResult(output_path=wt_path, method=method, stats=stats,
                                outputs=[wt_path, clean_path])

    if boolean_exported and output_path.exists():
        # The winning boolean rung was already written to output_path and
        # re-validated during the back-off ladder — no need to re-export.
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
    progress("Done")
    return ConversionResult(output_path=output_path, method=method, stats=stats,
                            outputs=[output_path])


def _is_solid(shape) -> bool:
    """True if the shape is a single valid (watertight) solid."""
    solids = getattr(shape, "Solids", [])
    return bool(solids) and solids[0].isValid()


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

    # Watertight solid?
    solids = getattr(shape, "Solids", [])
    is_solid = bool(solids) and solids[0].isValid()
    stats["is_solid"] = is_solid
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
    final_reval = stats.get("export_revalidation")
    if isinstance(final_reval, dict) and final_reval.get("valid") is False:
        warnings.append(
            f"Exported STEP re-reads invalid ({final_reval.get('reason')}).")

    detected = stats.get("cylinders_detected")
    built = stats.get("cylinder_faces")
    if detected is not None and built is not None and built < detected:
        warnings.append(f"{detected - built} detected cylinder(s) could not be built as analytic faces.")

    # Bounding-box sanity: exported solid vs input mesh.
    try:
        bb = shape.BoundBox
        out_dims = sorted([bb.XLength, bb.YLength, bb.ZLength], reverse=True)
        stats["bbox_input_mm"] = [round(x, 3) for x in input_dims]
        stats["bbox_output_mm"] = [round(x, 3) for x in out_dims]
        rel = max((abs(o - i) / i) for o, i in zip(out_dims, input_dims) if i > 1e-9)
        stats["bbox_delta_pct"] = round(rel * 100, 2)
        if rel > 0.01:
            warnings.append(f"Output bounding box differs from input by {rel * 100:.1f}%.")
    except Exception:  # noqa: BLE001
        pass

    # A final export that re-reads invalid, or an undecimated base that never
    # round-trips cleanly, is a hard problem — the shipped file is not a valid
    # solid on disk even though the in-memory shape passed isValid().
    final_reval = stats.get("export_revalidation")
    export_invalid = (isinstance(final_reval, dict) and final_reval.get("valid") is False) \
        or stats.get("export_revalidated") is False

    stats["warnings"] = warnings
    if method == "faceted" or not is_solid or export_invalid:
        stats["quality"] = "problems"
    elif warnings:
        stats["quality"] = "warnings"
    else:
        stats["quality"] = "good"
