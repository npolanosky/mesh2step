"""End-to-end orchestration: STL path in, STEP path out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ConversionConfig
from .mesh_io import load_stl


@dataclass
class ConversionResult:
    output_path: Path
    method: str  # "reconstructed" or "faceted"
    stats: dict


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
    if config.repair_mesh or config.decimate:
        progress("Preparing mesh (repair/decimate)")
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

    input_dims = sorted((vertices.max(axis=0) - vertices.min(axis=0)).tolist(), reverse=True)

    method = "faceted"
    stats: dict = {"faces_in": int(len(faces))}
    if prep_report is not None:
        stats["mesh_prep"] = prep_report

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

    # Fully-closed toggle: guarantee a watertight solid.
    if config.full_closed and not _is_solid(shape):
        # Tier 2: boolean clean-up. Start from the (guaranteed watertight)
        # faceted solid and cut+fuse each detected hole/boss/cone to its exact
        # analytic geometry. Robust where sewing failed — booleans recompute the
        # intersection rather than needing analytic and mesh faces to already
        # share topology — and gives real round holes, not faceted ones.
        progress("Fully-closed: boolean clean-up (cut + fuse-back analytic holes)")
        # Use the ORIGINAL welded mesh — repair can leave it non-watertight,
        # which breaks the base faceted solid the booleans build on.
        bverts, bfaces = load_stl(input_path, weld_tol=config.weld_tol)
        if scale != 1.0:
            bverts = bverts * scale
        try:
            bshape, built2 = builder.build_boolean_clean_solid(
                bverts, bfaces, config, on_progress=progress
            )
            if _is_solid(bshape):
                shape, method = bshape, "boolean-clean"
                stats.update(built2)
            else:
                stats.setdefault("warnings_extra", []).append(
                    "Boolean clean-up did not produce a watertight solid.")
        except Exception as exc:  # noqa: BLE001
            progress(f"Boolean clean-up failed ({exc})")
            stats.setdefault("warnings_extra", []).append(f"Boolean clean-up failed: {exc}")

    if config.full_closed and not _is_solid(shape):
        # Tier 3: plain watertight faceted solid — no analytic holes, but
        # guaranteed closed. Last resort.
        progress("Fully-closed: building watertight faceted solid (slow on large meshes)")
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

    _assess_quality(shape, input_dims, method, stats)

    progress("Exporting STEP")
    builder.export_step(shape, output_path)
    progress("Done")
    return ConversionResult(output_path=output_path, method=method, stats=stats)


def _is_solid(shape) -> bool:
    """True if the shape is a single valid (watertight) solid."""
    solids = getattr(shape, "Solids", [])
    return bool(solids) and solids[0].isValid()


def _assess_quality(shape, input_dims, method: str, stats: dict) -> None:
    """Populate ``stats`` with a quality verdict + human-readable warnings.

    Gives the user an at-a-glance sense of result quality: watertightness,
    faceted fallbacks, features that failed to build, and any bounding-box
    mismatch between the input mesh and the exported solid.
    """
    warnings: list[str] = []

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

    stats["warnings"] = warnings
    if method == "faceted" or not is_solid:
        stats["quality"] = "problems"
    elif warnings:
        stats["quality"] = "warnings"
    else:
        stats["quality"] = "good"
