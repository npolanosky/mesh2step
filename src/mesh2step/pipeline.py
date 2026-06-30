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
) -> ConversionResult:
    """Convert an STL file to a STEP solid.

    Tries surface reconstruction first; falls back to a faceted solid if
    reconstruction can't produce geometry. Requires FreeCAD at runtime.
    """
    config = config or ConversionConfig()
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".step")
    output_path = Path(output_path)

    # Make FreeCAD importable, then pull in the builder (which needs it).
    from .freecad_env import ensure_freecad

    ensure_freecad(config.freecad_bin)
    from . import builder

    # Weld in raw units, then scale to millimetres (STEP is always mm). Welding
    # first keeps weld_tol meaningful regardless of the source unit scale.
    vertices, faces = load_stl(input_path, weld_tol=config.weld_tol)
    if faces.size == 0:
        raise ValueError(f"{input_path} contained no usable triangles")
    scale = config.scale_to_mm
    if scale != 1.0:
        vertices = vertices * scale

    method = "faceted"
    stats: dict = {"faces_in": int(len(faces))}

    if not config.faceted:
        try:
            shape, stats = builder.build_reconstructed_solid(vertices, faces, config)
            method = "reconstructed"
        except Exception as exc:  # noqa: BLE001 - reconstruction is best-effort
            stats["reconstruction_error"] = str(exc)
            shape = builder.build_faceted_solid(vertices, faces)
    else:
        shape = builder.build_faceted_solid(vertices, faces)

    builder.export_step(shape, output_path)
    return ConversionResult(output_path=output_path, method=method, stats=stats)
