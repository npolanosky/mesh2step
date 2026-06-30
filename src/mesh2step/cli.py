"""Command-line interface for mesh2step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConversionConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mesh2step",
        description="Convert an STL mesh to a STEP solid with surface reconstruction.",
    )
    p.add_argument("input", type=Path, help="input STL file")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output STEP file (default: input name with .step)",
    )
    p.add_argument(
        "--angle-tol", type=float, default=1.0, metavar="DEG",
        help="max normal deviation (deg) for coplanar facets (default: 1.0)",
    )
    p.add_argument(
        "--dist-tol", type=float, default=1e-2, metavar="MM",
        help="max point-to-plane distance within a region (default: 0.01)",
    )
    p.add_argument(
        "--weld-tol", type=float, default=1e-5, metavar="MM",
        help="coincident-vertex welding tolerance (default: 1e-5)",
    )
    p.add_argument(
        "--units", choices=["mm", "cm", "m", "in"], default="mm",
        help="source units of the STL; scaled to mm for STEP (default: mm)",
    )
    p.add_argument(
        "--no-cylinders", action="store_true",
        help="disable cylindrical hole/boss detection",
    )
    p.add_argument(
        "--faceted", action="store_true",
        help="skip reconstruction; emit the classic faceted solid",
    )
    p.add_argument(
        "--freecad-bin", type=str, default=None, metavar="PATH",
        help="path to FreeCAD's bin/ (Windows) or lib/ (macOS/Linux)",
    )
    return p


def _split_argv(argv: list[str]) -> list[str]:
    """Drop a leading ``--`` separator used when launched via freecadcmd."""
    return argv[1:] if argv and argv[0] == "--" else argv


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(_split_argv(raw))

    config = ConversionConfig(
        weld_tol=args.weld_tol,
        angle_tol_deg=args.angle_tol,
        dist_tol=args.dist_tol,
        source_units=args.units,
        detect_cylinders=not args.no_cylinders,
        faceted=args.faceted,
        freecad_bin=args.freecad_bin,
    )

    # Imported here so `mesh2step --help` works without FreeCAD present.
    from .pipeline import convert

    try:
        result = convert(args.input, args.output, config)
    except FileNotFoundError:
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"error: conversion failed: {exc}", file=sys.stderr)
        return 1

    s = result.stats
    print(f"wrote {result.output_path}  [{result.method}]")
    if "faces_out" in s:
        print(
            f"  faces: {s['faces_in']} triangles -> {s['faces_out']} STEP faces"
            f"  (solid={s.get('is_solid')})"
        )
    if "reconstruction_error" in s:
        print(f"  note: fell back to faceted ({s['reconstruction_error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
