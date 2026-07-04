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
        "--repair", action="store_true",
        help="repair the mesh first (fix self-intersections, duplicates, normals)",
    )
    p.add_argument(
        "--multibody", choices=list(ConversionConfig.MULTIBODY_MODES), default="auto",
        metavar="MODE",
        help="how to handle a mesh with several disjoint bodies: 'separate' "
             "(one STEP compound of N solids), 'combine' (union all bodies into "
             "one solid), or 'auto' (default: combine only bodies that share a "
             "coincident seam, else keep separate)",
    )
    p.add_argument(
        "--closed", action="store_true",
        help="guarantee a watertight solid (faceted fallback if reconstruction can't close)",
    )
    p.add_argument(
        "--faceted", action="store_true",
        help="skip reconstruction; emit the classic faceted solid",
    )
    p.add_argument(
        "--freecad-bin", type=str, default=None, metavar="PATH",
        help="path to FreeCAD's bin/ (Windows) or lib/ (macOS/Linux)",
    )
    p.add_argument(
        "--save-failures", nargs="?", const="", default=None, metavar="DIR",
        help="copy inputs that fail to convert to a single watertight solid into "
             "a regression corpus, sorted by failure category (default DIR: "
             "tests/data/community/failures in a source checkout, else the "
             "per-user support dir); a later pass on a saved file is recorded "
             "in the corpus manifest",
    )
    return p


def _split_argv(argv: list[str]) -> list[str]:
    """Drop a leading ``--`` separator used when launched via freecadcmd."""
    return argv[1:] if argv and argv[0] == "--" else argv


def _ensure_prep_on_path() -> None:
    """Inject the auto-provisioned prep-deps dir onto ``sys.path`` (headless).

    If the CLI is running under FreeCAD's Python and pymeshlab/manifold3d aren't
    importable, provision them once (into the per-user pydeps dir) and add that
    dir to ``sys.path``. Silent + best-effort: any failure just leaves the deps
    unavailable and the pipeline degrades gracefully.
    """
    try:
        from . import provision
    except Exception:  # noqa: BLE001
        return
    # We're presumably running under FreeCAD's own interpreter here.
    fc_python = sys.executable
    target = provision.pydeps_dir(fc_python)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    # Only probe manifold3d here (the required dep). We must NOT `import
    # pymeshlab` in-process: on some macOS builds that hard-crashes the
    # interpreter (SIGTRAP), which no try/except can catch. meshprep probes
    # pymeshlab crash-safely in a subprocess when it's actually needed.
    try:
        import manifold3d  # type: ignore  # noqa: F401

        return  # already importable — nothing to provision
    except Exception:  # noqa: BLE001
        pass
    try:
        got = provision.ensure_prep_deps(fc_python, log=lambda m: print(m, file=sys.stderr))
        if got and str(got) not in sys.path:
            sys.path.insert(0, str(got))
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(_split_argv(raw))

    config = ConversionConfig(
        weld_tol=args.weld_tol,
        angle_tol_deg=args.angle_tol,
        dist_tol=args.dist_tol,
        source_units=args.units,
        detect_cylinders=not args.no_cylinders,
        repair_mesh=args.repair,
        multibody_mode=args.multibody,
        full_closed=args.closed,
        faceted=args.faceted,
        freecad_bin=args.freecad_bin,
    )

    # Headless self-provisioning: when running under FreeCAD's own Python (the
    # usual way the CLI is invoked), make the auto-installed prep deps
    # (pymeshlab/manifold3d) importable and provision them on first use, so the
    # bundled `mesh2step` works with no manual pip install. Best-effort.
    _ensure_prep_on_path()

    # Imported here so `mesh2step --help` works without FreeCAD present.
    from .pipeline import convert

    def _record_outcome(outcome: dict) -> None:
        """Book-keep the failure corpus when --save-failures is on. Best-effort."""
        if args.save_failures is None:
            return
        from . import failstore

        failstore.record_result(args.input, outcome,
                                dest=args.save_failures or None,
                                log=lambda m: print(m, file=sys.stderr))

    try:
        result = convert(args.input, args.output, config)
    except FileNotFoundError:
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    except ImportError as exc:
        # Environment problem (FreeCAD missing), not a property of the mesh —
        # don't pollute the failure corpus with it.
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        _record_outcome({"ok": False, "error": str(exc)})
        print(f"error: conversion failed: {exc}", file=sys.stderr)
        return 1
    _record_outcome({"ok": True, "stats": result.stats})

    s = result.stats
    print(f"wrote {result.output_path}  [{result.method}]")
    if s.get("solids"):
        bodies = s.get("bodies") or []
        ok = sum(1 for b in bodies if b.get("is_solid"))
        print(f"  multi-body: {s['solids']} solids ({ok} watertight, "
              f"all_watertight={s.get('is_solid')})")
    if "faces_out" in s:
        print(
            f"  faces: {s['faces_in']} triangles -> {s['faces_out']} STEP faces"
            f"  (solid={s.get('is_solid')})"
        )
    if s.get("skipped_facets"):
        print(f"  skipped facets: {s['skipped_facets']:,}")
    if s.get("rtaf") is not None:
        print(f"  residual tessellation (RTAF): {s['rtaf'] * 100:.0f}% of surface area")
    if "reconstruction_error" in s:
        print(f"  note: fell back to faceted ({s['reconstruction_error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
