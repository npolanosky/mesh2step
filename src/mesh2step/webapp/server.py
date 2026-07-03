"""``mesh2step-web`` console entry point: build the app and serve it with uvicorn."""

from __future__ import annotations

import argparse
import sys

from .config import WebConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mesh2step-web",
        description="Local web UI for STL -> STEP conversion (FreeCAD-backed).")
    p.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port (default 8765)")
    p.add_argument("--data-dir", default=None,
                   help="working dir for jobs (default ~/.mesh2step-web)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="conversions to run at once (default 1)")
    p.add_argument("--freecad-python", default=None,
                   help="path to FreeCAD's bundled python (default: auto-detect)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = WebConfig()
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.data_dir is not None:
        from pathlib import Path

        cfg.data_dir = Path(args.data_dir).expanduser()
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency
    if args.freecad_python is not None:
        cfg.freecad_python = args.freecad_python

    import uvicorn

    from .app import create_app

    app = create_app(cfg)
    fc = cfg.freecad_python or "NOT FOUND"
    print(f"mesh2step-web  →  http://{cfg.host}:{cfg.port}")
    print(f"  data dir : {cfg.data_dir}")
    print(f"  FreeCAD  : {fc}")
    if not cfg.freecad_python:
        print("  ⚠ FreeCAD not found — conversions will fail until it is installed "
              "(see docs/WEBAPP.md).")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
