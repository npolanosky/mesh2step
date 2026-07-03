"""mesh2step-web — a local web UI for STL -> STEP conversion.

A FastAPI app that drives the same out-of-process conversion worker the desktop
GUI uses (FreeCAD's bundled Python, subprocess + streamed progress). The web
process itself NEVER imports FreeCAD or numpy-heavy CAD code — everything heavy
runs in the worker subprocess exactly like ``gui.py`` does.

Run it with the ``mesh2step-web`` console script (see :mod:`mesh2step.webapp.server`)
or ``python -m mesh2step.webapp``.
"""

from __future__ import annotations

__all__ = ["create_app", "WebConfig", "main"]


def __getattr__(name: str):
    # Lazy so ``import mesh2step.webapp`` stays cheap and doesn't pull FastAPI in
    # unless the app is actually built.
    if name in ("create_app",):
        from .app import create_app

        return create_app
    if name == "WebConfig":
        from .config import WebConfig

        return WebConfig
    if name == "main":
        from .server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
