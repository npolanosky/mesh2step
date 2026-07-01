"""mesh2step — STL mesh to STEP solid conversion with surface reconstruction.

Public surface (lazily imported so ``import mesh2step`` never pulls in numpy or
FreeCAD — the GUI imports this package under a plain Python that has neither):

    ConversionConfig   tolerances and flags
    convert            run the full pipeline (requires FreeCAD at runtime)
    load_stl           parse + weld an STL into numpy arrays (no FreeCAD)
    segment_planar     planar region growing over a welded mesh (no FreeCAD)
"""

from __future__ import annotations

__all__ = ["ConversionConfig", "load_stl", "segment_planar", "convert"]

__version__ = "0.2.0a3"          # PEP 440
DISPLAY_VERSION = "v0.2.0-alpha.3"  # shown in the GUI

# Map public name -> (submodule, attribute). Imports happen on first access so
# that e.g. ConversionConfig (pure stdlib) is usable without importing numpy.
_LAZY = {
    "ConversionConfig": ("config", "ConversionConfig"),
    "load_stl": ("mesh_io", "load_stl"),
    "segment_planar": ("segmentation", "segment_planar"),
    "convert": ("pipeline", "convert"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{target[0]}", __name__)
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
