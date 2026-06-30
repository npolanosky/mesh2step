"""mesh2step — STL mesh to STEP solid conversion with surface reconstruction.

Public surface:
    ConversionConfig   tolerances and flags
    convert            run the full pipeline (requires FreeCAD at runtime)
    load_stl           parse + weld an STL into numpy arrays (no FreeCAD)
    segment_planar     planar region growing over a welded mesh (no FreeCAD)
"""

from .config import ConversionConfig
from .mesh_io import load_stl
from .segmentation import segment_planar

__all__ = ["ConversionConfig", "load_stl", "segment_planar", "convert"]

__version__ = "0.1.0"


def convert(*args, **kwargs):
    """Lazy wrapper so importing the package never pulls in FreeCAD.

    The pipeline imports the FreeCAD-backed builder; we defer that import to
    call time so that ``import mesh2step`` works under any interpreter.
    """
    from .pipeline import convert as _convert

    return _convert(*args, **kwargs)
