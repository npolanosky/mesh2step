"""Generate ground-truth sample parts and tessellate them to STL.

Run under FreeCAD's Python:

    "C:\\Program Files\\FreeCAD 1.1\\bin\\python.exe" scripts/generate_samples.py

Produces parts with *known* dimensions and hole radii so cylinder detection can
be validated against ground truth. A sidecar ``samples.json`` records the truth.
"""

import json
import sys
from pathlib import Path

import FreeCAD as App  # noqa: F401  (initialises Part/Mesh module search path)
import Mesh
import Part

OUT = Path(__file__).resolve().parent.parent / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Linear deflection (mm) for tessellation — smaller => more triangles, rounder
# holes. 0.05 mm is a realistic 3D-print/export setting.
DEFLECTION = 0.05


def save(shape, name, truth):
    """Tessellate a Part shape to STL and record ground-truth metadata."""
    mesh = Mesh.Mesh()
    pts, facets = shape.tessellate(DEFLECTION)
    mesh.addFacets([(pts[a], pts[b], pts[c]) for a, b, c in facets])
    path = OUT / f"{name}.stl"
    mesh.write(str(path))
    truth["file"] = path.name
    truth["triangles"] = mesh.CountFacets
    print(f"  {path.name}: {mesh.CountFacets} triangles")
    return truth


def cube():
    box = Part.makeBox(10, 10, 10)
    return save(box, "cube", {"kind": "box", "dims_mm": [10, 10, 10], "cylinders": []})


def plate_with_holes():
    # 60 x 40 x 10 plate with two through holes (r=5 and r=3).
    plate = Part.makeBox(60, 40, 10)
    h1 = Part.makeCylinder(5, 10, App.Vector(20, 20, 0))
    h2 = Part.makeCylinder(3, 10, App.Vector(45, 20, 0))
    part = plate.cut(h1).cut(h2)
    return save(
        part,
        "plate_with_holes",
        {
            "kind": "plate_with_holes",
            "dims_mm": [60, 40, 10],
            "cylinders": [
                {"radius": 5.0, "axis": [0, 0, 1], "through": True},
                {"radius": 3.0, "axis": [0, 0, 1], "through": True},
            ],
        },
    )


def cylinder():
    cyl = Part.makeCylinder(8, 20)
    return save(
        cyl,
        "cylinder",
        {"kind": "cylinder", "dims_mm": [16, 16, 20],
         "cylinders": [{"radius": 8.0, "axis": [0, 0, 1], "through": False}]},
    )


def l_bracket():
    # L-profile extruded; no curved faces (planar reconstruction stress test).
    pts = [App.Vector(*p) for p in [(0, 0, 0), (40, 0, 0), (40, 10, 0),
                                    (10, 10, 0), (10, 30, 0), (0, 30, 0), (0, 0, 0)]]
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    solid = face.extrude(App.Vector(0, 0, 15))
    return save(solid, "l_bracket",
                {"kind": "l_bracket", "dims_mm": [40, 30, 15], "cylinders": []})


def flanged_pipe():
    # A boss with a central bore: tests a cylinder wall on the outside AND a
    # smaller bore on the inside.
    boss = Part.makeCylinder(15, 25)
    bore = Part.makeCylinder(9, 25)
    part = boss.cut(bore)
    return save(
        part,
        "flanged_pipe",
        {"kind": "flanged_pipe", "dims_mm": [30, 30, 25],
         "cylinders": [
             {"radius": 15.0, "axis": [0, 0, 1], "through": False, "role": "outer"},
             {"radius": 9.0, "axis": [0, 0, 1], "through": True, "role": "bore"},
         ]},
    )


def main():
    print(f"Writing samples to {OUT} (deflection={DEFLECTION} mm)")
    truths = [cube(), plate_with_holes(), cylinder(), l_bracket(), flanged_pipe()]
    (OUT / "samples.json").write_text(json.dumps(truths, indent=2))
    print(f"Wrote {len(truths)} samples + samples.json")


if __name__ == "__main__":
    sys.exit(main())
