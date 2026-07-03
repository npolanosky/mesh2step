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


def countersink_plate():
    # 40 x 40 x 10 plate; through hole r=2.5 with a 90-degree countersink
    # (cone from r=2.5 at z=7.5 up to r=5 at the z=10 surface => 45deg half-angle).
    plate = Part.makeBox(40, 40, 10)
    hole = Part.makeCylinder(2.5, 10, App.Vector(20, 20, 0))
    csink = Part.makeCone(5, 2.5, 2.5, App.Vector(20, 20, 10), App.Vector(0, 0, -1))
    part = plate.cut(hole).cut(csink)
    return save(
        part,
        "countersink_plate",
        {"kind": "countersink_plate", "dims_mm": [40, 40, 10],
         "cylinders": [{"radius": 2.5, "axis": [0, 0, 1], "through": True}],
         "cones": [{"r_large": 5.0, "r_small": 2.5, "half_angle_deg": 45.0, "axis": [0, 0, 1]}]},
    )


def angled_hole_plate():
    import math
    # 50 x 40 x 20 plate with a hole drilled at 30deg from vertical (axis in xz).
    plate = Part.makeBox(50, 40, 20)
    a = math.radians(30)
    axis = App.Vector(math.sin(a), 0, math.cos(a))
    base = App.Vector(25, 20, 20) - axis * 30
    hole = Part.makeCylinder(4, 60, base, axis)
    part = plate.cut(hole)
    return save(
        part,
        "angled_hole_plate",
        {"kind": "angled_hole_plate", "dims_mm": [50, 40, 20],
         "cylinders": [{"radius": 4.0, "axis": [round(axis.x, 4), 0, round(axis.z, 4)],
                        "through": True, "angled": True}]},
    )


def fillet_chamfer_plate():
    # 40 x 30 x 12 plate. One long top edge (y=0, z=12) is filleted to R=3;
    # the opposite long top edge (y=30, z=12) is chamfered at 45deg, width 3.
    # Ground truth for M1 straight-edge fillet + chamfer reconstruction.
    L, W, H, R, C = 40.0, 30.0, 12.0, 3.0, 3.0
    plate = Part.makeBox(L, W, H)
    # Fillet the y=0/z=H edge: subtract a box then add the rounding cylinder.
    # Easiest exact way: use Part's fillet on the specific edge.
    edges = []
    for e in plate.Edges:
        v = e.Vertexes
        p0 = App.Vector(v[0].X, v[0].Y, v[0].Z)
        p1 = App.Vector(v[-1].X, v[-1].Y, v[-1].Z)
        mid = (p0 + p1) * 0.5
        length = (p1 - p0).Length
        # long edges run in x; pick the two top long edges by y.
        if length > L - 1e-6 and abs(mid.z - H) < 1e-6:
            edges.append((e, mid.y))
    fillet_edge = min(edges, key=lambda t: t[1])[0]   # y=0 edge gets the fillet
    part = plate.makeFillet(R, [fillet_edge])
    # Re-find the chamfer edge on the filleted solid (edge indices changed).
    ce = None
    for e in part.Edges:
        v = e.Vertexes
        p0 = App.Vector(v[0].X, v[0].Y, v[0].Z)
        p1 = App.Vector(v[-1].X, v[-1].Y, v[-1].Z)
        mid = (p0 + p1) * 0.5
        length = (p1 - p0).Length
        if length > L - 1e-6 and abs(mid.y - W) < 1e-6 and abs(mid.z - H) < 1e-6:
            ce = e
            break
    if ce is not None:
        part = part.makeChamfer(C, [ce])
    return save(
        part,
        "fillet_chamfer_plate",
        {"kind": "fillet_chamfer_plate", "dims_mm": [L, W, H],
         "cylinders": [],
         "fillets": [{"radius": R, "axis": [1, 0, 0], "convex": True}],
         "chamfers": [{"width": C, "angle_deg": 45.0}]},
    )


def main():
    print(f"Writing samples to {OUT} (deflection={DEFLECTION} mm)")
    truths = [cube(), plate_with_holes(), cylinder(), l_bracket(), flanged_pipe(),
              countersink_plate(), angled_hole_plate(), fillet_chamfer_plate()]
    (OUT / "samples.json").write_text(json.dumps(truths, indent=2))
    print(f"Wrote {len(truths)} samples + samples.json")


if __name__ == "__main__":
    sys.exit(main())
