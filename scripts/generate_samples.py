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


def hex_and_round_plate():
    # Ground truth for the designed-polygon guard (P0): a plate with ONE regular
    # hexagonal through-hole and ONE round through-hole of the SAME nominal size
    # (both circum/hole radius 6). The hexagon must stay 6 flat planes (never be
    # replaced by an analytic cylinder); the round hole must fit a cylinder.
    plate = Part.makeBox(60, 30, 10)
    # Round hole (r=6) at (18, 15).
    rnd = Part.makeCylinder(6, 10, App.Vector(18, 15, 0))
    # Regular hexagon (circumradius 6) at (42, 15), extruded through the plate.
    import math as _m
    cx, cy, R = 42.0, 15.0, 6.0
    hpts = [App.Vector(cx + R * _m.cos(_m.radians(60 * k)),
                       cy + R * _m.sin(_m.radians(60 * k)), 0.0) for k in range(6)]
    hpts.append(hpts[0])
    hexwire = Part.makePolygon(hpts)
    hexprism = Part.Face(hexwire).extrude(App.Vector(0, 0, 10))
    part = plate.cut(rnd).cut(hexprism)
    return save(
        part,
        "hex_and_round_plate",
        {
            "kind": "hex_and_round_plate",
            "dims_mm": [60, 30, 10],
            # Only the ROUND hole is a true cylinder; the hexagon is a designed
            # polygon and is intentionally NOT listed (it must stay planar).
            "cylinders": [
                {"radius": 6.0, "axis": [0, 0, 1], "through": True},
            ],
            "hexagons": [
                {"circumradius": 6.0, "sides": 6, "axis": [0, 0, 1],
                 "center": [cx, cy], "through": True},
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


def swept_wavy_wall():
    """A constant-cross-section extruded wall with a known profile: a straight
    run, a tangent circular arc, and another straight run (line + arc + line with
    tangent joins), extruded along Z. Ground truth for M4 swept-wall
    reconstruction — the fitted profile must recover the arc radius and the
    extrusion, and RTAF must drop to ~0.

    Profile (in the XY plane): start at (0,0), go +X to (30,0); a tangent arc of
    radius R=10 turning the direction upward by 90deg (center at (30,10), ending
    at (40,10)); then +Y to (40,40). The wall is that profile given a 3 mm
    thickness (offset) and extruded 50 mm in Z. The outer edge is the line+arc+
    line curve; tangency at both joins is exact by construction.
    """
    import math

    R, T, HZ = 10.0, 3.0, 50.0
    # Outer profile points (fine): line (0,0)->(30,0), arc r=10 center (30,10)
    # from angle -90deg to 0deg, line (40,10)->(40,40).
    outer = [App.Vector(0, 0, 0), App.Vector(30, 0, 0)]
    steps = 24
    for k in range(1, steps + 1):
        a = math.radians(-90 + 90 * k / steps)
        outer.append(App.Vector(30 + R * math.cos(a), 10 + R * math.sin(a), 0))
    outer.append(App.Vector(40, 40, 0))
    # Inner profile: offset the outer curve inward (toward +Y for the first line,
    # toward -X for the last line) by T. Build via a simple parallel offset:
    # first line y=0 -> y=T; arc radius R-T same center; last line x=40 -> x=40-T.
    inner = [App.Vector(40 - T, 40, 0), App.Vector(40 - T, 10, 0)]
    for k in range(steps, -1, -1):
        a = math.radians(-90 + 90 * k / steps)
        inner.append(App.Vector(30 + (R - T) * math.cos(a), 10 + (R - T) * math.sin(a), 0))
    inner.append(App.Vector(30, T, 0))
    inner.append(App.Vector(0, T, 0))
    pts = outer + inner + [outer[0]]
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    solid = face.extrude(App.Vector(0, 0, HZ))
    return save(
        solid,
        "swept_wavy_wall",
        {"kind": "swept_wavy_wall", "dims_mm": [40, 40, HZ],
         "cylinders": [],
         "swept": [{"axis": [0, 0, 1], "extent": HZ,
                    "profile": "line+arc+line", "arc_radius": R,
                    "thickness": T}]},
    )


def domed_plate():
    """A flat plate with a convex spherical cap (dome) on top — ground truth for
    M3 sphere detection. The dome is a portion of a sphere of known radius R=20
    fused onto a 60x60x10 plate, so the fitted sphere must recover R=20 (and the
    cap's centre) while a prismatic part yields no false-positive sphere.

    The cap is built as the part of a ball of radius R centred above the plate top
    that pokes above it: centre at (30,30, 10 + (R - h)) where ``h`` is the cap
    height, so the sphere meets the plate top tangentially around a circle. Fusing
    the whole ball and keeping only the material above the plate would need a
    trim; instead we intersect the ball with a tall box over the plate footprint
    and fuse that cap, which leaves a clean dome tangent to the top face.
    """
    R = 20.0
    H = 6.0  # cap height above the plate top (a shallow cap, small sagitta)
    plate = Part.makeBox(60, 60, 10)
    # Sphere centre sits BELOW the plate top by (R - H) so only a shallow spherical
    # cap of height H rises above z=10 — apex at z = cz + R = 10 + H = 16.
    cz = 10.0 - (R - H)
    ball = Part.makeSphere(R, App.Vector(30, 30, cz))
    # Keep only the portion above the plate top (z >= 10): intersect with a slab.
    slab = Part.makeBox(60, 60, R, App.Vector(0, 0, 10.0))
    cap = ball.common(slab)
    part = plate.fuse(cap).removeSplitter()
    return save(
        part,
        "domed_plate",
        {"kind": "domed_plate", "dims_mm": [60, 60, 10 + H],
         "cylinders": [],
         "spheres": [{"radius": R, "center": [30.0, 30.0, round(cz, 4)],
                      "cap_height": H, "outward": True}]},
    )


def freeform_bump():
    """A plate whose top is a doubly-curved sinusoidal bump — ground truth for
    Candidate B (freeform B-spline sheet). z = amp*sin(0.3x)*cos(0.25y) over a
    40x40 footprint on a slab; the surface curves in BOTH parametric directions
    (not a cylinder, cone, sphere, or constant-cross-section sweep), so only a
    fitted B-spline sheet can reproduce it. Tessellation shatters it into a fan
    of thin planar strips (high RTAF); the freeform detector must collapse that
    fan into one analytic B-spline face (RTAF -> ~0, deviation < tol).

    Built by lofting the bump's isocurves into a B-spline face, then making a
    solid between that top and a flat base with vertical side walls.
    """
    import math

    span = 40.0
    amp = 3.0
    base_z = 0.0
    mid = 6.0  # mean height of the bump top above the base

    def zf(x, y):
        return amp * math.sin(x * 0.3) * math.cos(y * 0.25) + mid

    # Loft isocurves (one B-spline curve per x-row) into the top B-spline face.
    nu = 24
    xs = [span * i / (nu - 1) for i in range(nu)]
    ys = [span * j / (nu - 1) for j in range(nu)]
    curves = []
    for x in xs:
        pts = [App.Vector(x, y, zf(x, y)) for y in ys]
        c = Part.BSplineCurve()
        c.interpolate(pts)
        curves.append(c.toShape())
    top = Part.makeLoft(curves, False, True)  # smooth loft, not a solid
    top_face = top.Faces[0] if top.Faces else top

    # Thin plate: a short box (top well above the bump's peak) with the bump
    # carved out of its top by cutting the half-space above the top face. The
    # box is only as tall as the bump peak + margin, so the side walls are a
    # thin rim (not a tall block that would dwarf the curved top).
    top_of_box = mid + amp + 2.0
    box = Part.makeBox(span, span, top_of_box - base_z, App.Vector(0, 0, base_z))
    tool = top_face.extrude(App.Vector(0, 0, top_of_box + 5.0))
    part = box.cut(tool).removeSplitter()
    if not (part.Solids and part.Solids[0].isValid()):
        raise RuntimeError("freeform_bump: could not build a valid solid")
    return save(
        part,
        "freeform_bump",
        {"kind": "freeform_bump",
         "dims_mm": [span, span, round(mid + amp - base_z, 3)],
         "cylinders": [], "spheres": [],
         "freeform": {"amp": amp, "span": span,
                      "form": "amp*sin(0.3x)*cos(0.25y)+6"}},
    )


def threaded_rod():
    """A cylinder with a real helical thread of KNOWN pitch — ground truth for
    M5.2 thread detection. A V thread profile is swept along a helix of pitch
    P=2.0 mm over an R=6 mm core (depth 1.0 mm) and fused to the core, so the
    detector must recover the pitch and suppress the band to a cylinder.

    Built with Part.makeHelix + Part.BRepOffsetAPI.MakePipeShell (the profile is
    a WIRE, per the OCC API). A finer tessellation keeps the flanks sampled so the
    helix phase-fit has a clean signal.
    """
    R = 6.0        # core radius
    P = 2.0        # pitch (mm) — the ground truth to recover
    H = 20.0       # threaded length
    depth = 1.0    # thread depth
    core = Part.makeCylinder(R, H)
    helix = Part.makeHelix(P, H, R)
    # Symmetric V thread profile as a WIRE in the x-z plane at the helix start.
    a = App.Vector(R, 0, -P * 0.35)
    b = App.Vector(R + depth, 0, 0)
    c = App.Vector(R, 0, P * 0.35)
    wire = Part.makePolygon([a, b, c, a])
    mps = Part.BRepOffsetAPI.MakePipeShell(helix)
    mps.setFrenetMode(True)
    mps.add(wire)
    mps.build()
    mps.makeSolid()
    solid = core.fuse(mps.shape()).removeSplitter()
    if not (solid.Solids and solid.Solids[0].isValid()):
        raise RuntimeError("threaded_rod: could not build a valid solid")
    # Finer deflection than the default so the thread flanks stay sampled.
    mesh = Mesh.Mesh()
    pts, facets = solid.tessellate(0.03)
    mesh.addFacets([(pts[a2], pts[b2], pts[c2]) for a2, b2, c2 in facets])
    path = OUT / "threaded_rod.stl"
    mesh.write(str(path))
    truth = {"kind": "threaded_rod",
             "dims_mm": [2 * (R + depth), 2 * (R + depth), H],
             "cylinders": [],
             "threads": [{"pitch": P, "core_radius": R, "depth": depth,
                          "starts": 1, "handedness": "right", "is_internal": False}],
             "file": path.name, "triangles": mesh.CountFacets}
    print(f"  {path.name}: {mesh.CountFacets} triangles")
    return truth


def knurled_band():
    """A cylinder whose mid-section is a diamond knurl — ground truth for M5.1.

    Two crossing families of shallow helical grooves are cut into an R=10 mm
    cylinder over a central band, giving the micro-roughness a knurl detector
    keys off. The detector must claim the band and SUPPRESS it to a cylinder near
    R (crest radius). Note (design §3): a helical-cut synthetic knurl can read as
    a *thread* rather than a diamond — that is harmless, since both suppress the
    band to the same cylinder; the test checks the geometric suppression, not the
    metadata label.
    """
    R = 10.0
    H = 24.0
    band_lo, band_hi = 8.0, 16.0
    depth = 0.35
    rod = Part.makeCylinder(R, H)
    tools = []
    for left in (False, True):   # right- and left-hand helices (a diamond)
        helix = Part.makeHelix(8.0, band_hi - band_lo, R, 0, left)
        s = 0.7
        a = App.Vector(R - depth, -s, 0)
        b = App.Vector(R + depth, -s, 0)
        c = App.Vector(R + depth, s, 0)
        d = App.Vector(R - depth, s, 0)
        wire = Part.makePolygon([a, b, c, d, a])
        try:
            mps = Part.BRepOffsetAPI.MakePipeShell(
                helix.translated(App.Vector(0, 0, band_lo)))
            mps.setFrenetMode(True)
            mps.add(wire)
            mps.build()
            mps.makeSolid()
            tools.append(mps.shape())
        except Exception:  # noqa: BLE001
            pass
    solid = rod
    for t in tools:
        try:
            cut = solid.cut(t)
            if cut.Solids and cut.Solids[0].isValid():
                solid = cut
        except Exception:  # noqa: BLE001
            pass
    solid = solid.removeSplitter()
    if not (solid.Solids and solid.Solids[0].isValid()):
        solid = rod  # a plain rod still exercises "no knurl on a smooth cylinder"
    mesh = Mesh.Mesh()
    # Coarser deflection + a short band keep the sample small for git (a fine
    # knurl tessellates into hundreds of thousands of facets); the tolerant test
    # only needs the band claimed near R, not the full micro-roughness a scan has.
    pts, facets = solid.tessellate(0.3)
    mesh.addFacets([(pts[a2], pts[b2], pts[c2]) for a2, b2, c2 in facets])
    path = OUT / "knurled_band.stl"
    mesh.write(str(path))
    truth = {"kind": "knurled_band", "dims_mm": [2 * R, 2 * R, H],
             "cylinders": [],
             "knurling": [{"nominal_radius": R, "pattern": "diamond",
                           "band": [band_lo, band_hi], "outward": True}],
             "file": path.name, "triangles": mesh.CountFacets}
    print(f"  {path.name}: {mesh.CountFacets} triangles")
    return truth


def spur_gear():
    """A small spur gear (extruded involute-ish outline) — ground truth for M5.3.

    A closed toothed cross-section is extruded along Z and a central bore cut, so
    the whole-outline extrusion path must claim the outline as ONE extruded solid
    (a single guarded fuse, not per-tooth ops) with the bore surviving.
    """
    import math

    teeth = 12
    r_root = 8.0
    r_tip = 10.0
    thick = 6.0
    bore = 3.0
    pts = []
    for t in range(teeth):
        base = 2 * math.pi * t / teeth
        # simple trapezoidal tooth: root arc, rising flank, tip arc, falling flank
        prof = [(0.0, r_root), (0.25, r_root), (0.4, r_tip),
                (0.6, r_tip), (0.75, r_root), (1.0, r_root)]
        for frac, rr in prof:
            ang = base + (2 * math.pi / teeth) * frac
            pts.append(App.Vector(rr * math.cos(ang), rr * math.sin(ang), 0))
    pts.append(pts[0])
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    solid = face.extrude(App.Vector(0, 0, thick))
    hole = Part.makeCylinder(bore, thick, App.Vector(0, 0, 0))
    part = solid.cut(hole).removeSplitter()
    if not (part.Solids and part.Solids[0].isValid()):
        raise RuntimeError("spur_gear: could not build a valid solid")
    return save(
        part,
        "spur_gear",
        {"kind": "spur_gear", "dims_mm": [2 * r_tip, 2 * r_tip, thick],
         "cylinders": [{"radius": bore, "axis": [0, 0, 1], "through": True}],
         "gears": [{"teeth": teeth, "r_root": r_root, "r_tip": r_tip,
                    "extent": thick, "bore": bore}]},
    )


def main():
    print(f"Writing samples to {OUT} (deflection={DEFLECTION} mm)")
    truths = [cube(), plate_with_holes(), hex_and_round_plate(),
              cylinder(), l_bracket(), flanged_pipe(),
              countersink_plate(), angled_hole_plate(), fillet_chamfer_plate(),
              swept_wavy_wall(), domed_plate(), freeform_bump(),
              threaded_rod(), knurled_band(), spur_gear()]
    (OUT / "samples.json").write_text(json.dumps(truths, indent=2))
    print(f"Wrote {len(truths)} samples + samples.json")


if __name__ == "__main__":
    sys.exit(main())
