"""End-to-end conversion smoke test over the sample parts.

Run under FreeCAD's Python (it provides numpy + the Part/Mesh kernel):

    "C:\\Program Files\\FreeCAD 1.1\\bin\\python.exe" scripts/smoke_convert.py

Converts every sample to STEP and asserts the result is a valid solid with the
expected number of analytic cylindrical faces. Exits non-zero on any failure.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import FreeCAD  # noqa: F401,E402  (must precede `import Part`)
import Part  # noqa: E402

from mesh2step.config import ConversionConfig  # noqa: E402
from mesh2step.pipeline import convert  # noqa: E402

DATA = REPO / "tests" / "data"
OUT = REPO / "build" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)
truths = {t["file"]: t for t in json.loads((DATA / "samples.json").read_text())}

failures = 0
# Known-partial cases tracked but not counted as failures by this smoke check,
# which only exercises the DEFAULT (plain reconstructed) path and asserts
# cylinder/cone counts. These parts get their analytic geometry from the curved-
# feature tiers (spheres/fillets/swept walls/freeform sheets) delivered via the
# boolean-clean / fully-closed path, not plain reconstruction — they are covered
# by pytest (test_spheres/test_swept/test_freeform), not here:
#   angled_hole_plate  — arbitrary-axis hole (not detected in the plain path)
#   domed_plate        — M3 sphere (boolean tier)
#   fillet_chamfer_plate — M1 fillet (boolean tier)
#   swept_wavy_wall    — M4 swept wall (boolean/closed tier)
#   freeform_bump      — Candidate B B-spline sheet (boolean/closed tier)
KNOWN_PARTIAL = {"angled_hole_plate", "domed_plate", "fillet_chamfer_plate",
                 "swept_wavy_wall", "freeform_bump"}

for stl in sorted(DATA.glob("*.stl")):
    truth = truths[stl.name]
    out = OUT / (stl.stem + ".step")
    res = convert(stl, out, ConversionConfig())

    shape = Part.Shape()
    shape.read(str(out))
    solids = shape.Solids
    cyl = sum(1 for f in shape.Faces if f.Surface.TypeId == "Part::GeomCylinder")
    cone = sum(1 for f in shape.Faces if f.Surface.TypeId == "Part::GeomCone")
    valid = bool(solids) and solids[0].isValid()
    want_cyl = len(truth["cylinders"])
    want_cone = len(truth.get("cones", []))

    ok = (valid and res.method == "reconstructed"
          and cyl == want_cyl and cone == want_cone)
    if stl.stem in KNOWN_PARTIAL:
        tag = "~~ "  # tracked, not a failure
    else:
        tag = "OK " if ok else "XX "
        failures += not ok
    print(f"[{tag}] {stl.name:22s} solid={valid} faces={len(shape.Faces)} "
          f"cyl={cyl}/{want_cyl} cone={cone}/{want_cone} method={res.method}")

print(f"\n{'PASSED' if not failures else f'{failures} FAILED'}  "
      f"(~~ = known-partial, tracked)")
sys.exit(1 if failures else 0)
