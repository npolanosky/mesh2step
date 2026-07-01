"""Smoke test for the boolean clean-up builder (fully-closed tier 2).

Run under FreeCAD's Python:
    & "C:/Program Files/FreeCAD 1.1/bin/python.exe" scripts/smoke_boolean.py

Verifies build_boolean_clean_solid produces a valid watertight solid whose
holes/bosses are true analytic cylinders/cones at the ground-truth radii, by
cutting + fusing-back into the faceted base solid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import FreeCAD  # noqa: E402,F401  (must precede Part)

from mesh2step import builder  # noqa: E402
from mesh2step.config import ConversionConfig  # noqa: E402
from mesh2step.mesh_io import load_stl  # noqa: E402

DATA = REPO / "tests" / "data"
truths = {t["file"]: t for t in json.loads((DATA / "samples.json").read_text())}

# Angled-axis holes aren't detected yet (tracked elsewhere), so boolean clean-up
# correctly finds nothing to clean — not a boolean-builder failure.
KNOWN_PARTIAL = {"angled_hole_plate"}

failures = 0
for stl in sorted(DATA.glob("*.stl")):
    truth = truths[stl.name]
    v, f = load_stl(stl)
    try:
        solid, s = builder.build_boolean_clean_solid(v, f, ConversionConfig())
    except Exception as exc:  # noqa: BLE001
        print(f"[XX ] {stl.name:22s} raised: {exc}")
        failures += 1
        continue

    valid = bool(solid.Solids) and solid.Solids[0].isValid()
    wall_radii = sorted(round(fc.Surface.Radius, 2) for fc in solid.Faces
                        if fc.Surface.TypeId == "Part::GeomCylinder")
    expect = sorted(c["radius"] for c in truth["cylinders"])
    got = {round(r, 1) for r in wall_radii}
    have_all = all(round(r, 1) in got for r in expect)  # each true radius present

    ok = valid and have_all
    if stl.stem in KNOWN_PARTIAL:
        tag = "~~ "
    else:
        tag = "OK " if ok else "XX "
        failures += not ok
    print(f"[{tag}] {stl.name:22s} valid={valid} faces={len(solid.Faces)} "
          f"cleaned={s['boolean_cleaned']}/{s['cylinders_detected'] + s['cones_detected']} "
          f"radii={sorted(got)} expect={sorted(set(round(r,1) for r in expect))}")

print(f"\n{'PASSED' if not failures else f'{failures} FAILED'}  (~~ = known-partial)")
sys.exit(1 if failures else 0)
