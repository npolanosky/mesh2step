"""Out-of-process OCC boolean CUT runner (timeout-guarded).

Some OCC boolean cuts grind for minutes even on a valid, small tool: a B-spline
tool face intersected against a faceted base can send ``BOPAlgo_PaveFiller``'s
face-face intersection (``PerformFF``) into a pathological path. Such a call is
uninterruptible in-process (it never checks a Python signal), so a wall-clock
budget in the caller cannot stop it — the only robust guard is process
isolation with a hard timeout.

This module is that child process. It is invoked as::

    <freecad-python> -m mesh2step.boolean_runner <base.brep> <tool.brep> <out.brep>

It imports FreeCAD + Part (FreeCAD FIRST, so ``import Part`` doesn't segfault on
macOS), reads the base + tool shapes from BREP files, performs ONE ``base.cut``
in this isolated process, and writes the result BREP back out. The parent runs
it under ``subprocess.run(..., timeout=...)`` and treats a timeout / non-zero
exit as a failed op (reverts to the pre-op solid — never regress, never hang).
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: boolean_runner <base.brep> <tool.brep> <out.brep>",
              file=sys.stderr)
        return 2
    base_path, tool_path, out_path = argv[1], argv[2], argv[3]
    import FreeCAD  # noqa: F401  (import first so `import Part` is safe)
    import Part

    base = Part.Shape()
    base.importBrep(base_path)
    tool = Part.Shape()
    tool.importBrep(tool_path)
    result = base.cut(tool)
    result.exportBrep(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
