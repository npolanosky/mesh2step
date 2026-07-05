"""Out-of-process OCC boolean CUT runner (timeout-guarded).

Some OCC boolean cuts grind for minutes even on a valid, small tool: a B-spline
tool face intersected against a faceted base can send ``BOPAlgo_PaveFiller``'s
face-face intersection (``PerformFF``) into a pathological path. Such a call is
uninterruptible in-process (it never checks a Python signal), so a wall-clock
budget in the caller cannot stop it — the only robust guard is process
isolation with a hard timeout.

This module is that child process. It runs in two modes:

**One-shot** (legacy)::

    <freecad-python> -m mesh2step.boolean_runner <base.brep> <tool.brep> <out.brep>

reads the base + tool shapes, performs ONE ``base.cut``, writes the result.

**Server** (``--serve``): a persistent line-protocol worker that imports FreeCAD
ONCE and then loops reading JSON requests from stdin, doing one cut per request,
and writing a JSON response per line. This amortises the ~2-5 s FreeCAD cold
import across every isolated op of a run instead of paying it per op (a dense
part does dozens of isolated cuts; the cold import dominated the ~30 min server
timings). The parent (:func:`mesh2step.builder.isolated_cut`) keeps the worker
warm, sends BREP file paths, enforces a per-request wall-clock timeout, and
respawns the worker if it dies or a request times out — so a pathological cut is
still bounded and reverted exactly as before, never hanging, never regressing.

FreeCAD is imported FIRST (so ``import Part`` doesn't segfault on macOS).
"""

from __future__ import annotations

import json
import sys


def _do_cut(base_path: str, tool_path: str, out_path: str, Part) -> None:
    base = Part.Shape()
    base.importBrep(base_path)
    tool = Part.Shape()
    tool.importBrep(tool_path)
    result = base.cut(tool)
    result.exportBrep(out_path)


def serve() -> int:
    """Line-protocol boolean server: JSON request in, JSON result out, until EOF.

    Imports FreeCAD + Part up front so the first request already runs warm. Each
    request is ``{"base": path, "tool": path, "out": path, "id": n}``; the
    response is ``{"ok": true, "out": path, "id": n}`` or ``{"ok": false,
    "error": msg, "id": n}``. The server never dies from a bad cut (reported as
    an error line), only from EOF (parent exited) or a hard native crash /
    timeout kill (the parent respawns it)."""
    import FreeCAD  # noqa: F401 - import first so `import Part` is safe / warm
    import Part

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            if req.get("ping"):
                resp: dict = {"ok": True, "pong": True}
            else:
                _do_cut(req["base"], req["tool"], req["out"], Part)
                resp = {"ok": True, "out": req["out"]}
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            resp = {"ok": False, "error": str(exc)}
        if req_id is not None:
            resp["id"] = req_id
        print(json.dumps(resp), flush=True)
    return 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--serve"]:
        return serve()
    if len(argv) != 4:
        print("usage: boolean_runner <base.brep> <tool.brep> <out.brep> | --serve",
              file=sys.stderr)
        return 2
    base_path, tool_path, out_path = argv[1], argv[2], argv[3]
    import FreeCAD  # noqa: F401  (import first so `import Part` is safe)
    import Part

    _do_cut(base_path, tool_path, out_path, Part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
