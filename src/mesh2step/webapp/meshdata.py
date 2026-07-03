"""Mesh payloads for the browser viewer + server-side deviation heatmap.

The browser needs three things to draw:

  * the input STL (raw triangle soup),
  * the converted STEP tessellated to a mesh, and
  * that same STEP mesh coloured by per-vertex deviation from the STL.

We ship each as a small **binary blob** (``M2SM`` format below) rather than glTF.
Rationale: the data is a flat non-indexed triangle soup with optional per-vertex
colours — exactly three or four ``Float32Array`` / ``Uint8Array`` views. A custom
header lets the browser ``new Float32Array(buffer, offset, count)`` straight onto
the response with zero parsing, no third-party glTF loader to vendor, and a
payload ~40% smaller than an equivalent ASCII glTF. (glTF's advantages —
materials, scene graph, animation — are irrelevant here.)

This module is pure numpy; it never imports FreeCAD. The STEP is tessellated to
an STL by the worker subprocess (``tessellate`` mode) beforehand; here we only
load meshes and compute point-to-triangle distances.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ---- binary mesh format ---------------------------------------------------- #
#
# Layout (all little-endian):
#   magic   4 bytes  b"M2SM"
#   version uint32   = 1
#   flags   uint32   bit0: has vertex normals, bit1: has vertex colours (rgb u8)
#   nverts  uint32   number of vertices (triangle soup: multiple of 3)
#   --- then, tightly packed ---
#   positions  float32[nverts*3]
#   normals    float32[nverts*3]   (only if flags bit0)
#   colours    uint8  [nverts*3]   (only if flags bit1)
#
# The browser reads the header, then wraps typed-array views over the tail.

_MAGIC = b"M2SM"
_VERSION = 1
_FLAG_NORMALS = 1 << 0
_FLAG_COLORS = 1 << 1


def _face_normals(tri: np.ndarray) -> np.ndarray:
    """Per-triangle normals for an (F,3,3) triangle-vertex array."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def load_triangles(stl_path: str | Path) -> np.ndarray:
    """Return an (F,3,3) float32 triangle-vertex array from an STL file.

    Reuses the package's pure-numpy STL reader (no FreeCAD, no welding — we want
    the raw soup so flat shading and the deviation viewer match the mesh 1:1).
    """
    from ..mesh_io import _looks_binary, _read_ascii, _read_binary

    raw = Path(stl_path).read_bytes()
    tri = _read_binary(raw) if _looks_binary(raw) else _read_ascii(raw.decode("latin-1"))
    return np.ascontiguousarray(tri, dtype=np.float64)


def _pack(positions: np.ndarray, normals: np.ndarray | None,
          colors: np.ndarray | None) -> bytes:
    """Pack flat vertex arrays into the ``M2SM`` binary blob."""
    positions = np.ascontiguousarray(positions, dtype="<f4")
    nverts = positions.shape[0]
    flags = 0
    parts = [positions.tobytes()]
    if normals is not None:
        flags |= _FLAG_NORMALS
        parts.append(np.ascontiguousarray(normals, dtype="<f4").tobytes())
    if colors is not None:
        flags |= _FLAG_COLORS
        parts.append(np.ascontiguousarray(colors, dtype="<u1").tobytes())
    header = _MAGIC + struct.pack("<III", _VERSION, flags, nverts)
    return header + b"".join(parts)


def mesh_blob(stl_path: str | Path, *, with_normals: bool = True) -> bytes:
    """A drawable ``M2SM`` blob for a plain STL (no colours)."""
    tri = load_triangles(stl_path)
    fn = _face_normals(tri)
    positions = tri.reshape(-1, 3)
    normals = np.repeat(fn, 3, axis=0) if with_normals else None
    return _pack(positions, normals, None)


# ---- deviation heatmap ----------------------------------------------------- #


def _point_triangle_sq_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                            c: np.ndarray) -> np.ndarray:
    """Squared distance from each point ``p[i]`` to triangle ``(a[i],b[i],c[i])``.

    Vectorised closest-point-on-triangle (Ericson, *Real-Time Collision
    Detection*, §5.1.5), evaluated in parallel over all i. Returns squared
    distances so callers can defer the sqrt.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    n = p.shape[0]
    closest = np.empty_like(p)

    # Region masks (mutually exclusive, evaluated in priority order).
    done = np.zeros(n, dtype=bool)

    # Vertex A
    m = (d1 <= 0) & (d2 <= 0)
    closest[m] = a[m]
    done |= m

    # Vertex B
    m = (~done) & (d3 >= 0) & (d4 <= d3)
    closest[m] = b[m]
    done |= m

    # Edge AB
    m = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    denom = np.where(m, d1 - d3, 1.0)
    v = np.where(m, d1 / denom, 0.0)
    closest[m] = a[m] + v[m, None] * ab[m]
    done |= m

    # Vertex C
    m = (~done) & (d6 >= 0) & (d5 <= d6)
    closest[m] = c[m]
    done |= m

    # Edge AC
    m = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    denom = np.where(m, d2 - d6, 1.0)
    w = np.where(m, d2 / denom, 0.0)
    closest[m] = a[m] + w[m, None] * ac[m]
    done |= m

    # Edge BC
    m = (~done) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    denom = np.where(m, (d4 - d3) + (d5 - d6), 1.0)
    w = np.where(m, (d4 - d3) / denom, 0.0)
    closest[m] = b[m] + w[m, None] * (c[m] - b[m])
    done |= m

    # Interior (face)
    m = ~done
    denom = np.where(m, va + vb + vc, 1.0)
    v = np.where(m, vb / denom, 0.0)
    w = np.where(m, vc / denom, 0.0)
    closest[m] = a[m] + v[m, None] * ab[m] + w[m, None] * ac[m]

    diff = p - closest
    return np.einsum("ij,ij->i", diff, diff)


def _nearest_distance(points: np.ndarray, tri: np.ndarray,
                      *, block: int = 2000) -> np.ndarray:
    """Unsigned distance from each point to the nearest triangle in ``tri``.

    Brute force but vectorised and blocked, so memory stays bounded. Corpus
    meshes tessellate to a few thousand triangles; points are a similar count.
    A block of 2000 points keeps the (points x tris) working set modest while
    still amortising numpy overhead.
    """
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ntri = tri.shape[0]
    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], block):
        pb = points[start:start + block]
        # (P,1,3) broadcast against (T,3): compute per-(point,tri) sq dist by
        # tiling. Flatten to reuse the vectorised primitive, then min over tris.
        P = pb.shape[0]
        pp = np.repeat(pb, ntri, axis=0)
        aa = np.tile(a, (P, 1))
        bb = np.tile(b, (P, 1))
        cc = np.tile(c, (P, 1))
        sq = _point_triangle_sq_dist(pp, aa, bb, cc).reshape(P, ntri)
        out[start:start + block] = np.sqrt(sq.min(axis=1))
    return out


# Deviation colour ramp: blue (on-surface) -> cyan -> green -> yellow -> red
# (far). A compact jet-like ramp so it reads the same as the pyvista viewer.
_RAMP = np.array([
    [0.00, 0, 0, 255],
    [0.25, 0, 255, 255],
    [0.50, 0, 220, 0],
    [0.75, 255, 220, 0],
    [1.00, 255, 0, 0],
], dtype=np.float64)


def _colormap(t: np.ndarray) -> np.ndarray:
    """Map normalised deviation ``t`` in [0,1] to uint8 RGB via the jet ramp."""
    t = np.clip(t, 0.0, 1.0)
    stops = _RAMP[:, 0]
    cols = _RAMP[:, 1:]
    idx = np.clip(np.searchsorted(stops, t) - 1, 0, len(stops) - 2)
    lo = stops[idx]
    hi = stops[idx + 1]
    span = np.where(hi > lo, hi - lo, 1.0)
    f = ((t - lo) / span)[:, None]
    rgb = cols[idx] * (1 - f) + cols[idx + 1] * f
    return rgb.astype(np.uint8)


def deviation_payload(stl_path: str | Path, step_mesh_path: str | Path,
                      *, clamp: float | None = None) -> tuple[bytes, dict]:
    """Build the coloured ``M2SM`` blob + deviation stats for the heatmap tab.

    ``step_mesh_path`` is the STEP already tessellated to an STL by the worker.
    For every vertex of that mesh we measure the unsigned distance to the input
    STL surface, normalise against ``clamp`` (default: max(p95, max/2)) and map
    to a jet colour. Returns ``(blob, stats)`` where stats has max/rms/p95/mean
    (mm) and the colour-scale ``clamp`` used for the scalar bar.
    """
    step_tri = load_triangles(step_mesh_path)
    stl_tri = load_triangles(stl_path)

    step_verts = step_tri.reshape(-1, 3)
    dev = _nearest_distance(step_verts, stl_tri)

    stats = {
        "max": float(dev.max()) if dev.size else 0.0,
        "rms": float(np.sqrt(np.mean(dev ** 2))) if dev.size else 0.0,
        "p95": float(np.percentile(dev, 95)) if dev.size else 0.0,
        "mean": float(dev.mean()) if dev.size else 0.0,
    }
    hi = clamp if clamp is not None else max(stats["p95"], stats["max"] * 0.5, 1e-6)
    stats["clamp"] = float(hi)

    colors = _colormap(dev / hi)
    fn = _face_normals(step_tri)
    normals = np.repeat(fn, 3, axis=0)
    blob = _pack(step_verts, normals, colors)
    return blob, stats
