"""STL loading and vertex welding.

STL files store three independent vertices per triangle with no shared
topology. To do any adjacency-based analysis we first weld coincident
vertices into a shared index space. This module is pure numpy and has no
FreeCAD dependency.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_BINARY_HEADER = 80


def _looks_binary(raw: bytes) -> bool:
    """Decide whether STL bytes are binary or ASCII.

    ASCII STL starts with "solid", but so can some binary files, so we also
    cross-check the declared triangle count against the file length.
    """
    if not raw[:5].lower().startswith(b"solid"):
        return True
    if len(raw) < _BINARY_HEADER + 4:
        return False
    n_tri = struct.unpack_from("<I", raw, _BINARY_HEADER)[0]
    expected = _BINARY_HEADER + 4 + n_tri * 50
    return expected == len(raw)


def _read_binary(raw: bytes) -> np.ndarray:
    """Return an (F, 3, 3) array of triangle vertices from binary STL bytes."""
    n_tri = struct.unpack_from("<I", raw, _BINARY_HEADER)[0]
    # Each triangle: 12 little-endian float32 (normal + 3 verts) + 2 byte attr.
    rec = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    data = np.frombuffer(raw, dtype=rec, count=n_tri, offset=_BINARY_HEADER + 4)
    return data["verts"].astype(np.float64)


def _read_ascii(text: str) -> np.ndarray:
    """Return an (F, 3, 3) array of triangle vertices from ASCII STL text."""
    verts: list[list[float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.asarray(verts, dtype=np.float64)
    if arr.size == 0 or arr.shape[0] % 3 != 0:
        raise ValueError("ASCII STL did not contain a whole number of triangles")
    return arr.reshape(-1, 3, 3)


def weld_vertices(
    tri_verts: np.ndarray, weld_tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and build a shared index space.

    Parameters
    ----------
    tri_verts : (F, 3, 3) float array
        Per-triangle vertex coordinates.
    weld_tol : float
        Vertices within this distance (per axis, after quantization) are merged.

    Returns
    -------
    vertices : (V, 3) float array of unique vertices.
    faces : (F, 3) int array indexing into ``vertices``.
    """
    flat = tri_verts.reshape(-1, 3)
    # Quantize to a grid of size weld_tol, then dedupe by the integer key.
    scale = 1.0 / max(weld_tol, 1e-12)
    keys = np.round(flat * scale).astype(np.int64)
    _, first_idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    vertices = flat[first_idx]
    faces = inverse.reshape(-1, 3).astype(np.int64)
    # Drop degenerate triangles (two welded corners collapsed to one vertex).
    good = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return vertices, faces[good]


def load_stl(
    path: str | Path, weld_tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL (binary or ASCII) and return welded ``(vertices, faces)``.

    ``vertices`` is (V, 3) float64; ``faces`` is (F, 3) int indexing vertices.
    """
    raw = Path(path).read_bytes()
    if _looks_binary(raw):
        tri = _read_binary(raw)
    else:
        tri = _read_ascii(raw.decode("utf-8", errors="replace"))
    return weld_vertices(tri, weld_tol=weld_tol)


def split_components(
    vertices: np.ndarray, faces: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a welded mesh into its disjoint connected components (bodies).

    Two facets belong to the same body if they share a (welded) vertex; the
    union-find over face-vertex connectivity groups facets into bodies that touch
    nowhere. A multi-body STL — a print-in-place hinge, a snap-fit lid+base — has
    several such components; a single rigid part has exactly one.

    Returns a list of ``(verts, faces)`` sub-meshes (each re-indexed to its own
    vertices), ordered largest-first by facet count. A single-component mesh
    returns a one-element list holding the input arrays unchanged, so callers can
    treat the common case identically without paying a copy.
    """
    n = len(vertices)
    if n == 0 or len(faces) == 0:
        return [(vertices, faces)]
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for tri in faces:
        a, b, c = find(int(tri[0])), find(int(tri[1])), find(int(tri[2]))
        m = min(a, b, c)
        parent[a] = parent[b] = parent[c] = m

    roots = np.array([find(i) for i in range(n)])
    face_root = roots[faces[:, 0]]
    unique_roots = np.unique(face_root)
    if unique_roots.size <= 1:
        return [(vertices, faces)]

    comps: list[tuple[np.ndarray, np.ndarray]] = []
    for r in unique_roots:
        mask = face_root == r
        comp_faces = faces[mask]
        used = np.unique(comp_faces)
        remap = np.full(n, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        comps.append((vertices[used].copy(), remap[comp_faces].copy()))
    comps.sort(key=lambda vf: len(vf[1]), reverse=True)
    return comps


def write_binary_stl(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    """Write a welded ``(vertices, faces)`` mesh to a binary STL file.

    Facet normals are computed from the triangle winding; the 2-byte attribute
    field is zero. Used to hand one body of a multi-body mesh to the existing
    single-body conversion path via a temp file (so every existing reload —
    boolean base, faceted fallback — sees exactly that one body).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tris = v[f]  # (F, 3, 3)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    rec = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    data = np.zeros(len(f), dtype=rec)
    data["normal"] = normals.astype("<f4")
    data["verts"] = tris.astype("<f4")
    with open(path, "wb") as fh:
        fh.write(b"\x00" * _BINARY_HEADER)
        fh.write(struct.pack("<I", len(f)))
        fh.write(data.tobytes())
