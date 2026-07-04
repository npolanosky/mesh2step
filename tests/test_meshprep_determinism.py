"""Decimation determinism regression guard (Task B).

Repeated runs of the pipeline must produce byte-identical decimated meshes, so
regression comparisons (RTAF / face-count / bbox diffs across commits) are stable.
Decimation is the only stage with a third-party mesh library in the loop
(pymeshlab), so it is the one to pin: ``meshprep_runner.decimate`` uses a fixed
filter chain with fixed parameters and no RNG, and the pipeline runs it in a fresh
SUBPROCESS, so identical input must give identical output.

The subprocess isolation is load-bearing here for a second reason: pymeshlab
bundles Qt 5 and FreeCAD bundles Qt 6, and importing both into one process aborts
(SIGTRAP). So this test drives the real out-of-process runner
(``meshprep.decimate_planar``) rather than importing pymeshlab in-process — which
would crash under FreeCAD's interpreter exactly as the pipeline's design avoids.
It is skipped when the runner can't locate a working pymeshlab.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


def _bumpy_sphere(n_lat: int = 40, n_lon: int = 60, r: float = 20.0):
    """A closed, over-tessellated bumpy sphere — enough facets that decimation to a
    target must make tie-breaking collapse choices (where nondeterminism, if any,
    would surface)."""
    verts = []
    idx = {}
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            rr = r * (1.0 + 0.05 * np.sin(5 * phi) * np.sin(3 * theta))
            idx[(i, j)] = len(verts)
            verts.append([rr * np.sin(theta) * np.cos(phi),
                          rr * np.sin(theta) * np.sin(phi),
                          rr * np.cos(theta)])
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = idx[(i, j)]
            b = idx[(i, (j + 1) % n_lon)]
            c = idx[(i + 1, (j + 1) % n_lon)]
            d = idx[(i + 1, j)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _hash(v: np.ndarray, f: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(v, dtype=np.float64).tobytes()
        + np.ascontiguousarray(f, dtype=np.int64).tobytes()
    ).hexdigest()


def _runner_available() -> bool:
    """True if the out-of-process pymeshlab decimation runner can run here."""
    try:
        from mesh2step import meshprep

        return meshprep._pymeshlab_importable()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _runner_available(),
                    reason="out-of-process pymeshlab decimation runner unavailable")
def test_decimation_is_byte_deterministic():
    """Three consecutive decimations of the same mesh to the same target, each via
    the real out-of-process runner, are byte-identical (same face count AND same
    vertex/face data)."""
    from mesh2step import meshprep

    v, f = _bumpy_sphere()
    hashes = []
    counts = []
    for _ in range(3):
        out_v, out_f, report = meshprep.decimate_planar(v.copy(), f.copy(), 1200)
        assert "error" not in report, report
        hashes.append(_hash(out_v, out_f))
        counts.append(int(len(out_f)))
    assert len(set(counts)) == 1, f"face counts drifted across runs: {counts}"
    assert len(set(hashes)) == 1, "decimated mesh is not byte-identical across runs"
