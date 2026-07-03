"""API tests for the mesh2step web app (no FreeCAD required).

The conversion worker subprocess is faked by monkeypatching
``mesh2step.webapp.app.run_worker`` / ``tessellate_step``, so these tests
exercise the full HTTP surface — upload, queueing, SSE progress, results,
downloads, viewer payloads, failstore recording — without CAD dependencies.
The real end-to-end conversion path is covered by the manual verification in
docs/WEBAPP.md (and by running the server against a real FreeCAD).
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mesh2step.webapp import app as app_module  # noqa: E402
from mesh2step.webapp.config import WebConfig  # noqa: E402

DATA = Path(__file__).parent / "data"
CUBE = DATA / "cube.stl"


# --------------------------------------------------------------------------- #
# fakes                                                                        #
# --------------------------------------------------------------------------- #


def _fake_run_worker_ok(job, freecad_python, *, on_line=None, timeout=0):
    """Emulate a successful conversion: stream progress, write the STEP."""
    for msg in ("PROGRESS: Locating FreeCAD", "PROGRESS: Loading and welding mesh",
                "PROGRESS: Sewing 8 faces into a solid", "PROGRESS: Exporting STEP",
                "PROGRESS: Done"):
        if on_line:
            on_line(msg)
        time.sleep(0.01)
    out = Path(job["output"])
    out.write_text("ISO-10303-21; /* fake step */ END-ISO-10303-21;")
    return {
        "ok": True, "mode": "convert", "output": str(out), "outputs": [str(out)],
        "method": "reconstructed",
        "stats": {"is_solid": True, "quality": "good", "faces_in": 12, "faces_out": 8,
                  "rtaf": 0.0, "warnings": []},
    }


def _fake_run_worker_not_watertight(job, freecad_python, *, on_line=None, timeout=0):
    """Emulate a conversion that finishes but is NOT watertight (a corpus failure)."""
    if on_line:
        on_line("PROGRESS: Exporting STEP")
    out = Path(job["output"])
    out.write_text("ISO-10303-21; /* fake open step */ END-ISO-10303-21;")
    return {
        "ok": True, "mode": "convert", "output": str(out), "outputs": [str(out)],
        "method": "reconstructed",
        "stats": {"is_solid": False, "quality": "problems", "faces_in": 12,
                  "faces_out": 8,
                  "warnings": ["Result is not a single watertight solid."]},
    }


def _fake_tessellate(step_path, out_mesh, freecad_python, *, deflection=0.1,
                     timeout=0):
    """Pretend to tessellate the STEP: reuse the job's input STL as the mesh."""
    src = next(p for p in Path(step_path).parent.glob("*.stl")
               if p != Path(out_mesh))
    shutil.copyfile(src, out_mesh)
    return {"ok": True, "facets": 12}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _client(tmp_path, monkeypatch, worker=_fake_run_worker_ok) -> TestClient:
    monkeypatch.setattr(app_module, "run_worker", worker)
    monkeypatch.setattr(app_module, "tessellate_step", _fake_tessellate)
    # Keep tests hermetic: never provision real prep deps from a test run.
    from mesh2step import provision

    monkeypatch.setattr(provision, "ensure_prep_deps",
                        lambda fc, log=None, force=False: tmp_path)
    cfg = WebConfig(data_dir=tmp_path / "web", freecad_python="/fake/python",
                    failures_dir=str(tmp_path / "corpus"))
    return TestClient(app_module.create_app(cfg))


def _convert(client: TestClient, stl: Path = CUBE, options: dict | None = None) -> str:
    with open(stl, "rb") as fh:
        r = client.post("/api/convert",
                        files={"file": (stl.name, fh, "model/stl")},
                        data={"options": json.dumps(options or {"source_units": "mm"})})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(f"/api/jobs/{job_id}").json()
        if d["state"] in ("done", "failed"):
            return d
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def _parse_m2sm(blob: bytes) -> dict:
    assert blob[:4] == b"M2SM"
    version, flags, nverts = struct.unpack_from("<III", blob, 4)
    assert version == 1
    off = 16
    out = {"nverts": nverts, "flags": flags}
    out["positions"] = blob[off:off + nverts * 12]
    off += nverts * 12
    if flags & 1:
        out["normals"] = blob[off:off + nverts * 12]
        off += nverts * 12
    if flags & 2:
        out["colors"] = blob[off:off + nverts * 3]
        off += nverts * 3
    assert off == len(blob), "trailing bytes in mesh blob"
    return out


# --------------------------------------------------------------------------- #
# tests                                                                        #
# --------------------------------------------------------------------------- #


def test_health(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    d = client.get("/api/health").json()
    assert d["ok"] is True
    assert d["freecad_ready"] is True


def test_index_serves_ui(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "mesh2step" in r.text
    # Vendored viewer assets resolve.
    assert client.get("/static/vendor/three.min.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_convert_happy_path(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client, options={"source_units": "mm", "full_closed": True,
                                       "bogus_option": "ignored"})
    d = _wait_done(client, job_id)
    assert d["state"] == "done"
    assert d["result"]["stats"]["is_solid"] is True
    assert d["outputs"] == ["cube.step"]
    assert d["options"] == {"source_units": "mm", "full_closed": True}  # cleaned
    assert any("Sewing" in ln for ln in d["log"])

    # Download the STEP.
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.content.startswith(b"ISO-10303-21")

    # History lists it.
    jobs = client.get("/api/jobs").json()["jobs"]
    assert any(j["id"] == job_id and j["state"] == "done" for j in jobs)


def test_upload_rejects_non_stl(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/api/convert", files={"file": ("part.obj", b"x" * 200)})
    assert r.status_code == 400


def test_mesh_payloads(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    _wait_done(client, job_id)

    # Input STL blob: positions + normals, 12 triangles for the cube.
    r = client.get(f"/api/jobs/{job_id}/mesh/stl")
    assert r.status_code == 200
    m = _parse_m2sm(r.content)
    assert m["nverts"] == 36  # 12 tris * 3
    assert m["flags"] & 1

    # STEP blob (tessellation faked to the same cube).
    r = client.get(f"/api/jobs/{job_id}/mesh/step")
    assert r.status_code == 200
    assert _parse_m2sm(r.content)["nverts"] == 36

    # Heatmap blob: must carry vertex colours + stats header, and the deviation
    # of a mesh against itself must be ~0 (all vertices coloured "low"/blue).
    r = client.get(f"/api/jobs/{job_id}/mesh/heatmap")
    assert r.status_code == 200
    m = _parse_m2sm(r.content)
    assert m["flags"] & 2, "heatmap payload missing vertex colours"
    assert len(m["colors"]) == 36 * 3
    stats = json.loads(r.headers["X-Deviation-Stats"])
    assert stats["max"] < 1e-6
    assert stats["clamp"] > 0


def test_sse_events_for_finished_job(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    _wait_done(client, job_id)
    with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        body = "".join(chunk for chunk in r.iter_text())
    events = [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: ")]
    assert events[0]["type"] == "snapshot"
    assert any(e.get("state") == "done" for e in events)


def test_failing_mesh_records_to_failstore(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, worker=_fake_run_worker_not_watertight)
    # Toggle failure saving on (mirrors the GUI checkbox).
    r = client.post("/api/settings", json={"save_failures": True})
    assert r.json()["save_failures"] is True

    job_id = _convert(client)
    d = _wait_done(client, job_id)
    assert d["state"] == "done"  # conversion finished; result just isn't watertight
    assert d["result"]["stats"]["is_solid"] is False
    assert d["corpus_action"] is not None
    assert d["corpus_action"]["action"] == "saved"
    assert d["corpus_action"]["category"] == "not_watertight"

    # The corpus manifest + copied STL exist on disk and the API serves them.
    corpus_dir = tmp_path / "corpus"
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    assert len(manifest["files"]) == 1
    entry = next(iter(manifest["files"].values()))
    assert entry["category"] == "not_watertight"
    assert (corpus_dir / entry["file"]).is_file()

    api_corpus = client.get("/api/corpus").json()
    assert len(api_corpus["files"]) == 1


def test_failstore_not_recorded_when_toggle_off(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, worker=_fake_run_worker_not_watertight)
    job_id = _convert(client)
    d = _wait_done(client, job_id)
    assert d["corpus_action"] is None
    assert not (tmp_path / "corpus" / "manifest.json").exists()


def test_flag_for_improvement(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    _wait_done(client, job_id)
    r = client.post(f"/api/jobs/{job_id}/flag")
    assert r.status_code == 200
    action = r.json()["action"]
    assert action["category"] == "faceted_improvable"
    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    entry = next(iter(manifest["files"].values()))
    assert entry["category"] == "faceted_improvable"
    assert entry["history"][0]["outcome"] == "flagged"


def test_queueing_two_jobs(tmp_path, monkeypatch):
    """A second conversion submitted while the first runs must queue, not crash."""
    order: list[str] = []

    def slow_worker(job, freecad_python, *, on_line=None, timeout=0):
        order.append(Path(job["input"]).parent.name)
        time.sleep(0.3)
        return _fake_run_worker_ok(job, freecad_python, on_line=on_line)

    client = _client(tmp_path, monkeypatch, worker=slow_worker)
    a = _convert(client)
    b = _convert(client)
    da = _wait_done(client, a)
    db = _wait_done(client, b)
    assert da["state"] == "done" and db["state"] == "done"
    assert order == [a, b]  # serial: b waited for a


def test_rerun(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    _wait_done(client, job_id)
    r = client.post(f"/api/jobs/{job_id}/rerun")
    assert r.status_code == 200
    new_id = r.json()["id"]
    assert new_id != job_id
    d = _wait_done(client, new_id)
    assert d["state"] == "done"
    assert d["filename"] == "cube.stl"


def test_download_path_traversal_guarded(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    _wait_done(client, job_id)
    r = client.get(f"/api/jobs/{job_id}/download", params={"name": "../../etc/passwd"})
    assert r.status_code == 404
