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


def _fake_run_worker_ok(job, freecad_python, *, on_line=None, on_start=None, timeout=0):
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


def _fake_run_worker_not_watertight(job, freecad_python, *, on_line=None, on_start=None, timeout=0):
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


def _fake_tessellate_typed(step_path, out_blob, out_meta, freecad_python, *,
                           deflection=0.1, timeout=0):
    """Fake typed tessellation: one gray plane triangle + one orange residual."""
    import numpy as np

    from mesh2step.webapp.meshdata import _pack

    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64)
    normals = np.tile([0.0, 0.0, 1.0], (6, 1))
    colors = np.array([[203, 208, 215]] * 3 + [[249, 115, 22]] * 3,
                      dtype=np.uint8)
    Path(out_blob).write_bytes(_pack(pos, normals, colors))
    Path(out_meta).write_text(json.dumps({
        "legend": {
            "plane": {"color": [203, 208, 215], "label": "Planar (analytic)",
                      "faces": 1, "tris": 1, "area_mm2": 0.5, "area_frac": 0.5},
            "residual": {"color": [249, 115, 22], "label": "Residual tessellation",
                         "faces": 1, "tris": 1, "area_mm2": 0.5, "area_frac": 0.5},
        },
        "faces": 2, "tris": 2, "residual_faces": 1,
    }), encoding="utf-8")


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _client(tmp_path, monkeypatch, worker=_fake_run_worker_ok) -> TestClient:
    monkeypatch.setattr(app_module, "run_worker", worker)
    monkeypatch.setattr(app_module, "tessellate_step", _fake_tessellate)
    monkeypatch.setattr(app_module, "tessellate_typed", _fake_tessellate_typed)
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


def test_job_timestamps_server_authoritative(tmp_path, monkeypatch):
    """Jobs carry created/started/finished epochs and endpoints expose a server
    clock (``now``) so the UI can compute elapsed without the page-load clock."""
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    d = _wait_done(client, job_id)

    # Detail payload: monotonic timestamps + server-computed elapsed + now.
    assert d["created"] <= d["started"] <= d["finished"]
    assert d["elapsed"] == pytest.approx(d["finished"] - d["started"], abs=0.02)
    assert d["now"] >= d["finished"]

    # List payload: per-job timestamps + a top-level server clock for the
    # history page's live-ticking elapsed column.
    lst = client.get("/api/jobs").json()
    assert lst["now"] > 0
    me = next(j for j in lst["jobs"] if j["id"] == job_id)
    assert me["started"] == d["started"] and me["finished"] == d["finished"]

    # SSE snapshot: same timestamps + server now, so a late subscriber (page
    # reload / history Open) bases its timer on true process time.
    with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        body = "".join(chunk for chunk in r.iter_text())
    snap = next(json.loads(ln[6:]) for ln in body.splitlines()
                if ln.startswith("data: "))
    assert snap["type"] == "snapshot"
    assert snap["started"] == d["started"]
    assert snap["finished"] == d["finished"]
    assert snap["now"] > 0


def test_running_state_event_carries_started(tmp_path, monkeypatch):
    """The RUNNING state event includes the start epoch, so a client already
    watching a queued job can base its elapsed timer on the true start time."""
    import queue as _q
    import threading

    gate = threading.Event()

    def slow_worker(job, freecad_python, *, on_line=None, on_start=None, timeout=0):
        gate.wait(5.0)
        return _fake_run_worker_ok(job, freecad_python, on_line=on_line)

    client = _client(tmp_path, monkeypatch, worker=slow_worker)
    # Occupy the single worker, then queue a second job and subscribe to its
    # events BEFORE it starts (exactly what a watching browser does).
    a = _convert(client)
    _wait_state(client, a, ("running",))
    b = _convert(client)
    store = client.app.state.store
    sub = store.subscribe(b)
    gate.set()  # release a -> b starts

    running_ev = None
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            ev = sub.get(timeout=0.5)
        except _q.Empty:
            continue
        if ev.get("type") == "state" and ev.get("state") == "running":
            running_ev = ev
            break
    store.unsubscribe(b, sub)
    assert running_ev is not None, "never saw the running state event"
    assert running_ev.get("started"), "running event missing 'started' epoch"
    _wait_done(client, a)
    _wait_done(client, b)


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
    # Saving now defaults ON — turn it off explicitly for this test.
    client.post("/api/settings", json={"save_failures": False})
    job_id = _convert(client)
    d = _wait_done(client, job_id)
    assert d["corpus_action"] is None
    assert not (tmp_path / "corpus" / "manifest.json").exists()


def test_save_failures_defaults_on_and_persists(tmp_path, monkeypatch):
    """'Save failing models' defaults ON and the toggle survives a restart."""
    client = _client(tmp_path, monkeypatch, worker=_fake_run_worker_not_watertight)
    assert client.get("/api/health").json()["save_failures"] is True

    # Default ON: a non-watertight result lands in the corpus with no toggling.
    job_id = _convert(client)
    d = _wait_done(client, job_id)
    assert d["corpus_action"] is not None and d["corpus_action"]["action"] == "saved"

    # Turn it off; a fresh app over the same data dir (restart) must still be off.
    client.post("/api/settings", json={"save_failures": False})
    cfg = WebConfig(data_dir=tmp_path / "web", freecad_python="/fake/python",
                    failures_dir=str(tmp_path / "corpus"))
    client2 = TestClient(app_module.create_app(cfg))
    assert client2.get("/api/health").json()["save_failures"] is False


def test_steptypes_endpoint(tmp_path, monkeypatch):
    """/mesh/steptypes serves a colour-coded M2SM + X-Face-Types legend."""
    client = _client(tmp_path, monkeypatch)
    job_id = _convert(client)
    _wait_done(client, job_id)

    r = client.get(f"/api/jobs/{job_id}/mesh/steptypes")
    assert r.status_code == 200
    m = _parse_m2sm(r.content)
    assert m["flags"] & 2, "typed payload must carry vertex colours"
    assert m["nverts"] == 6
    # First triangle gray (plane), second orange (residual).
    assert m["colors"][0:3] == bytes([203, 208, 215])
    assert m["colors"][9:12] == bytes([249, 115, 22])
    meta = json.loads(r.headers["X-Face-Types"])
    assert meta["legend"]["plane"]["faces"] == 1
    assert meta["legend"]["residual"]["area_frac"] == 0.5
    assert meta["residual_faces"] == 1

    # Cached on disk: a second request works even with typed tess broken.
    monkeypatch.setattr(app_module, "tessellate_typed",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r2 = client.get(f"/api/jobs/{job_id}/mesh/steptypes")
    assert r2.status_code == 200 and r2.content == r.content

    # Unfinished/unknown job -> 404 with a JSON detail (the UI shows it).
    assert client.get("/api/jobs/nope/mesh/steptypes").status_code == 404


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

    # The flagged state is on the job record (what the UI's "already flagged"
    # button state reads on reopen)...
    d = client.get(f"/api/jobs/{job_id}").json()
    assert d["corpus_action"]["category"] == "faceted_improvable"
    # ...and survives a server restart (persisted to job.json).
    cfg = WebConfig(data_dir=tmp_path / "web", freecad_python="/fake/python",
                    failures_dir=str(tmp_path / "corpus"))
    client2 = TestClient(app_module.create_app(cfg))
    d2 = client2.get(f"/api/jobs/{job_id}").json()
    assert d2["corpus_action"]["category"] == "faceted_improvable"


def test_queueing_two_jobs(tmp_path, monkeypatch):
    """A second conversion submitted while the first runs must queue, not crash."""
    order: list[str] = []

    def slow_worker(job, freecad_python, *, on_line=None, on_start=None, timeout=0):
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


# --------------------------------------------------------------------------- #
# history: by-id access, dual outputs, restart persistence                     #
# --------------------------------------------------------------------------- #


def _fake_run_worker_two_outputs(job, freecad_python, *, on_line=None,
                                 on_start=None, timeout=0):
    """Emulate a conversion that writes TWO STEP files (e.g. solid + faceted)."""
    out = Path(job["output"])
    out2 = out.with_name(out.stem + "_faceted.step")
    out.write_text("ISO-10303-21; /* primary */ END-ISO-10303-21;")
    out2.write_text("ISO-10303-21; /* faceted */ END-ISO-10303-21;")
    return {
        "ok": True, "mode": "convert", "output": str(out),
        "outputs": [str(out), str(out2)], "method": "reconstructed",
        "stats": {"is_solid": True, "quality": "good", "faces_in": 12,
                  "faces_out": 8, "rtaf": 0.0, "warnings": []},
    }


def test_dual_output_job_lists_and_serves_both_files(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, worker=_fake_run_worker_two_outputs)
    job_id = _convert(client)
    d = _wait_done(client, job_id)
    assert d["outputs"] == ["cube.step", "cube_faceted.step"]

    # Each output is downloadable by name, keyed only by job id.
    r1 = client.get(f"/api/jobs/{job_id}/download", params={"name": "cube.step"})
    assert r1.status_code == 200 and b"primary" in r1.content
    r2 = client.get(f"/api/jobs/{job_id}/download",
                    params={"name": "cube_faceted.step"})
    assert r2.status_code == 200 and b"faceted" in r2.content
    # Default (no name) falls back to the first output.
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200 and b"primary" in r.content
    # The history list carries both output names for the UI's download links.
    jobs = client.get("/api/jobs").json()["jobs"]
    me = next(j for j in jobs if j["id"] == job_id)
    assert me["outputs"] == ["cube.step", "cube_faceted.step"]


def test_completed_jobs_openable_by_id_after_restart(tmp_path, monkeypatch):
    """Queued-then-completed jobs stay fully accessible by id across a restart.

    Two jobs are submitted back-to-back (concurrency 1, so the second takes the
    QUEUED -> RUNNING path). Then a brand-new app instance is built over the
    same data dir — simulating a server restart — and BOTH jobs must still
    serve detail, download, and viewer payloads purely from the job dir on
    disk (nothing keyed off in-memory 'current job' state).
    """
    client = _client(tmp_path, monkeypatch)
    a = _convert(client)
    b = _convert(client)  # queued behind a
    da = _wait_done(client, a)
    db = _wait_done(client, b)
    # The queue path must persist results identically to a direct run.
    for d in (da, db):
        assert d["state"] == "done"
        assert d["outputs"] == ["cube.step"]
        assert d["result"]["stats"]["is_solid"] is True
    # On-disk records match what the API returned.
    for jid in (a, b):
        rec = json.loads((tmp_path / "web" / "jobs" / jid / "job.json").read_text())
        assert rec["state"] == "done" and rec["outputs"] == ["cube.step"]
        assert rec["result"]["stats"]["quality"] == "good"

    # ---- simulate a restart: fresh app over the same data dir ---- #
    cfg = WebConfig(data_dir=tmp_path / "web", freecad_python="/fake/python",
                    failures_dir=str(tmp_path / "corpus"))
    client2 = TestClient(app_module.create_app(cfg))

    for jid in (a, b):
        d = client2.get(f"/api/jobs/{jid}").json()
        assert d["state"] == "done"
        assert d["result"]["stats"]["is_solid"] is True
        assert d["outputs"] == ["cube.step"]

        r = client2.get(f"/api/jobs/{jid}/download")
        assert r.status_code == 200 and r.content.startswith(b"ISO-10303-21")
        r = client2.get(f"/api/jobs/{jid}/download", params={"name": "cube.step"})
        assert r.status_code == 200

        # Viewer payloads: input STL and tessellated STEP re-derive from disk.
        r = client2.get(f"/api/jobs/{jid}/mesh/stl")
        assert r.status_code == 200 and r.content[:4] == b"M2SM"
        r = client2.get(f"/api/jobs/{jid}/mesh/step")
        assert r.status_code == 200 and r.content[:4] == b"M2SM"
        r = client2.get(f"/api/jobs/{jid}/mesh/heatmap")
        assert r.status_code == 200 and r.content[:4] == b"M2SM"
        assert "X-Deviation-Stats" in r.headers

    # SSE for a finished job replays its terminal state (what the history
    # "Open" button relies on to render the full result).
    with client2.stream("GET", f"/api/jobs/{a}/events") as r:
        body = "".join(chunk for chunk in r.iter_text())
    events = [json.loads(ln[6:]) for ln in body.splitlines()
              if ln.startswith("data: ")]
    assert events[0]["type"] == "snapshot"
    assert any(e.get("state") == "done" for e in events)


# --------------------------------------------------------------------------- #
# FreeCAD subprocess env parity                                                #
# --------------------------------------------------------------------------- #


def test_all_freecad_spawns_share_worker_env(tmp_path, monkeypatch):
    """Every FreeCAD subprocess the webapp launches — the conversion/tessellate
    worker (``run_worker``) AND the typed tessellation (``tessellate_typed``) —
    must build its environment through the same ``_worker_env`` helper. One
    source of truth: the viewer's tessellation can never again diverge from the
    conversion worker's env (the deployed 'Failed to load FreeCAD module!'
    class of bug)."""
    import os
    import subprocess as _sp

    from mesh2step.webapp import conversion

    sentinel_env = {"M2S_ENV_SENTINEL": "1", "PATH": os.environ.get("PATH", "")}
    monkeypatch.setattr(conversion, "_worker_env",
                        lambda fc: dict(sentinel_env))
    captured: list[tuple[str, dict | None]] = []

    class _FakeProc:
        pid = 4242
        returncode = 0
        stdout = iter(())

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def fake_popen(cmd, **kw):
        captured.append(("run_worker", kw.get("env")))
        res = Path(cmd[cmd.index("--result") + 1])
        res.write_text('{"ok": true, "mode": "tessellate"}')
        return _FakeProc()

    def fake_run(cmd, **kw):
        captured.append(("tessellate_typed", kw.get("env")))
        Path(cmd[cmd.index("--out-blob") + 1]).write_bytes(b"M2SM")
        Path(cmd[cmd.index("--out-meta") + 1]).write_text("{}")

        class _R:
            returncode = 0
            stderr = ""
            stdout = ""

        return _R()

    monkeypatch.setattr(_sp, "Popen", fake_popen)
    monkeypatch.setattr(_sp, "run", fake_run)

    conversion.run_worker({"mode": "tessellate", "input": "x", "output": "y"},
                          "/fake/python")
    conversion.tessellate_typed("in.step", tmp_path / "b.m2sm",
                                tmp_path / "m.json", "/fake/python")

    assert [k for k, _ in captured] == ["run_worker", "tessellate_typed"]
    for kind, env in captured:
        assert env is not None and env.get("M2S_ENV_SENTINEL") == "1", \
            f"{kind} spawn did not build its env via _worker_env"


# --------------------------------------------------------------------------- #
# cancel + active count                                                        #
# --------------------------------------------------------------------------- #


def _wait_state(client: TestClient, job_id: str, states, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(f"/api/jobs/{job_id}").json()
        if d["state"] in states:
            return d
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} not in {states} within {timeout}s")


def test_active_count(tmp_path, monkeypatch):
    """/api/active reports running + queued while a slow job is in flight."""
    gate = __import__("threading").Event()

    def slow_worker(job, freecad_python, *, on_line=None, on_start=None, timeout=0):
        gate.wait(5.0)
        return _fake_run_worker_ok(job, freecad_python, on_line=on_line)

    client = _client(tmp_path, monkeypatch, worker=slow_worker)
    assert client.get("/api/active").json() == {"running": 0, "queued": 0, "active": 0}

    a = _convert(client)
    b = _convert(client)  # queues behind a (concurrency 1)
    _wait_state(client, a, ("running",))
    act = client.get("/api/active").json()
    assert act["running"] == 1 and act["queued"] == 1 and act["active"] == 2

    gate.set()
    _wait_done(client, a)
    _wait_done(client, b)
    assert client.get("/api/active").json()["active"] == 0


def test_cancel_queued_job(tmp_path, monkeypatch):
    """A queued job is dequeued and marked cancelled; it never runs."""
    ran: list[str] = []
    gate = __import__("threading").Event()

    def slow_worker(job, freecad_python, *, on_line=None, on_start=None, timeout=0):
        ran.append(Path(job["input"]).parent.name)
        gate.wait(5.0)
        return _fake_run_worker_ok(job, freecad_python, on_line=on_line)

    client = _client(tmp_path, monkeypatch, worker=slow_worker)
    a = _convert(client)  # occupies the single worker
    b = _convert(client)  # sits in the queue
    _wait_state(client, a, ("running",))

    r = client.post(f"/api/jobs/{b}/cancel")
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    assert client.get(f"/api/jobs/{b}").json()["state"] == "cancelled"

    gate.set()
    _wait_done(client, a)
    # b must never have entered the worker.
    assert b not in ran
    # Cancelling a terminal job is a 409.
    assert client.post(f"/api/jobs/{a}/cancel").status_code == 409


def test_cancel_running_kills_process_tree(tmp_path, monkeypatch):
    """Cancelling a running job kills the worker AND its child (no orphans).

    Uses a runner that spawns a real subprocess tree (parent shell -> long-lived
    child ``sleep``) in its own session, registers it via ``emit("proc", proc)``
    exactly as the real conversion runner does, then asserts both PIDs are gone
    after cancel.
    """
    import os
    import subprocess

    from mesh2step.webapp.conversion import CancelledError

    pids: dict[str, int] = {}

    def tree_runner(job, emit):
        # Parent bash spawns a child `sleep` then waits; the child's PID is
        # printed so the test can check it too. start_new_session -> own pgroup.
        proc = subprocess.Popen(
            ["bash", "-c", "sleep 300 & echo CHILD:$!; wait"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        emit("proc", proc)
        pids["parent"] = proc.pid
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("CHILD:"):
                pids["child"] = int(line.split(":", 1)[1])
        proc.wait()
        if proc.returncode is not None and proc.returncode < 0:
            raise CancelledError("killed")
        return

    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    store = app_module.JobStore(tmp_path / "jobs", concurrency=1, runner=tree_runner)
    from mesh2step.webapp.jobs import Job

    job = Job(id="killme", filename="x.stl", options={})
    with store._lock:
        store._jobs[job.id] = job
    store._queue.put(job.id)

    # Wait until both parent + child are up and registered.
    deadline = time.time() + 5.0
    while time.time() < deadline and ("child" not in pids or job.id not in store._procs):
        time.sleep(0.02)
    assert "parent" in pids and "child" in pids, "subprocess tree did not start"
    assert _alive(pids["parent"]) and _alive(pids["child"])

    result = store.cancel(job.id)
    assert result is not None

    # The worker thread should finalize the job as cancelled.
    deadline = time.time() + 5.0
    while time.time() < deadline and store.get(job.id).state != "cancelled":
        time.sleep(0.02)
    assert store.get(job.id).state == "cancelled"

    # Both processes must be gone (killpg reached the child) — no orphans.
    deadline = time.time() + 5.0
    while time.time() < deadline and (_alive(pids["parent"]) or _alive(pids["child"])):
        time.sleep(0.05)
    assert not _alive(pids["parent"]), "worker parent still alive after cancel"
    assert not _alive(pids["child"]), "orphaned child survived cancel"
    # No lingering proc registration.
    assert job.id not in store._procs
