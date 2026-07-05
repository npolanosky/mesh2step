"""FastAPI application: routes, job runner, static UI.

The web process is FreeCAD-free. Conversions run in the worker subprocess (see
:mod:`.conversion`); the deviation heatmap is computed here in pure numpy
(:mod:`.meshdata`). Progress is streamed to the browser with **Server-Sent
Events** rather than WebSockets: the traffic is strictly one-way (server ->
browser progress/log), SSE auto-reconnects, needs no extra client library, and
survives proxies trivially — a plain ``text/event-stream`` GET. WebSockets would
add bidirectional machinery we don't need.
"""

from __future__ import annotations

import asyncio
import json
import queue
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .config import WebConfig
from .conversion import run_worker, tessellate_step, tessellate_typed
from .jobs import DONE, Job, JobStore

_STATIC = Path(__file__).resolve().parent / "static"

# Conversion options the UI sends, mapped straight onto ConversionConfig fields.
# Mirrors the desktop GUI's option set (units, detect, faceted, repair, closed).
_ALLOWED_OPTIONS = {
    "source_units": str,
    "detect_cylinders": bool,
    "faceted": bool,
    "repair_mesh": bool,
    "full_closed": bool,
}


def _clean_options(raw: dict) -> dict:
    out: dict = {}
    for key, typ in _ALLOWED_OPTIONS.items():
        if key in raw and raw[key] is not None:
            out[key] = typ(raw[key])
    return out


def _make_runner(cfg: WebConfig, store: JobStore, save_failures_flag):
    """Build the per-job conversion runner closure.

    ``save_failures_flag`` is a mutable single-element list so the toggle can be
    flipped at runtime from the UI without rebuilding the store.
    """

    prep_ready = [False]  # provision pymeshlab/manifold3d once per server run

    def runner(job: Job, emit) -> None:
        fc = cfg.freecad_python
        if not fc:
            raise RuntimeError(
                "FreeCAD Python not found. Set MESH2STEP_FREECAD_PYTHON or install "
                "FreeCAD 0.20+ (see docs/WEBAPP.md).")

        # First conversion of this server run: make sure the prep deps
        # (manifold3d + pymeshlab) are importable by FreeCAD's Python — same
        # self-provisioning the desktop GUI does. Best-effort: the pipeline
        # degrades gracefully without them.
        if not prep_ready[0]:
            try:
                from .. import provision

                prep_ready[0] = provision.ensure_prep_deps(
                    fc, log=lambda m: emit("log", m)) is not None
            except Exception as exc:  # noqa: BLE001
                emit("log", f"prep-dep provisioning error: {exc}")

        jd = store.job_dir(job.id)
        stl = store.input_path(job.id)
        out_step = jd / (Path(job.filename).stem + ".step")

        conv_job = {
            "mode": "convert",
            "input": str(stl),
            "output": str(out_step),
            "config": job.options,
        }
        result = run_worker(conv_job, fc, on_line=lambda ln: emit("log", ln),
                            on_start=lambda proc: emit("proc", proc),
                            timeout=cfg.convert_timeout)
        job.result = result

        # Record failures into the corpus when the toggle is on (mirrors the GUI).
        if save_failures_flag[0]:
            try:
                from .. import failstore

                action = failstore.record_result(
                    str(stl), result, dest=cfg.failures_dir,
                    log=lambda m: emit("log", m))
                if action:
                    emit("corpus", action)
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a job
                emit("log", f"failure-corpus bookkeeping error: {exc}")

        if not result.get("ok"):
            raise RuntimeError(result.get("error", "conversion failed"))

        outputs = result.get("outputs") or [result.get("output")]
        written = [Path(p) for p in outputs if p and Path(p).exists()]
        emit("output", [p.name for p in written])
        emit("log", f"PROGRESS: Done — {len(written)} file(s) written")

    return runner


def create_app(config: WebConfig | None = None, *, runner=None) -> FastAPI:
    """Build the FastAPI app. ``runner`` overrides the conversion runner (tests)."""
    cfg = config or WebConfig()
    cfg.ensure_dirs()
    if cfg.freecad_python is None:
        try:
            from ..freecad_env import find_freecad_python

            cfg.freecad_python = find_freecad_python()
        except Exception:  # noqa: BLE001
            cfg.freecad_python = None

    app = FastAPI(title="mesh2step-web")

    # "Save failing models" defaults ON and persists across restarts
    # (data_dir/settings.json), so the failure corpus keeps growing unless the
    # user explicitly turns it off.
    settings_path = cfg.data_dir / "settings.json"
    save_failures = [True]
    try:
        persisted = json.loads(settings_path.read_text(encoding="utf-8"))
        save_failures[0] = bool(persisted.get("save_failures", True))
    except (OSError, ValueError):
        pass

    def _persist_settings() -> None:
        try:
            settings_path.write_text(
                json.dumps({"save_failures": save_failures[0]}), encoding="utf-8")
        except OSError:
            pass

    store = JobStore(cfg.jobs_dir, concurrency=cfg.concurrency,
                     runner=runner or _make_runner(cfg, None, save_failures))
    # The default runner needs the store (for paths); rebuild now that it exists.
    if runner is None:
        store._runner = _make_runner(cfg, store, save_failures)

    app.state.cfg = cfg
    app.state.store = store
    app.state.save_failures = save_failures

    # ---- meta ------------------------------------------------------------- #
    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """Cheap liveness probe for container/orchestrator healthchecks.

        Always 200 while the process is serving; it deliberately does NOT gate on
        FreeCAD (use ``/api/health`` -> ``freecad_ready`` for readiness) so a
        transient FreeCAD detection miss doesn't make the orchestrator kill an
        otherwise-healthy web process. Returns a tiny body, no disk/subprocess.
        """
        return JSONResponse({"status": "ok"})

    @app.get("/api/health")
    def health() -> dict:
        from .. import DISPLAY_VERSION

        return {
            "ok": True,
            "version": DISPLAY_VERSION,
            "freecad": cfg.freecad_python,
            "freecad_ready": bool(cfg.freecad_python),
            "concurrency": cfg.concurrency,
            "save_failures": save_failures[0],
        }

    @app.post("/api/settings")
    async def settings(request: Request) -> dict:
        body = await request.json()
        if "save_failures" in body:
            save_failures[0] = bool(body["save_failures"])
            _persist_settings()
        return {"save_failures": save_failures[0]}

    # ---- convert ---------------------------------------------------------- #
    @app.post("/api/convert")
    async def convert(request: Request, file: UploadFile) -> JSONResponse:
        data = await file.read()
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Upload too large.")
        if len(data) < 84 or not (file.filename or "").lower().endswith(".stl"):
            raise HTTPException(status_code=400, detail="Please upload a .stl file.")
        form = await request.form()
        raw_opts = {}
        if form.get("options"):
            try:
                raw_opts = json.loads(form["options"])
            except (ValueError, TypeError):
                raw_opts = {}
        job = store.create(file.filename, _clean_options(raw_opts), data)
        return JSONResponse({"id": job.id})

    # ---- job state -------------------------------------------------------- #
    @app.get("/api/jobs")
    def jobs() -> dict:
        # Trim logs in the list view; the detail endpoint has the full log.
        # ``now`` is the server clock so the UI can tick running rows'
        # elapsed = now - started without trusting the client clock.
        out = []
        for j in store.list():
            d = j.public()
            d["log"] = d["log"][-3:]
            out.append(d)
        return {"jobs": out, "now": time.time()}

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        d = job.public()
        d["now"] = time.time()
        return d

    @app.get("/api/active")
    def active() -> dict:
        """In-progress job count (running + queued) for the main-page badge."""
        return store.active_count()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        """Cancel a queued or running job.

        Queued jobs are dequeued and marked cancelled; running jobs have their
        worker process tree killed (killpg) so the FreeCAD/meshprep child dies
        too. Idempotent-ish: a job that's already terminal returns 409.
        """
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        result = store.cancel(job_id)
        if result is None:
            raise HTTPException(status_code=409,
                                detail="Job is not running or queued.")
        return {"id": job_id, "state": result.state}

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun(job_id: str) -> dict:
        job = store.requeue(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Cannot re-run this job.")
        return {"id": job.id}

    @app.post("/api/jobs/{job_id}/flag")
    def flag(job_id: str) -> dict:
        """Flag a watertight result for improvement (failstore faceted_improvable)."""
        job = store.get(job_id)
        if job is None or job.state != DONE:
            raise HTTPException(status_code=404, detail="No finished job to flag.")
        from .. import failstore

        action = failstore.record_flag(
            str(store.input_path(job_id)), job.result, dest=cfg.failures_dir)
        job.corpus_action = action
        return {"action": action}

    # ---- streaming progress (SSE) ---------------------------------------- #
    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str) -> StreamingResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        sub = store.subscribe(job_id)

        async def gen():
            # Replay current state so a late subscriber isn't stuck. Timestamps
            # + the server clock (``now``) make the client's elapsed timer
            # server-authoritative: elapsed = server_now - started, immune to
            # page reloads and client clock skew.
            snapshot = {"type": "snapshot", "state": job.state,
                        "progress": job.progress, "status": job.status_line,
                        "log": job.log[-50:],
                        "created": job.created, "started": job.started,
                        "finished": job.finished, "now": time.time()}
            yield f"data: {json.dumps(snapshot)}\n\n"
            if job.state in ("done", "failed", "cancelled"):
                yield f"data: {json.dumps({'type': 'state', 'state': job.state, 'error': job.error})}\n\n"
                store.unsubscribe(job_id, sub)
                return
            try:
                while True:
                    try:
                        event = await asyncio.get_event_loop().run_in_executor(
                            None, sub.get, True, 20.0)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "state" and event["state"] in ("done", "failed", "cancelled"):
                        break
            finally:
                store.unsubscribe(job_id, sub)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ---- downloads + viewer payloads ------------------------------------- #
    def _first_output(job: Job) -> Path | None:
        for name in job.outputs:
            p = store.job_dir(job.id) / name
            if p.is_file():
                return p
        return None

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, name: str | None = None) -> FileResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        if name:
            # Guard against path traversal: only basenames from this job's dir.
            target = store.job_dir(job_id) / Path(name).name
        else:
            target = _first_output(job)
        if not target or not target.is_file():
            raise HTTPException(status_code=404, detail="No output file.")
        return FileResponse(target, filename=target.name,
                            media_type="application/step")

    @app.get("/api/jobs/{job_id}/input")
    def download_input(job_id: str) -> FileResponse:
        """The original uploaded STL (so history entries stay re-downloadable)."""
        job = store.get(job_id)
        src = store.input_path(job_id)
        if job is None or not src.is_file():
            raise HTTPException(status_code=404, detail="No such job.")
        return FileResponse(src, filename=job.filename, media_type="model/stl")

    @app.get("/api/jobs/{job_id}/mesh/stl")
    def mesh_stl(job_id: str) -> Response:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        from .meshdata import mesh_blob

        blob = mesh_blob(store.input_path(job_id))
        return Response(blob, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/mesh/step")
    def mesh_step(job_id: str) -> Response:
        job = store.get(job_id)
        if job is None or job.state != DONE:
            raise HTTPException(status_code=404, detail="No converted STEP.")
        step = _first_output(job)
        if step is None:
            raise HTTPException(status_code=404, detail="No output file.")
        mesh_file = _ensure_step_mesh(job, step)
        from .meshdata import mesh_blob

        try:
            blob = mesh_blob(mesh_file)
        except Exception as exc:  # noqa: BLE001 - surface a clear reason to the UI
            raise HTTPException(status_code=500,
                                detail=f"STEP mesh failed: {exc}") from exc
        return Response(blob, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/mesh/steptypes")
    def mesh_steptypes(job_id: str) -> Response:
        """STEP tessellation coloured by surface provenance (analytic planes,
        cylinders/cones, spheres, freeform, residual RTAF strips). Cached in
        the job dir; the legend rides in the ``X-Face-Types`` header."""
        job = store.get(job_id)
        if job is None or job.state != DONE:
            raise HTTPException(status_code=404, detail="No converted STEP.")
        step = _first_output(job)
        if step is None:
            raise HTTPException(status_code=404, detail="No output file.")
        blob_file = store.job_dir(job_id) / (step.stem + "_types.m2sm")
        meta_file = store.job_dir(job_id) / (step.stem + "_types.json")
        if not (blob_file.is_file() and meta_file.is_file()):
            if not cfg.freecad_python:
                raise HTTPException(status_code=503,
                                    detail="FreeCAD not available on the server; "
                                           "cannot analyse the STEP's faces.")
            try:
                tessellate_typed(step, blob_file, meta_file, cfg.freecad_python,
                                 deflection=cfg.deflection)
            except Exception as exc:  # noqa: BLE001 - clear reason, not a bare 500
                raise HTTPException(
                    status_code=500,
                    detail=f"Surface-type analysis failed: {exc}") from exc
        try:
            meta = meta_file.read_text(encoding="utf-8")
            blob = blob_file.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=500,
                                detail=f"Surface-type payload unreadable: {exc}") from exc
        return Response(blob, media_type="application/octet-stream",
                        headers={"X-Face-Types": meta})

    @app.get("/api/jobs/{job_id}/mesh/heatmap")
    def mesh_heatmap(job_id: str) -> Response:
        job = store.get(job_id)
        if job is None or job.state != DONE:
            raise HTTPException(status_code=404, detail="No converted STEP.")
        step = _first_output(job)
        if step is None:
            raise HTTPException(status_code=404, detail="No output file.")
        mesh_file = _ensure_step_mesh(job, step)
        from .meshdata import deviation_payload

        # Deviation is pure numpy but can still blow up on a degenerate/empty
        # tessellation — return a clear detail the UI shows in the viewer rather
        # than a bare 500 traceback (silent blank on the deployed app).
        try:
            blob, stats = deviation_payload(store.input_path(job_id), mesh_file)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Deviation computation failed: {exc}") from exc
        return Response(blob, media_type="application/octet-stream",
                        headers={"X-Deviation-Stats": json.dumps(stats)})

    def _ensure_step_mesh(job: Job, step: Path) -> Path:
        """Tessellate the STEP to an STL once, caching it in the job dir."""
        mesh_file = store.job_dir(job.id) / (step.stem + "_tess.stl")
        if not mesh_file.is_file():
            if not cfg.freecad_python:
                raise HTTPException(status_code=503,
                                    detail="FreeCAD not available on the server; "
                                           "cannot tessellate the STEP for preview.")
            try:
                tessellate_step(step, mesh_file, cfg.freecad_python,
                                deflection=cfg.deflection)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - clear reason, not a bare 500
                raise HTTPException(
                    status_code=500,
                    detail=f"STEP tessellation failed: {exc}") from exc
        return mesh_file

    # ---- corpus ----------------------------------------------------------- #
    @app.get("/api/corpus")
    def corpus() -> dict:
        from .. import failstore

        dest = failstore.resolve_dest(cfg.failures_dir)
        manifest_path = dest / "manifest.json"
        files: dict = {}
        if manifest_path.is_file():
            try:
                files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
            except (OSError, ValueError):
                files = {}
        return {"dest": str(dest), "files": list(files.values())}

    # ---- static UI -------------------------------------------------------- #
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app
