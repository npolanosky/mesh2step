"""Job model + queue for the web app.

One :class:`Job` per uploaded mesh. Jobs live on disk under
``<data_dir>/jobs/<id>/`` (the STL, the STEP output(s), a ``job.json`` record)
so history survives a restart. A :class:`JobStore` owns an in-memory index plus
a small thread pool of workers (``concurrency``, default 1) that pull queued
jobs and run the conversion. Progress/log lines and state transitions are
published to per-job subscriber queues, which the SSE endpoint drains.

Threading model: conversions are blocking subprocess calls, so we run them on
worker *threads* (not the event loop). The FastAPI handlers only ever touch the
thread-safe store methods; the SSE endpoint bridges the sync subscriber queue to
async with ``run_in_threadpool``-style polling.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Job lifecycle states.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# Map a progress-message substring to a percentage, so the browser can show a
# determinate bar. Mirrors gui._MILESTONES.
_MILESTONES = [
    ("Locating FreeCAD", 4), ("Preparing mesh", 8), ("Loading", 12),
    ("Scaling", 16), ("Detecting cylinders", 28), ("Found", 34),
    ("countersink", 38), ("Segmenting", 48), ("Building", 62),
    ("Gap-filling", 68), ("local patch", 70), ("merging large patch", 72),
    ("gap patches merged", 76), ("Sewing", 82), ("sewShape", 86),
    ("watertight faceted solid", 88), ("faceted solid", 88),
    ("Exporting", 94), ("Done", 100),
]


def _progress_pct(msg: str) -> int | None:
    for key, pct in _MILESTONES:
        if key in msg:
            return pct
    return None


@dataclass
class Job:
    id: str
    filename: str                 # original upload name
    options: dict                 # conversion config from the UI
    state: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: int = 0
    status_line: str = "Queued"
    log: list[str] = field(default_factory=list)
    result: dict | None = None    # worker result (stats etc.)
    outputs: list[str] = field(default_factory=list)  # basenames written
    error: str | None = None
    corpus_action: dict | None = None  # failstore.record_result summary, if any

    def public(self) -> dict:
        """A JSON-safe view for the API (drops nothing sensitive; log truncated
        by the caller when listing)."""
        d = asdict(self)
        d["elapsed"] = round((self.finished or time.time()) - (self.started or self.created), 2) \
            if self.started else 0.0
        return d


class JobStore:
    """Thread-safe registry + queue of conversions.

    The store is created with a ``runner`` callable that performs the actual
    conversion for one job (injected so tests can stub FreeCAD out). The runner
    receives ``(job, emit)`` where ``emit(kind, payload)`` publishes events.
    """

    def __init__(self, jobs_dir: Path, *, concurrency: int,
                 runner: Callable[[Job, Callable[[str, Any], None]], None]):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._load_existing()
        self._threads = [
            threading.Thread(target=self._work_loop, daemon=True, name=f"m2s-worker-{i}")
            for i in range(max(1, concurrency))
        ]
        for t in self._threads:
            t.start()

    # ---- persistence ------------------------------------------------------ #
    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def _persist(self, job: Job) -> None:
        try:
            self._record_path(job.id).write_text(
                json.dumps(job.public(), indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_existing(self) -> None:
        """Reload finished/failed jobs from disk on startup (history)."""
        if not self.jobs_dir.is_dir():
            return
        for rec in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                d = json.loads(rec.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # A job left mid-run by a crash is marked failed on reload — we can't
            # resume a subprocess we no longer own.
            if d.get("state") in (QUEUED, RUNNING):
                d["state"] = FAILED
                d["error"] = d.get("error") or "interrupted (server restarted)"
            job = Job(
                id=d["id"], filename=d.get("filename", "input.stl"),
                options=d.get("options", {}), state=d.get("state", DONE),
                created=d.get("created", time.time()), started=d.get("started"),
                finished=d.get("finished"), progress=d.get("progress", 0),
                status_line=d.get("status_line", ""), log=d.get("log", []),
                result=d.get("result"), outputs=d.get("outputs", []),
                error=d.get("error"), corpus_action=d.get("corpus_action"),
            )
            self._jobs[job.id] = job

    # ---- creation --------------------------------------------------------- #
    def create(self, filename: str, options: dict, stl_bytes: bytes) -> Job:
        job_id = uuid.uuid4().hex[:12]
        # Keep the ORIGINAL basename on disk (sanitised to a bare name) so
        # everything downstream — outputs, the failure corpus manifest — carries
        # the user's filename instead of a generic "input.stl".
        safe = Path(filename).name or "input.stl"
        if not safe.lower().endswith(".stl"):
            safe += ".stl"
        job = Job(id=job_id, filename=safe, options=options)
        jd = self.job_dir(job_id)
        jd.mkdir(parents=True, exist_ok=True)
        (jd / safe).write_bytes(stl_bytes)
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        self._queue.put(job_id)
        return job

    # ---- accessors -------------------------------------------------------- #
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def input_path(self, job_id: str) -> Path:
        job = self.get(job_id)
        if job is not None:
            named = self.job_dir(job_id) / job.filename
            if named.is_file():
                return named
        # Jobs created before uploads kept their original name.
        return self.job_dir(job_id) / "input.stl"

    def requeue(self, job_id: str) -> Job | None:
        """Re-run an existing job's input with its stored options (new job)."""
        old = self.get(job_id)
        if old is None:
            return None
        src = self.input_path(job_id)
        if not src.is_file():
            return None
        return self.create(old.filename, dict(old.options), src.read_bytes())

    # ---- pub/sub ---------------------------------------------------------- #
    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs and q in subs:
                subs.remove(q)

    def _publish(self, job_id: str, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        for q in subs:
            q.put(event)

    # ---- the worker loop -------------------------------------------------- #
    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        job.state = RUNNING
        job.started = time.time()
        job.status_line = "Starting…"
        self._persist(job)
        self._publish(job.id, {"type": "state", "state": RUNNING})

        def emit(kind: str, payload: Any) -> None:
            if kind == "log":
                line = str(payload)
                job.log.append(line)
                if line.startswith("PROGRESS:"):
                    msg = line[len("PROGRESS:"):].strip()
                    job.status_line = msg
                    pct = _progress_pct(msg)
                    if pct is not None:
                        job.progress = max(job.progress, pct)
                    self._publish(job.id, {"type": "progress",
                                           "message": msg, "progress": job.progress})
                else:
                    self._publish(job.id, {"type": "log", "message": line})
            elif kind == "output":
                job.outputs = list(payload)
            elif kind == "corpus":
                job.corpus_action = payload

        try:
            self._runner(job, emit)
            job.state = DONE
            job.progress = 100
            job.status_line = "Done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            job.state = FAILED
            job.error = str(exc)
            job.status_line = f"Failed: {exc}"
        finally:
            job.finished = time.time()
            self._persist(job)
            self._publish(job.id, {"type": "state", "state": job.state,
                                   "error": job.error})
