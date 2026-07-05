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
CANCELLED = "cancelled"

# States a job can be cancelled from.
_CANCELLABLE = (QUEUED, RUNNING)
# Terminal states (no further transitions, no live worker).
_TERMINAL = (DONE, FAILED, CANCELLED)

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
        try:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Unwritable data dir (e.g. Docker named volume owned by another
            # uid). Don't kill startup: the app's write probe reports it and
            # the write endpoints answer 503 with the fix; reads still work.
            pass
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._subscribers: dict[str, list[queue.Queue]] = {}
        # Live worker subprocesses, keyed by job id, so cancel can kill the tree.
        self._procs: dict[str, Any] = {}
        # Job ids cancelled while queued (or mid-flight) — checked by the worker
        # loop so a dequeued-but-cancelled job never starts, and a running job
        # that raced a cancel is marked cancelled rather than failed.
        self._cancelled: set[str] = set()
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

    def active_count(self) -> dict:
        """Count in-progress jobs (running + queued) for the main-page badge."""
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.state == RUNNING)
            queued = sum(1 for j in self._jobs.values() if j.state == QUEUED)
        return {"running": running, "queued": queued, "active": running + queued}

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

    # ---- cancellation ----------------------------------------------------- #
    def cancel(self, job_id: str) -> Job | None:
        """Cancel a queued or running job.

        Queued → mark cancelled so the worker loop skips it when dequeued.
        Running → hard-kill the worker process tree (killpg) so the FreeCAD /
        meshprep child dies too, then mark cancelled. Returns the job, or None if
        it doesn't exist / isn't in a cancellable state.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state not in _CANCELLABLE:
                return None
            self._cancelled.add(job_id)
            proc = self._procs.get(job_id)
            was_running = job.state == RUNNING
            if not was_running:
                # Queued: finalize immediately; it will be skipped on dequeue.
                job.state = CANCELLED
                job.finished = time.time()
                job.status_line = "Cancelled"
                job.error = "cancelled before start"

        # Kill outside the lock (teardown may block briefly).
        if proc is not None:
            from .conversion import kill_process_tree

            kill_process_tree(proc)

        if not was_running:
            self._persist(job)
            self._publish(job_id, {"type": "state", "state": CANCELLED,
                                   "error": job.error})
        # Running: the worker thread finalizes the state transition (it observes
        # the killed subprocess and the _cancelled marker) so we don't race it.
        return job

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    # ---- the worker loop -------------------------------------------------- #
    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            # Skip a job cancelled while it sat in the queue.
            if self._is_cancelled(job_id):
                with self._lock:
                    self._cancelled.discard(job_id)
                continue
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        job.state = RUNNING
        job.started = time.time()
        job.status_line = "Starting…"
        self._persist(job)
        # ``started`` rides along so a client already subscribed (e.g. watching
        # a queued job) can base its elapsed timer on the true start time.
        self._publish(job.id, {"type": "state", "state": RUNNING,
                               "started": job.started})

        def emit(kind: str, payload: Any) -> None:
            if kind == "proc":
                # The conversion runner hands us the live subprocess so cancel
                # can kill its tree.
                with self._lock:
                    self._procs[job.id] = payload
                return
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

        from .conversion import CancelledError

        try:
            self._runner(job, emit)
            job.state = DONE
            job.progress = 100
            job.status_line = "Done"
        except CancelledError:
            job.state = CANCELLED
            job.error = "cancelled"
            job.status_line = "Cancelled"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            # A cancel that killed the worker may surface as a generic error
            # (e.g. WorkerError) rather than CancelledError; if this job was
            # marked cancelled, honour that over "failed".
            if self._is_cancelled(job.id):
                job.state = CANCELLED
                job.error = "cancelled"
                job.status_line = "Cancelled"
            else:
                job.state = FAILED
                job.error = str(exc)
                job.status_line = f"Failed: {exc}"
        finally:
            with self._lock:
                self._procs.pop(job.id, None)
                self._cancelled.discard(job.id)
            job.finished = time.time()
            self._persist(job)
            self._publish(job.id, {"type": "state", "state": job.state,
                                   "error": job.error})
