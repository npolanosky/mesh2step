"""Tests for worker cancellation: no orphaned child processes on SIGTERM.

The worker makes itself a process-group leader and traps SIGTERM/SIGINT to fan
the signal out to its children (the pymeshlab decimation runner), so a single
kill reaps the whole tree. These tests drive that mechanism deterministically
with a stand-in long-lived child (``sleep``) instead of a real conversion, so
they need neither FreeCAD nor pymeshlab and run in a couple of seconds.

POSIX-only: process groups / killpg are the mechanism under test. Skipped on
Windows, where the launcher uses a Job object to the same end.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process-group cancellation is POSIX-only")


def _src_dir() -> str:
    # tests/ -> repo/ ; src is repo/src
    return str(Path(__file__).resolve().parents[1] / "src")


# A stand-in "worker": installs the real cancellation handler, spawns a
# long-lived child (like the decimation runner), records both PIDs, then blocks.
_PROBE = textwrap.dedent(
    """
    import os, sys, time, subprocess
    sys.path.insert(0, {src!r})
    from mesh2step.worker import _install_cancellation_handler
    _install_cancellation_handler()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open({pidfile!r}, "w") as fh:
        fh.write("%d %d %d" % (os.getpid(), child.pid, os.getpgrp()))
    sys.stdout.write("READY\\n"); sys.stdout.flush()
    time.sleep(60)
    """
)


def _spawn_probe(tmp_path: Path):
    pidfile = tmp_path / "pids.txt"
    code = _PROBE.format(src=_src_dir(), pidfile=str(pidfile))
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Wait for READY (handler installed, child spawned, pidfile written).
    deadline = time.time() + 10
    while time.time() < deadline:
        if pidfile.exists() and proc.stdout is not None:
            line = proc.stdout.readline()
            if "READY" in line:
                break
        time.sleep(0.05)
    assert pidfile.exists(), "probe never became ready"
    worker_pid, child_pid, pgid = (int(x) for x in pidfile.read_text().split())
    return proc, worker_pid, child_pid, pgid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_worker_is_its_own_process_group_leader(tmp_path):
    """The worker detaches into its own group so a launcher can killpg the tree."""
    proc, worker_pid, child_pid, pgid = _spawn_probe(tmp_path)
    try:
        assert pgid == worker_pid           # worker leads its own group
        assert os.getpgid(child_pid) == pgid  # child inherits it
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait(timeout=5)


def test_sigterm_to_worker_pid_reaps_child(tmp_path):
    """Killing only the worker PID still reaps its child (the handler fans out)."""
    proc, worker_pid, child_pid, _pgid = _spawn_probe(tmp_path)
    assert _alive(child_pid)
    os.kill(worker_pid, signal.SIGTERM)   # worker PID only — NOT the group
    # Reap the worker via its Popen handle (os.kill(pid,0) would still see a
    # not-yet-waited zombie as "alive"); its exit proves it terminated.
    rc = proc.wait(timeout=5)
    assert rc == 143  # 128 + SIGTERM, the worker's cancelled-exit code
    assert _wait_dead(child_pid, 5.0), "child was orphaned (still alive after 5s)"


def test_killpg_reaps_whole_tree(tmp_path):
    """Signalling the group (the recommended launcher path) reaps everything."""
    proc, worker_pid, child_pid, pgid = _spawn_probe(tmp_path)
    os.killpg(pgid, signal.SIGTERM)
    proc.wait(timeout=5)
    assert _wait_dead(child_pid, 5.0)


def test_terminate_worker_helper_reaps_tree(tmp_path):
    """The launcher-side terminate_worker() helper reaps the worker + child."""
    from mesh2step.worker import terminate_worker

    proc, worker_pid, child_pid, _pgid = _spawn_probe(tmp_path)
    assert _alive(child_pid)
    ok = terminate_worker(proc, timeout=5.0)
    assert ok is True
    assert proc.poll() is not None        # worker reaped by the helper
    assert _wait_dead(child_pid, 5.0)


def test_cancelled_marker_is_emitted(tmp_path):
    """The worker prints the CANCELLED marker so the launcher can classify it."""
    from mesh2step.worker import CANCELLED_MARKER

    proc, worker_pid, _child_pid, _pgid = _spawn_probe(tmp_path)
    os.kill(worker_pid, signal.SIGTERM)
    out = proc.stdout.read() if proc.stdout is not None else ""
    proc.wait(timeout=5)
    assert CANCELLED_MARKER in out
