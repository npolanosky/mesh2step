# mesh2step-web — local web app

A browser UI for the STL → STEP converter, designed for a small home server
(LAN use, no authentication). The web process is a plain FastAPI app; every
conversion runs out-of-process under **FreeCAD's Python** exactly like the
desktop GUI does (`mesh2step.worker` subprocess with streamed progress), so the
server process itself never imports FreeCAD.

```
pip install ".[web]"
mesh2step-web                       # → http://127.0.0.1:8765
```

Features:

* drag-and-drop STL upload with the same options as the desktop GUI
  (source units, detect cylinders, repair mesh, fully-closed, faceted-only,
  save-failures toggle),
* live progress bar + log streamed over Server-Sent Events,
* quality verdict, watertight/RTAF/face-count badges, STEP download,
* a real three.js viewer (orbit/pan/zoom) with **Input STL / STEP / Deviation
  heatmap** tabs and shaded/edges/wireframe modes — the deviation is computed
  server-side (per-vertex distance from the tessellated STEP to the input mesh)
  and shipped as vertex colours,
* "Flag for improvement" (failure-corpus `faceted_improvable`, same as the GUI),
* a Corpus & history page: every conversion with a Re-run button, plus the
  failure-corpus manifest.

The UI is plain HTML/CSS/JS with a vendored `three.min.js` — no npm, no build
step, works fully offline.

---

## Configuration

Everything can be set by CLI flag or environment variable (flags win):

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host` | `MESH2STEP_WEB_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` to serve the LAN. |
| `--port` | `MESH2STEP_WEB_PORT` | `8765` | HTTP port. |
| `--data-dir` | `MESH2STEP_WEB_DIR` | `~/.mesh2step-web` | Working dir; uploads/results live in `jobs/<id>/`. |
| `--concurrency` | `MESH2STEP_WEB_CONCURRENCY` | `1` | Conversions run at once (queued beyond that). |
| `--freecad-python` | `MESH2STEP_FREECAD_PYTHON` | auto-detect | Path to FreeCAD's Python interpreter. |
| — | `MESH2STEP_WEB_DEFLECTION` | `0.1` | STEP tessellation deflection (mm) for the viewer/heatmap. |
| — | `MESH2STEP_WEB_TIMEOUT` | `1800` | Per-conversion wall-clock ceiling (s). |
| — | `MESH2STEP_WEB_MAX_UPLOAD` | `209715200` | Max upload size (bytes). |
| — | `MESH2STEP_WEB_FAILURES_DIR` | failstore default | Failure-corpus destination. |

Job history survives restarts (each job keeps a `job.json` record on disk).
Nothing is auto-deleted; clear `~/.mesh2step-web/jobs` yourself if it grows.

---

## Debian / Ubuntu install

> **Honesty note:** the web app was developed and verified end-to-end on macOS
> (FreeCAD 1.x app bundle). The Linux paths below follow the documented layouts
> of the apt package and the official AppImage, and the platform branching in
> `freecad_env.py` / `provision.py` is additive — but none of it has been
> executed on a real Linux box yet. Expect to use `MESH2STEP_FREECAD_PYTHON`
> as the escape hatch if auto-detection misses.

### 1. Python + mesh2step

```bash
sudo apt install python3 python3-pip python3-venv
python3 -m venv ~/mesh2step-venv
~/mesh2step-venv/bin/pip install "mesh2step[web] @ git+https://github.com/npolanosky/mesh2step"
# or from a checkout:  ~/mesh2step-venv/bin/pip install ".[web]"
```

### 2. FreeCAD

Two supported routes:

**a) apt (simplest):**

```bash
sudo apt install freecad
```

The apt package builds FreeCAD against the **system python3** and installs the
Python modules under `/usr/lib/freecad-python3/lib` (older releases:
`/usr/lib/freecad/lib`). `find_freecad_python()` detects this layout and uses
`/usr/bin/python3` as the worker interpreter with that lib dir injected on
`PYTHONPATH`. Caveat: Debian stable can ship an old FreeCAD (0.20/0.21 era);
the converter supports 0.20+, but newer is better. *(untested)*

**b) AppImage (newest FreeCAD, no root):**

```bash
cd ~ && wget https://github.com/FreeCAD/FreeCAD/releases/download/<ver>/FreeCAD_<ver>-Linux-x86_64.AppImage
chmod +x FreeCAD_*.AppImage
./FreeCAD_*.AppImage --appimage-extract     # creates ~/squashfs-root/
```

The extracted AppImage bundles its own Python at
`~/squashfs-root/usr/bin/python`, which detection prefers (it is the ideal
worker interpreter — self-contained, current). Running the AppImage *without*
extracting is not supported: the worker needs a plain `python -m` entry point.
*(untested)*

**Manual override (always works):**

```bash
export MESH2STEP_FREECAD_PYTHON=/usr/bin/python3          # apt route
# or
export MESH2STEP_FREECAD_PYTHON=~/squashfs-root/usr/bin/python
```

Sanity-check the worker interpreter can see FreeCAD:

```bash
$MESH2STEP_FREECAD_PYTHON -c "import sys; sys.path.insert(0,'/usr/lib/freecad-python3/lib'); import FreeCAD; print(FreeCAD.Version())"
```

### 3. Prep deps (pymeshlab + manifold3d)

The fully-closed/watertight path wants `manifold3d` (required) and `pymeshlab`
(optional decimation) importable **by the worker interpreter**. The server
self-provisions them on the first conversion: it runs the worker python's `pip`
with `--target` into `~/.local/share/mesh2step/pydeps/<py-tag>/` (XDG data dir
— never into the FreeCAD install) and injects that dir via `PYTHONPATH`.

Notes for Linux *(untested)*:

* apt route: needs `python3-pip` installed (step 1 covers it).
* AppImage route: the bundled python usually has pip; if not, provision
  manually with a matching-ABI pip:
  `python3 -m pip install --target ~/.local/share/mesh2step/pydeps/pyX.Y-x86_64 manifold3d pymeshlab`
  (the tag directory name is printed in the server log on first conversion).
* Provisioning failure is non-fatal — conversions still run, the watertight
  boolean path just degrades (a log line says so).

### 4. Run it

```bash
~/mesh2step-venv/bin/mesh2step-web --host 0.0.0.0 --port 8765
```

Open `http://<server>:8765` from any machine on the LAN.

### 5. systemd unit

`/etc/systemd/system/mesh2step-web.service`:

```ini
[Unit]
Description=mesh2step web UI
After=network.target

[Service]
Type=simple
User=nick
Environment=MESH2STEP_WEB_HOST=0.0.0.0
Environment=MESH2STEP_WEB_PORT=8765
# Uncomment for an apt FreeCAD if auto-detection misses:
#Environment=MESH2STEP_FREECAD_PYTHON=/usr/bin/python3
ExecStart=/home/nick/mesh2step-venv/bin/mesh2step-web
Restart=on-failure
# Conversions are CPU-bound; keep the box responsive:
Nice=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mesh2step-web
journalctl -u mesh2step-web -f
```

**Security note:** there is no authentication by design (trusted LAN). Do not
expose the port to the internet; if you need remote access, put it behind a
VPN (WireGuard/Tailscale) or a reverse proxy with auth.

---

## Architecture notes

* **SSE, not WebSockets** — progress is strictly server→browser; SSE
  (`GET /api/jobs/<id>/events`) auto-reconnects, needs no client library, and
  passes through proxies as a plain HTTP stream.
* **Binary mesh payloads, not glTF** — the viewer data is a flat triangle soup
  with optional per-vertex colours. A tiny custom header (`M2SM`) lets the
  browser wrap `Float32Array` views directly over the HTTP response with zero
  parsing and no loader dependency. See `webapp/meshdata.py`.
* **Deviation heatmap** — the STEP is tessellated once by the worker
  (`tessellate` mode, cached as `<name>_tess.stl` in the job dir); the web
  process then computes exact point-to-triangle distances in blocked, vectorised
  numpy and maps them to a jet ramp. Stats (max/rms/p95/mean and the colour
  clamp) ride along in an `X-Deviation-Stats` response header.
* **Job model** — `~/.mesh2step-web/jobs/<id>/` holds the upload (original
  basename), outputs, tessellation cache and a `job.json` record. A queue with
  `concurrency` worker threads (default 1) serialises conversions; jobs
  submitted meanwhile wait in `queued` state. Jobs interrupted by a server
  restart are marked failed on reload (a subprocess we no longer own can't be
  resumed).

### API surface

| Method + path | Purpose |
|---|---|
| `GET /api/health` | version, FreeCAD status, concurrency |
| `POST /api/convert` | multipart upload (`file`, `options` JSON) → `{id}` |
| `GET /api/jobs` | history (newest first) |
| `GET /api/jobs/{id}` | full job record incl. worker stats |
| `GET /api/jobs/{id}/events` | SSE progress/log stream |
| `GET /api/jobs/{id}/download[?name=]` | the STEP file(s) |
| `GET /api/jobs/{id}/input` | the original uploaded STL |
| `GET /api/jobs/{id}/mesh/stl\|step\|heatmap` | viewer payloads (M2SM binary) |
| `POST /api/jobs/{id}/rerun` | re-queue the same input/options as a new job |
| `POST /api/jobs/{id}/flag` | failure-corpus `faceted_improvable` flag |
| `GET /api/corpus` | failure-corpus manifest |
| `POST /api/settings` | toggle `save_failures` |

## What is tested where

* `tests/test_webapp.py` — FastAPI TestClient suite (12 tests): upload/convert/
  status/download happy path, non-STL rejection, mesh/heatmap payload structure,
  SSE replay, failing-mesh → failstore recording (and the toggle-off case),
  flag-for-improvement, queueing of a second job, re-run, and path-traversal
  guarding. The worker subprocess is faked, so the suite runs without FreeCAD.
* Real end-to-end (macOS, FreeCAD app bundle): verified manually — upload →
  watertight STEP → download → viewer payloads → heatmap, plus a queued second
  conversion. See the repo history / PR notes.
* Linux: **not executed yet** — the sections marked *(untested)* above.
