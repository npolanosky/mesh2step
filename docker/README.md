# mesh2step-web in Docker (Portainer home server)

A self-contained container for the mesh2step web UI: **headless FreeCAD + the
FastAPI app**, ready for LAN use behind Portainer. First conversion needs no
downloads — FreeCAD, the app, and the prep deps (`manifold3d`, `pymeshlab`) are
all baked into the image.

> **Security:** the app has **no authentication by design** (trusted LAN). Do
> not publish the port to the internet. For remote access use a VPN
> (WireGuard/Tailscale) or a reverse proxy that adds auth.

---

## What's in the image

| Piece | Choice | Why |
|---|---|---|
| Base | `debian:bookworm-slim` | FreeCAD **and** `python3` from one distro → matching C-ABI (bookworm = Python 3.11, FreeCAD 0.21). |
| FreeCAD | apt `freecad` | Installs modules under `/usr/lib/freecad-python3/lib`; the app's `freecad_env` detection uses `/usr/bin/python3` as the worker interpreter automatically. Smaller than conda-forge / AppImage. |
| App | `pip install ".[web]"` | FastAPI + uvicorn; the web process and the FreeCAD worker are the **same** `/usr/bin/python3`, so one install serves both. |
| Prep deps | provisioned at build time | `manifold3d` (watertight boolean) + `pymeshlab` (decimation) baked into `/opt/mesh2step-state/mesh2step/pydeps/`. |
| User | non-root `uid:gid 10001` | Data volumes are chowned to it. |

**Ports:** the container listens on **8799** and is published on the host at
`${MESH2STEP_PORT:-8799}`.

**Volumes:**

| Volume | Mount | Contents |
|---|---|---|
| `mesh2step-web-data` | `/data/web` | Job history, uploads, STEP outputs (`MESH2STEP_WEB_DIR`). |
| `mesh2step-corpus` | `/data/corpus` | Failure corpus / flagged meshes (`MESH2STEP_WEB_FAILURES_DIR`). |

The prep-dep cache lives in the image (`/opt/mesh2step-state`), **not** on a
volume — it is an image artifact, so an upgraded image ships fresh deps.

---

## Deploy in Portainer

### Option A — build from this Git repo (recommended)

1. Portainer → **Stacks → Add stack**.
2. Name it `mesh2step-web`.
3. **Build method → Git repository.**
   * *Repository URL:* `https://github.com/npolanosky/mesh2step`
   * *Reference:* `refs/heads/main` (or a tag)
   * *Compose path:* `docker/docker-compose.yml`
4. (Optional) **Environment variables** — see the table below.
5. **Deploy the stack.** The first deploy builds the image (FreeCAD apt install +
   prep-dep bake — expect several minutes and a few hundred MB downloaded once).
6. Open `http://<server-ip>:8799`.

The `build:` block in `docker/docker-compose.yml` uses `context: ..` so the
build sees the whole repo (it needs `pyproject.toml`, `README.md`, `LICENSE`,
`src/`). Nothing else to configure.

### Option B — prebuilt image

If you'd rather build once and push to a registry:

```bash
# on a build host (see "Building the image manually" below)
docker build -f docker/Dockerfile -t your-registry.example/mesh2step-web:latest .
docker push your-registry.example/mesh2step-web:latest
```

Then in the stack's compose, comment out the `build:` block and set
`image: your-registry.example/mesh2step-web:latest`. Deploy the stack; Portainer
pulls the image.

### Option C — paste the stackfile

Portainer → **Stacks → Add stack → Web editor**, paste the contents of
`docker/docker-compose.yml`, and use **Option B** (a prebuilt `image:`), since
the web editor has no repo context to build from.

---

## Environment variables (stack-level, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `MESH2STEP_PORT` | `8799` | Host port to publish. |
| `MESH2STEP_CONCURRENCY` | `1` | Conversions run at once. Raise on a beefy box. |
| `MESH2STEP_TIMEOUT` | `1800` | Per-conversion wall-clock ceiling (s). |
| `MESH2STEP_MAX_UPLOAD` | `209715200` | Max upload size (bytes, 200 MB). |

Set these in Portainer's stack **Environment variables** panel. Deeper knobs
(data dir, corpus dir, FreeCAD python, bind host) are baked into the image with
container-correct defaults and normally shouldn't be changed.

---

## Building the image manually

Run from the **repo root** (the build context must be the repo, not `docker/`):

```bash
docker build -f docker/Dockerfile -t mesh2step-web:latest .
```

Or with compose:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

Home servers are usually `linux/amd64`. If you build on Apple Silicon and want
an amd64 image for the server, use buildx:

```bash
docker buildx build --platform linux/amd64 -f docker/Dockerfile \
    -t mesh2step-web:latest --load .
```

---

## Health & logs

* Liveness: `GET /healthz` → `{"status":"ok"}` (what the compose healthcheck
  hits). Portainer shows the container **healthy/unhealthy** from it.
* Readiness detail: `GET /api/health` → version, `freecad_ready`, concurrency.
  If `freecad_ready` is `false`, FreeCAD wasn't detected (shouldn't happen in
  this image — `MESH2STEP_FREECAD_PYTHON=/usr/bin/python3` is baked in).
* Logs: Portainer → the container → **Logs**, or `docker logs -f mesh2step-web`.

Quick smoke test from the host:

```bash
curl -fsS http://localhost:8799/healthz
curl -s   http://localhost:8799/api/health | python3 -m json.tool
```

---

## Upgrade procedure

Job history and the failure corpus live in named volumes, so upgrades keep them.

**Git-repo stack (Option A):**

1. Push/pull the new code (or move the stack's Git reference to a new tag).
2. Portainer → the stack → **Pull and redeploy** (tick *Re-pull image and
   rebuild* / *Re-fetch repository*). This rebuilds the image, giving you fresh
   FreeCAD + prep deps, and recreates the container against the same volumes.

**Prebuilt-image stack (Option B):**

```bash
docker build -f docker/Dockerfile -t your-registry.example/mesh2step-web:latest .
docker push your-registry.example/mesh2step-web:latest
```

Then Portainer → the stack → **Update / Pull and redeploy**.

**Fresh start (wipe history):** delete the `mesh2step-web-data` volume. Delete
`mesh2step-corpus` to also clear the failure corpus. Both are safe to remove
while the stack is down; they're recreated empty on next start.

---

## Troubleshooting

* **Build is slow / large:** the FreeCAD apt package pulls OCCT and a lot of
  libs; the first build downloads a few hundred MB. Subsequent builds reuse
  layers unless `pyproject.toml`/`src/` changed.
* **`freecad_ready: false` in `/api/health`:** exec into the container and check
  `python3 -c "import sys; sys.path.insert(0,'/usr/lib/freecad-python3/lib'); import FreeCAD; print(FreeCAD.Version())"`.
  If FreeCAD's lib dir moved (a newer Debian), set `MESH2STEP_FREECAD_PYTHON`
  explicitly and add the lib dir — but on bookworm the baked default is correct.
* **Prep deps missing (`manifold3d` not importable):** the watertight boolean
  path degrades but conversions still run. Rebuild with network access so the
  build-time bake step can fetch the wheels.
* **Permission errors on the volume:** the container runs as `uid:gid 10001`;
  named volumes are chowned automatically. If you bind-mount a host directory
  instead, `chown 10001:10001` it on the host first.
