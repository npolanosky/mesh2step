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

| Mount source | Container path | Kind | Contents |
|---|---|---|---|
| `mesh2step-web-data` | `/data/web` | named volume | Job history, uploads, STEP outputs (`MESH2STEP_WEB_DIR`). |
| `${MESH2STEP_CORPUS_DIR:-./mesh2step-corpus}` | `/data/corpus` | **host bind mount** | Failure corpus / flagged meshes (`MESH2STEP_WEB_FAILURES_DIR`). |

The prep-dep cache lives in the image (`/opt/mesh2step-state`), **not** on a
volume — it is an image artifact, so an upgraded image ships fresh deps.

### The failure corpus is a host bind mount (for Syncthing)

Job history stays in an opaque named Docker volume, but the **failure corpus**
is bind-mounted from a **host directory** so the flagged/failed meshes land on
the host filesystem where they can be synced back to a dev machine (e.g. with
**Syncthing** — point Syncthing at this directory).

* **Default:** `./mesh2step-corpus`, i.e. a `mesh2step-corpus/` directory
  **next to the compose file** (created on first run if missing).
* **Override:** set `MESH2STEP_CORPUS_DIR` to any absolute (or relative) host
  path — e.g. `MESH2STEP_CORPUS_DIR=/srv/syncthing/mesh2step-corpus`.

**Permissions (important):** the container runs as **uid:gid 10001**. Unlike a
named volume, a bind-mounted host directory is **not** auto-chowned, so the
container can't write to it unless 10001 can. Do **one** of:

```bash
# 1) chown the host dir to the container's uid (simplest):
mkdir -p ./mesh2step-corpus && sudo chown -R 10001:10001 ./mesh2step-corpus
```

or override the container user to match your host user in the compose service
(add under `mesh2step-web:`), so files are written as *you* and stay easy to
sync/edit:

```yaml
    # 2) run the container as your host uid:gid instead of the baked 10001:
    user: "1000:1000"   # <-- replace with your `id -u`:`id -g`
```

Symptom of getting this wrong: conversions run but flagged/failed meshes never
appear in the host dir, and the container log shows permission-denied writing to
`/data/corpus`.

### Pitfall: changing `user:` with an EXISTING named volume

The `mesh2step-web-data` **named volume** is chowned to the *first* uid that
used it. If you later add (or change) a `user:` override — e.g. switch to
`user: "1000:1000"` after the volume was created by the default uid 10001 —
the volume keeps its **old** ownership and every upload fails with
`PermissionError: ... mkdir '/data/web/jobs'` (the app returns **503** on
convert, logs a boxed `DATA DIRECTORY NOT WRITABLE` message at startup, shows
a red banner in the UI, and reports `data_writable: false` in `/api/health`).

Fix the volume's ownership **once** to match your `user:` line, then restart:

```bash
docker run --rm -v mesh2step-web-data:/data alpine chown -R 1000:1000 /data
```

(Replace `1000:1000` with the uid:gid from your `user:` override. The same
one-liner works for any named volume; the corpus **bind mount** is a plain
host directory — `sudo chown` it directly instead.)

---

## Deploy in Portainer

The repo (and the `mesh2step-web` package on GHCR) are **private**. Every push
to `main` that touches `src/mesh2step/**`, `docker/**`, or `pyproject.toml`
triggers `.github/workflows/docker.yml`, which builds a multi-arch
(`linux/amd64` + `linux/arm64`) image and pushes it to
`ghcr.io/npolanosky/mesh2step-web` tagged `latest` + the short commit SHA (and
semver tags on version-tag pushes). `docker/docker-compose.yml` already points
at that image by default, so Portainer just needs pull credentials.

### One-time: give Portainer a GHCR pull credential

GHCR requires auth to pull a private image even with `docker.io`-style
anonymous pulls disabled, so set this up once:

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens
   (classic)** → **Generate new token**.
   * Scope: **`read:packages`** only.
   * Expiration: your call (rotate it when it expires).
2. Portainer → **Registries → Add registry**.
   * *Registry type:* **GitHub Container Registry** (or "Custom registry" with
     URL `ghcr.io` if your Portainer version doesn't have a GHCR preset).
   * *Username:* `npolanosky`.
   * *Password:* the PAT you just created.
3. Save. Portainer can now pull `ghcr.io/npolanosky/mesh2step-web` images.

### Deploy the stack

1. Portainer → **Stacks → Add stack**.
2. Name it `mesh2step-web`.
3. Either:
   * **Repository** build method pointed at
     `https://github.com/npolanosky/mesh2step` (Portainer needs a GitHub PAT
     with `repo` read access configured under **Settings → Git** for private
     repos), compose path `docker/docker-compose.yml`; or
   * **Web editor**, paste the contents of `docker/docker-compose.yml`
     directly (simplest — no repo credentials needed for the stack file
     itself, only the registry credential from above for the image pull).
4. (Optional) **Environment variables** — see the table below.
5. **Deploy the stack.** Portainer pulls
   `ghcr.io/npolanosky/mesh2step-web:latest` using the registry credential and
   starts the container.
6. Open `http://<server-ip>:8799`.

### Building locally instead (optional)

If you'd rather build on the server itself instead of pulling from GHCR,
edit `docker/docker-compose.yml`: comment out `image:` and uncomment the
`build:` block (`context: ..`, `dockerfile: docker/Dockerfile`), then deploy
with the **Repository** build method so Portainer has the full repo context.

---

## Environment variables (stack-level, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `MESH2STEP_PORT` | `8799` | Host port to publish. |
| `MESH2STEP_CORPUS_DIR` | `./mesh2step-corpus` | **Host directory** bind-mounted to `/data/corpus` for the failure corpus (Syncthing this back to your dev machine). Must be writable by uid 10001 — see the permissions note above. |
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

Job history lives in a named volume and the failure corpus in a host bind mount,
so upgrades keep both.

**Default (prebuilt GHCR image):**

1. Merge/push to `main` (touching `src/mesh2step/**`, `docker/**`, or
   `pyproject.toml`) — this triggers the `docker.yml` workflow, which builds
   and pushes a fresh `ghcr.io/npolanosky/mesh2step-web:latest`. You can also
   trigger it manually from the Actions tab (`workflow_dispatch`) or watch it
   with `gh run watch`.
2. Portainer → the stack → **Pull and redeploy** (tick *Re-pull image*). This
   pulls the new `latest` from GHCR and recreates the container against the
   same volumes.

**Building locally instead:**

1. Push/pull the new code.
2. Portainer → the stack → **Pull and redeploy** (tick *Re-pull image and
   rebuild* / *Re-fetch repository*). This rebuilds the image, giving you fresh
   FreeCAD + prep deps, and recreates the container against the same volumes.

**Fresh start (wipe history):** delete the `mesh2step-web-data` volume to clear
job history. To clear the failure corpus, empty the **host bind-mount
directory** (`./mesh2step-corpus` or your `MESH2STEP_CORPUS_DIR`) instead — it's
a plain host folder now, not a Docker volume. Both are safe to clear while the
stack is down; they're recreated empty on next start.

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
* **Permission errors on the volume:** the container runs as `uid:gid 10001`.
  The `mesh2step-web-data` **named** volume is chowned automatically. The
  **corpus bind mount** is a host directory and is **not** — `sudo chown -R
  10001:10001 ./mesh2step-corpus` (or add a `user: "<uid>:<gid>"` override to the
  service). See "The failure corpus is a host bind mount" above.
