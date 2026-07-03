# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# mesh2step-web — headless FreeCAD + FastAPI, for a Linux home server.
#
# Base image choice: debian:bookworm-slim.
#   * FreeCAD and python3 come from the SAME distro, so FreeCAD.so is built
#     against the exact CPython that runs the web process and the conversion
#     worker (bookworm = Python 3.11, FreeCAD 0.21). No C-ABI drift — which is
#     the trap when you mix python.org's 3.11 with a distro-built FreeCAD.
#   * The apt package installs FreeCAD's Python modules under
#     /usr/lib/freecad-python3/lib. freecad_env._linux_system_python() detects
#     exactly this: it returns /usr/bin/python3 as the worker interpreter once
#     that lib dir exists, and freecad_lib_dir() injects the lib dir on
#     PYTHONPATH. So `<python3> -m mesh2step.worker` gets `import FreeCAD` — the
#     pattern conversion.run_worker uses — with zero extra config.
#   * Auto-detection therefore just works; MESH2STEP_FREECAD_PYTHON is set below
#     anyway as a belt-and-braces override so a future layout change can't break
#     the container silently.
#   * Alternatives considered: conda-forge freecad (much larger image, its own
#     Python — workable but heavier and a second Python to reason about) and the
#     extracted AppImage (newest FreeCAD, but ~1GB of squashfs to carry and a
#     home-relative detection path that fights a non-root container user). apt
#     wins on image size + freecadcmd/python availability + our worker pattern.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

# freecad          — the CAD kernel (pulls OCCT); provides /usr/lib/freecad-python3/lib
# python3/-pip/-venv/-dev — the SYSTEM interpreter the worker + web process share
# libgl1/libglu1-mesa/libxrender1/libxcursor1/libxft2/libxinerama1 — OCCT/Coin3D
#   link against GL/X libs even for headless geometry ops; without them
#   `import FreeCAD` fails on a missing .so. No X server is needed (no GUI).
# ca-certificates  — TLS for the one-time prep-dep wheel download at build time
# curl             — used by the compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        freecad \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        libgl1 \
        libglu1-mesa \
        libxrender1 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Belt-and-braces: name the worker interpreter explicitly so the container never
# depends on auto-detection succeeding. This is the apt route's system python3.
ENV MESH2STEP_FREECAD_PYTHON=/usr/bin/python3

# Where mesh2step keeps its per-"user" state (provision.support_dir reads
# XDG_DATA_HOME). Baked into the image so the provisioned prep deps (pydeps/)
# ship inside the image and the FIRST conversion needs no network. Kept OFF the
# data volume on purpose — the deps are an image artifact, not user data.
ENV XDG_DATA_HOME=/opt/mesh2step-state
# Data volume mount points (see docker-compose.yml).
ENV MESH2STEP_WEB_DIR=/data/web \
    MESH2STEP_WEB_FAILURES_DIR=/data/corpus \
    MESH2STEP_WEB_HOST=0.0.0.0 \
    MESH2STEP_WEB_PORT=8799 \
    PYTHONUNBUFFERED=1

# Debian's pip refuses to install into the system environment without this
# (PEP 668). We install app deps system-wide on purpose: the web process AND the
# FreeCAD worker are the same /usr/bin/python3, so one system install serves both.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_INPUT=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# --- Python deps first (better layer caching) -----------------------------
# Copy only what pip needs to resolve + build the wheel, so edits to source or
# assets don't bust the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install the package with the [web] extras (FastAPI + uvicorn + multipart).
# numpy: the [web]/base deps pull numpy>=1.21. Debian's freecad is built against
# the distro numpy (1.24 line); provision.py installs the prep deps with
# --no-deps precisely so numpy is never duplicated/shadowed. We therefore let
# pip resolve numpy for the app but keep it <2 to stay ABI-compatible with the
# freecad build (matches provision.py's contract that the worker uses one numpy).
RUN pip install --no-cache-dir "numpy<2" \
    && pip install --no-cache-dir ".[web]"

# --- Bake the prep deps (manifold3d [+ pymeshlab]) at build time ----------
# The prep deps must live in provision.pydeps_dir("/usr/bin/python3") =
# $XDG_DATA_HOME/mesh2step/pydeps/<py-tag>/ so prep_deps_present() finds them and
# the FIRST real conversion needs no network.
#
# We do NOT just call provision.ensure_prep_deps() because it installs manifold3d
# (REQUIRED) and pymeshlab (OPTIONAL) in ONE `pip install --no-deps` command — if
# pymeshlab has no wheel for this platform (e.g. only pymeshlab 2025.x publishes
# aarch64 wheels, and provision pins <2025), pip fails the whole command and even
# manifold3d is lost. So we bake them independently, mirroring provision's
# contract (--no-deps into the exact pydeps tag dir; never pull numpy in — FreeCAD
# provides it). manifold3d is REQUIRED and hard-fails the build if it can't land;
# pymeshlab is best-effort (decimation-only; the pipeline degrades without it).
RUN set -eux; \
    PYDEPS="$(python3 -c 'from mesh2step import provision; print(provision.pydeps_dir("/usr/bin/python3"))')"; \
    mkdir -p "$PYDEPS"; \
    pip install --no-cache-dir --no-deps --target "$PYDEPS" "manifold3d>=3.0"; \
    pip install --no-cache-dir --no-deps --target "$PYDEPS" "pymeshlab>=2023.12,<2025" \
        || echo "pymeshlab bake skipped (no wheel for this platform; decimation degrades, conversions still work)"; \
    # provision._purge_shadowing_numpy contract: numpy must never sit in pydeps \
    # (it would shadow FreeCAD's and crash manifold3d via a 1.x/2.x ABI mismatch). \
    rm -rf "$PYDEPS"/numpy "$PYDEPS"/numpy-*.dist-info "$PYDEPS"/numpy.libs "$PYDEPS"/numpy-*.data; \
    python3 -c "import sys; sys.path.insert(0, '$PYDEPS'); import manifold3d; print('baked manifold3d', manifold3d.__version__ if hasattr(manifold3d,\"__version__\") else \"ok\")"

# --- Non-root user + data dirs --------------------------------------------
# uid/gid 10001 so the mounted volume can be chowned predictably on the host.
RUN groupadd --gid 10001 mesh2step \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin mesh2step \
    && mkdir -p /data/web /data/corpus /opt/mesh2step-state \
    && chown -R 10001:10001 /data /opt/mesh2step-state

USER 10001:10001

EXPOSE 8799

# mesh2step-web reads host/port from the MESH2STEP_WEB_* env above (0.0.0.0:8799).
CMD ["mesh2step-web"]
