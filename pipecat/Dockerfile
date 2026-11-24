# syntax=docker/dockerfile:1.7
#
# HyperNoesis starter voice agent (Pipecat, Python) for Self-Hosted LiveKit.
# Build from the repository root:
#   docker build -f pipecat/Dockerfile -t hypernoesis-livekit-starter-pipecat .
#
# No secret is baked in: provider keys and LiveKit credentials arrive as
# environment variables when the container starts. Runs as uid/gid 10001 and
# works with a read-only root filesystem (`--read-only --tmpfs /tmp`).

ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ARG RUST_IMAGE=rust:1.100-slim-trixie@sha256:6f14b2b4b278ec0d50aa1d69541c304ce0f95e0f6268487aa9924af1c2e9aebe
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d

# --- DeepFilterNet3 noise cancellation: libDF C API (MIT OR Apache-2.0) ---------
# The same pinned build as livekit-agents/Dockerfile. Pure-Rust inference
# (tract) of the DeepFilterNet3 ONNX model: no PyTorch.
FROM ${RUST_IMAGE} AS deepfilternet
ARG DFN_COMMIT=7b14b895fe7fc42cab72b4a4c1e49c982bd6404b
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q \
 && git remote add origin https://github.com/Rikorose/DeepFilterNet.git \
 && git fetch -q --depth 1 origin "${DFN_COMMIT}" \
 && git checkout -q FETCH_HEAD
# The Cargo.lock at that commit is out of date; this one pins the crate
# versions the noise-cancellation tests ran against.
COPY pipecat/deepfilternet/Cargo.lock ./Cargo.lock
RUN cargo build --release --locked -p deep_filter --no-default-features --features capi \
 && install -D -m 0644 target/release/libdf.so /out/libdf.so \
 && install -D -m 0644 models/DeepFilterNet3_onnx.tar.gz /out/DeepFilterNet3_onnx.tar.gz \
 && install -D -m 0644 LICENSE-MIT /out/LICENSE-MIT \
 && install -D -m 0644 LICENSE-APACHE /out/LICENSE-APACHE

# --- uv binary -----------------------------------------------------------------
FROM ${UV_IMAGE} AS uv

# --- Python dependencies ---------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY pipecat/pyproject.toml pipecat/uv.lock pipecat/.python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY pipecat/src ./src
RUN uv sync --locked --no-dev --no-editable

# --- Runtime -----------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
RUN groupadd --system --gid 10001 agent \
 && useradd --system --uid 10001 --gid 10001 --home-dir /tmp --no-create-home \
      --shell /usr/sbin/nologin agent
COPY --from=build /app/.venv /app/.venv
COPY --from=deepfilternet /out/ /opt/deepfilternet/
COPY LICENSE THIRD_PARTY_NOTICES /usr/share/doc/hypernoesis-livekit-starter/

# Everything the process writes goes to /tmp (settings cache, library caches),
# so the root filesystem can be mounted read-only.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HN_CACHE_DIR=/tmp \
    HN_DFN_LIB=/opt/deepfilternet/libdf.so \
    HN_DFN_MODEL=/opt/deepfilternet/DeepFilterNet3_onnx.tar.gz
WORKDIR /app
USER 10001:10001

# LiveKit webhooks (POST /webhook) and worker status: GET / (health, 200 once
# LiveKit answers) and GET /worker (JSON for the console's agent list).
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/', timeout=4)"]

CMD ["python", "-m", "hn_pipecat"]
