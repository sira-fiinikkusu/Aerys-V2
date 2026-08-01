# syntax=docker/dockerfile:1

### Stage 1: builder — install deps + project into a venv with uv
FROM python:3.11-slim AS builder

# bring in the uv binary, pinned to match your local version
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) deps ONLY first — this layer caches unless uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) then source + install the project itself (baked in, not editable)
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev --no-editable


### Stage 2: runtime — clean image, just the venv + a non-root user
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# non-root for least privilege
RUN useradd --create-home --uid 1000 aerys
WORKDIR /app

# copy the finished venv from the builder, owned by the non-root user
COPY --from=builder --chown=aerys:aerys /app/.venv /app/.venv
USER aerys

# healthy iff config loads AND her durable store answers (factory.store_reachable).
#
# The budget is sized from measurement on the deploy box, not from habit.
# `aerys-v2 --health` costs ~4.1s wall when idle — almost entirely Python
# interpreter start plus imports, before any probing happens. Against the old 5s
# timeout that left under a second of headroom, and this box regularly sits at
# load ~3.6 with the extractor and gaps miners running, so a perfectly healthy
# container would have been reported UNHEALTHY under nothing worse than work.
#
# A check that cries wolf gets ignored exactly like one that always says ok, so:
#   timeout 20s      — ~5x the measured cost; a real failure still fails fast
#                      (an unreachable store returns in ~2s via connect_timeout)
#   interval 60s     — each run costs ~4s of CPU; every 30s across five
#                      containers is real load on a small box for no new signal
#   start-period 60s — the server needs ~12s to bind; a slow boot must not bank
#                      failures against a container that is merely starting
HEALTHCHECK --interval=60s --timeout=20s --start-period=60s --retries=3 \
    CMD aerys-v2 --health || exit 1

ENTRYPOINT ["aerys-v2"]