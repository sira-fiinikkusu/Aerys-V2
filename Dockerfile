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

# healthy iff config loads cleanly (proves the key is present)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD aerys-v2 --health || exit 1

ENTRYPOINT ["aerys-v2"]