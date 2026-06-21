.PHONY: lock sync test run build build-arm64

lock:          ## regenerate uv.lock from pyproject
	uv lock

sync:          ## install deps (incl. dev) into .venv
	uv sync --dev

test:          ## run the test suite
	uv run pytest

run:           ## run the agent locally (needs ANTHROPIC_API_KEY)
	uv run aerys-v2

build:         ## build the container image (native arch)
	docker build -t aerys-v2 .

build-arm64:   ## cross-build for the Jetson (arm64)
	docker buildx build --platform linux/arm64 -t aerys-v2:arm64 .