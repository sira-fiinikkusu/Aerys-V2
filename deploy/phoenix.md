# Phoenix on the Jetson — trace store for the V2 brain (dossier 01-05)

Arize Phoenix receives OpenTelemetry spans from the brain (`src/aerys_v2/tracing.py`)
and gives us the n8n Executions tab equivalent: every model call, graph step, and
(later) tool call — prompts, outputs, latencies, token counts — browsable in a UI.

n8n mapping: in V1 this visibility was free (the engine recorded every node run).
In V2 it's this container + one `OTLP_ENDPOINT` line in the brain's `.env`.

## Run it (on the Jetson, `ssh jetson`)

```bash
# Named volume = trace data survives container recreation (like the NAS Postgres
# surviving n8n container swaps — storage outlives the process).
docker volume create phoenix-data

docker run -d \
  --name phoenix \
  --restart unless-stopped \
  -p 6006:6006 \
  -v phoenix-data:/mnt/data \
  -e PHOENIX_WORKING_DIR=/mnt/data \
  -e PHOENIX_ENABLE_TELEMETRY=false \
  arizephoenix/phoenix:latest
```

Notes on each choice:

- **`arizephoenix/phoenix` is multi-arch** — the arm64 image runs natively on the
  Jetson (5.15 tegra kernel; `docker exec`/`stop` behave normally there, this is
  NOT the Tachyon quirk box).
- **Port 6006** — Phoenix serves BOTH the web UI and the OTLP-over-HTTP ingest on
  this one port (`/v1/traces`). We deliberately do NOT publish 4317 (OTLP gRPC):
  the brain uses the HTTP exporter, and every unpublished port is one less door.
- **`PHOENIX_WORKING_DIR=/mnt/data` + the volume** — Phoenix persists its SQLite
  trace store here; without it, traces vanish on every container recreate.
- **`PHOENIX_ENABLE_TELEMETRY=false`** — disables Phoenix's anonymous usage
  telemetry phone-home. This is a WORK-APPROVED CONDITION, not a preference:
  telemetry stays off.
- **`--restart unless-stopped`** — same posture as the n8n container: comes back
  after Jetson reboots without a watchdog.

Verify:

```bash
docker ps --filter name=phoenix          # Up, port 6006
curl -s http://192.168.1.107:6006 | head -1   # HTML back = UI alive
```

UI: `http://192.168.1.107:6006` from the LAN.

## Point the brain at it

In the brain's `.env` (pydantic maps `OTLP_ENDPOINT` → `Settings.otlp_endpoint`):

```bash
OTLP_ENDPOINT=http://192.168.1.107:6006/v1/traces
```

The path matters — `/v1/traces` is the OTLP/HTTP ingest route, not just the host.
When the brain runs as a container ON the same Jetson, `192.168.1.107` still works
(LAN IP resolves fine from inside the container); `localhost` would not.

Restart the brain (`--serve`). Startup log says `tracing armed | otlp=...` when
live. Unset/wrong endpoint can never block serving — `wire_tracing()` logs and
degrades (the rule in `src/aerys_v2/tracing.py`).

## Security posture (the work-approved conditions)

- **LAN-only.** Port 6006 is bound on the Jetson's LAN interface and must NOT be
  exposed through the Cloudflare tunnel. Nothing outside 192.168.1.0/24 reaches it.
- **Telemetry off** (`PHOENIX_ENABLE_TELEMETRY=false`) — see above; non-negotiable.
- **Treat the trace store AT DATA SENSITIVITY.** Spans contain full prompts and
  replies — which means soul.md content, memory context, and real conversation
  text. The `phoenix-data` volume is as sensitive as the NAS `aerys` database:
  don't ship it off-box, don't screenshot traces into public places, wipe it with
  the same "ask first" data-destruction rule.
- **Auth note:** Phoenix ships with NO authentication by default. On this
  single-operator LAN behind the router that's the accepted posture (same as the
  n8n UI). If that ever changes — guests on the LAN, a tunnel exposure request —
  enable Phoenix's built-in auth first: add
  `-e PHOENIX_ENABLE_AUTH=true -e PHOENIX_SECRET=<long-random-string>` and log in
  with the initial admin account (`admin@localhost` / `admin`, forced password
  change on first login).
