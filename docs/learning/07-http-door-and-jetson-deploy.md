# LEARNING 07 — the HTTP door + first deploy to Siraheart

*2026-07-02, late. The Brain runs on her own hardware now.*

## The one-sentence version

`transports/http_api.py` gives the Brain an authed HTTP `/ask` (the webhook-trigger
pattern, done once for every future caller), and your 01-01 Dockerfile just deployed
it to the Jetson as the `aerys-brain` container.

## The door (n8n mapping)

| n8n | here |
|---|---|
| A webhook workflow per caller (kael-dm, myo-gesture, …) | ONE FastAPI app; callers differ by token + thread_id |
| Auth Guard Code node checking a header | `require_token` dependency — runs before anything else |
| No token configured = webhook still up | unset token = **503 locked shut**, never an open door |
| Session = which workflow you hit | caller names its `thread_id` (`voice:pe`, `http:default`, …) |

## The deploy — why tonight's demo was actually two machines

The container mounts the REAL `~/aerys/config/soul.md` read-only and points
`DATABASE_URL` at the NAS `aerys_v2` db. First live question through the Jetson
container answered with the number **47 — which you told her from Leviathan, in a
different process, an hour earlier.** Same thread key (`cli`), same NAS checkpointer:
conversation state is now machine-independent. That's what "durable" buys.

## Run/inspect (on the Jetson)

```
docker ps --filter name=aerys-brain      # Up, healthy
docker logs aerys-brain --tail 20
curl -s http://jetson.local:8300/health
```

## Deferred on purpose

HA pipeline pointing at /ask (tomorrow — one config step now), compose file +
deploy-repo home (wave-0 repo reconciliation), streaming responses, per-caller
identity resolution (everything is "Chris (HTTP)" until the resolver seam wires in).
