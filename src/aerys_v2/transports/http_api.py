"""HTTP transport — the authed /ask door (1e's code half).

n8n mapping: this is the webhook-trigger pattern (like kael-dm or the Myo gesture
webhook), except one FastAPI app serves every future HTTP caller: the HA voice
pipeline tomorrow, the reTerminal later, ad-hoc curl always. Same rule as every
transport: normalize → ask() → reply. No model logic lives here.

Auth: a single Bearer token from Settings (api_token). The voice satellite webhook
taught the pattern — the header check happens before ANYTHING else runs.
"""

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from aerys_v2.state import Identity


class AskRequest(BaseModel):
    text: str = Field(min_length=1)
    # Callers name their conversation; HA voice will pass its satellite thread,
    # curl defaults to a shared scratch thread. (Checkpointer keys, not sessions.)
    thread_id: str = "http:default"
    display_name: str = "Chris (HTTP)"


class AskReply(BaseModel):
    reply: str
    thread_id: str


def build_app(ask_fn, api_token: str | None, owner_person_id: str | None = None) -> FastAPI:
    """App factory — ask_fn injected like every other transport (testable with fakes).

    owner_person_id: when set, every authed HTTP caller IS the owner. The Bearer
    token already proves it's the owner's own infrastructure calling (HA voice
    pipeline, curl from the LAN), so identity.user_id becomes the owner's
    persons.id — the key the memory-context seam retrieves by. That's how
    voice-Chris gets HIS memories instead of an anonymous "http-caller" bucket.
    display_name stays whatever the caller said (channel flavor, not identity).
    """
    app = FastAPI(title="aerys-v2 brain", docs_url=None, redoc_url=None)
    # n8n mapping: this is the Identity Resolver's job for HTTP — except HTTP has
    # exactly one possible person, so "resolution" is a constant.
    http_user_id = owner_person_id or "http-caller"

    def require_token(request: Request) -> None:
        # Locked shut unless a token is configured — an unset token means 503 for
        # everything except /health, never an open door.
        if api_token is None:
            raise HTTPException(status_code=503, detail="api_token not configured")
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {api_token}":
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/health")
    def health() -> dict:
        # Unauthenticated on purpose: docker HEALTHCHECK + HA availability probes.
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(_: None = Depends(require_token)) -> dict:
        # Extended OpenAI Conversation validates the connection with a models.list
        # call before saving (observed live: three 404s = "unexpected error" in the
        # HA UI). One stub entry satisfies it.
        return {"object": "list", "data": [
            {"id": "aerys-v2", "object": "model", "owned_by": "aerys"}]}

    @app.post("/v1/chat/completions")
    def openai_compat(body: dict, _: None = Depends(require_token)) -> dict:
        """OpenAI-protocol shim — exists so HA's Extended OpenAI Conversation can
        point at the Brain like any OpenAI server (same integration the current
        voice pipeline uses, different base_url). We take the LAST user message as
        the turn; HISTORY comes from our checkpointer, not from the request — HA
        resends its own transcript, and two history owners is the contamination
        bug again, so the request transcript is deliberately ignored."""
        msgs = body.get("messages", [])
        last_user = next(
            (m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), ""
        )
        if isinstance(last_user, list):  # OpenAI content-parts form
            last_user = " ".join(
                p.get("text", "") for p in last_user if isinstance(p, dict)
            )
        if not last_user.strip():
            raise HTTPException(status_code=400, detail="no user message")
        reply = ask_fn(
            last_user,
            {"user_id": http_user_id, "display_name": "Chris (Voice)"},
            "voice:beta",   # one shared voice thread for the beta pipeline (owner decision: voice rides the owner thread)
        )
        return {
            "id": "aerys-v2",
            "object": "chat.completion",
            "model": body.get("model", "aerys-v2"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/ask", response_model=AskReply)
    def ask_route(body: AskRequest, _: None = Depends(require_token)) -> AskReply:
        identity: Identity = {
            "user_id": http_user_id,
            "display_name": body.display_name,
        }
        reply = ask_fn(body.text, identity, body.thread_id)
        return AskReply(reply=reply, thread_id=body.thread_id)

    return app
