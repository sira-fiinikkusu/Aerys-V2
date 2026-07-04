"""HTTP transport — the authed /ask door (1e's code half).

n8n mapping: this is the webhook-trigger pattern (like kael-dm or the Myo gesture
webhook), except one FastAPI app serves every future HTTP caller: the HA voice
pipeline tomorrow, the reTerminal later, ad-hoc curl always. Same rule as every
transport: normalize → ask() → reply. No model logic lives here.

Auth: a single Bearer token from Settings (api_token). The voice satellite webhook
taught the pattern — the header check happens before ANYTHING else runs.
"""

import hmac

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from aerys_v2.state import Identity


class AskRequest(BaseModel):
    text: str = Field(min_length=1)
    # Callers name their conversation; HA voice will pass its satellite thread,
    # curl defaults to a shared scratch thread. (Checkpointer keys, not sessions.)
    thread_id: str = "http:default"
    display_name: str = "Chris (HTTP)"
    # HA's aerys_conversation component passes the originating satellite's
    # ConversationInput.device_id here so the spoken follow-up answers on the SAME
    # device. Every other caller (curl, tests) omits it -> single-satellite default.
    device_id: str | None = None


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
    # openapi_url=None too: docs_url/redoc_url only hide the HTML pages; the raw
    # /openapi.json schema is a separate default and would otherwise leak the exact
    # endpoint/field shapes to any unauthenticated prober on the tunnel.
    app = FastAPI(
        title="aerys-v2 brain", docs_url=None, redoc_url=None, openapi_url=None
    )
    # n8n mapping: this is the Identity Resolver's job for HTTP — except HTTP has
    # exactly one possible person, so "resolution" is a constant.
    http_user_id = owner_person_id or "http-caller"

    def require_token(request: Request) -> None:
        # Locked shut unless a REAL token is configured — None OR empty/whitespace
        # means 503 for everything except /health, never an open door. (An empty
        # API_TOKEN in .env parses to "" not None; without this check the credential
        # becomes the literal string "Bearer " — trivially guessable, public repo.)
        if not api_token or not api_token.strip():
            raise HTTPException(status_code=503, detail="api_token not configured")
        auth = request.headers.get("authorization", "")
        # Constant-time compare — no timing side channel on the token (CWE-208).
        if not hmac.compare_digest(auth, f"Bearer {api_token}"):
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
        # TOOLS block: the "voice" thread prefix is what arms ack-then-act inside
        # ask() (service.py parallel-start). For a device command, `reply` here is
        # the router's generated acknowledgment — HA speaks it immediately — and
        # the action finishes in the background, its real outcome appended to this
        # same thread. The transport contract stays one-request-one-string; the
        # asynchrony lives entirely behind the ask() seam.
        reply = ask_fn(
            last_user,
            # Voice/HTTP is the owner's own private channel — pin privacy_context
            # so his dm/private memories surface (the consumption default is now
            # 'public'/fail-closed, so private channels must opt in explicitly).
            {
                "user_id": http_user_id,
                "display_name": "Chris (Voice)",
                "privacy_context": "private",
            },
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
            # Owner's own authed channel → private context (see openai_compat note).
            "privacy_context": "private",
        }
        # Only carry device_id when the caller sent one — keeps the identity dict
        # byte-for-byte as before for every non-satellite caller (curl, tests).
        if body.device_id:
            identity["device_id"] = body.device_id
        # Same seam as the OpenAI shim: callers that name a "voice*" thread_id
        # get ack-then-act for device commands; every other thread gets the
        # sequential router (reply = the actual action outcome). No new fields —
        # the thread_id the caller already sends is the behavior switch.
        reply = ask_fn(body.text, identity, body.thread_id)
        return AskReply(reply=reply, thread_id=body.thread_id)

    return app
