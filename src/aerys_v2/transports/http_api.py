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


def build_app(ask_fn, api_token: str | None) -> FastAPI:
    """App factory — ask_fn injected like every other transport (testable with fakes)."""
    app = FastAPI(title="aerys-v2 brain", docs_url=None, redoc_url=None)

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

    @app.post("/ask", response_model=AskReply)
    def ask_route(body: AskRequest, _: None = Depends(require_token)) -> AskReply:
        identity: Identity = {
            "user_id": "http-caller",
            "display_name": body.display_name,
        }
        reply = ask_fn(body.text, identity, body.thread_id)
        return AskReply(reply=reply, thread_id=body.thread_id)

    return app
