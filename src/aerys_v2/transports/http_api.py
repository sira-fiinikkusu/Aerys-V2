"""HTTP transport — the authed /ask door (1e's code half).

n8n mapping: this is the webhook-trigger pattern (like kael-dm or the Myo gesture
webhook), except one FastAPI app serves every future HTTP caller: the HA voice
pipeline tomorrow, the reTerminal later, ad-hoc curl always. Same rule as every
transport: normalize → ask() → reply. No model logic lives here.

Auth: a single Bearer token from Settings (api_token). The voice satellite webhook
taught the pattern — the header check happens before ANYTHING else runs.
"""

import hmac
import threading
import time
import logging
import unicodedata
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aerys_v2.state import Identity, VOICE_OWNER_ASSUMED_KEY

log = logging.getLogger(__name__)


#: The identity an unrecognized voice lands on. Named once so the door and the
#: engage gate cannot drift on what "a stranger spoke" means.
GUEST_VOICE_ID = "voice-guest"
#: T-17 (11/13): one forever-thread for every unrecognized voice let a 2 AM stray ("Call Ryan's nearest")
#: answer his "Thank you" eight hours later. A guest conversation is a SESSION per device: after this
#: many quiet seconds the next guest turn starts a fresh thread and sees nothing older.
GUEST_SESSION_IDLE_S = 600


_STOP_PHRASES = {
    "stop", "stop talking", "stop it", "okay stop", "ok stop", "hold on",
    "hold up", "thats enough", "enough", "be quiet", "quiet", "shush", "shh",
}
_STOP_NAMES = {"aerys", "harris", "iris", "ares", "aries", "eris"}


def _is_stop_phrase(text: str) -> bool:
    # Apostrophes disappear; other punctuation separates words.
    normalized = "".join(
        "" if char in "'’" else " " if unicodedata.category(char).startswith("P") else char
        for char in text.lower()
    )
    words = normalized.split()
    if words and words[0] in _STOP_NAMES:
        words = words[1:]
    if words and words[-1] in _STOP_NAMES:
        words = words[:-1]
    return " ".join(words) in _STOP_PHRASES


class SpeakerTag(BaseModel):
    id: str = "unknown"
    confidence: float = 0.0


class AskRequest(BaseModel):
    text: str = Field(min_length=1)
    # Callers name their conversation; curl defaults to a shared scratch thread.
    # (Checkpointer keys, not sessions.) IGNORED when voice=True — a voice turn folds
    # into the owner's person thread regardless of what thread_id was sent.
    thread_id: str = "http:default"
    display_name: str = "Chris (HTTP)"
    # HA's aerys_conversation component passes the originating satellite's
    # ConversationInput.device_id here so the spoken follow-up answers on the SAME
    # device. Every other caller (curl, tests) omits it -> single-satellite default.
    device_id: str | None = None
    # EXPLICIT voice flag: a voice caller sets this True to arm the voice behaviors
    # (parallel-start, emotion tags, standard-tier pin) AND fold the turn into the
    # owner's person-keyed thread. Default False = a plain HTTP caller, byte-for-byte
    # the old behavior (thread_id flows through verbatim, no voice-ness). This replaces
    # the old 'name a voice:* thread_id' convention now that voice is person-keyed.
    voice: bool = False
    # Rendering-surface hint (state.SURFACE_KEY). The g2-bridge sends 'lens' so
    # she writes lens-native (~350 chars, one paragraph) and the bridge's display
    # summarizer stays a fallback instead of a paraphrase layer (owner ask
    # 2026-08-26). Absent/None = no surface styling, byte-for-byte old behavior.
    surface: str | None = None
    # Speaker ID (2026-09-20): set by HA's aerys_conversation from the
    # aerys_speaker_stt wrapper's verdict — {"id": "chris"|"<another enrolled voice>"|"unknown",
    # "confidence": 0.71}. Absent = untagged = the owner (pre-speaker-ID behavior).
    speaker: SpeakerTag | None = None


class AskReply(BaseModel):
    reply: str
    thread_id: str
    keep_listening: bool = False


class KaelNoteRequest(BaseModel):
    """POST /kael-note — Kael's reply into the thread she pinged him from.

    thread_id comes from the message_kael payload's return address (task #66).
    family_visible defaults False by owner design: "tests do poison memory" —
    only deliberately-flagged notes become shared family context.
    """

    thread_id: str
    text: str
    family_visible: bool = False


class KaelNoteReply(BaseModel):
    ok: bool
    message_id: str


class KaelTicketUpdate(BaseModel):
    """POST /kael-ticket/{id} — Kael's word on a ticket she opened (board #176).

    status is an A2A task state Kael may set: working, input-required, completed,
    failed, canceled. text is required when it goes back to the owner
    (input-required, completed, failed): an answer with no words is the 1-way
    line all over again.
    """

    status: str
    text: str = ""
    family_visible: bool = False


def build_app(ask_fn, api_token: str | None, owner_person_id: str | None = None,
              gaps_fn=None, health_probe=None, kael_note_fn=None,
              a2a_memory_fn=None, settings=None, speaker_redirect=None,
              kael_tickets=None, home_notice_fn=None, home_webhook_token: str | None = None,
              guest_clock=None) -> FastAPI:
    """App factory — ask_fn injected like every other transport (testable with fakes).

    owner_person_id: when set, every authed HTTP caller IS the owner. The Bearer
    token already proves it's the owner's own infrastructure calling (HA voice
    pipeline, curl from the LAN), so identity.user_id becomes the owner's
    persons.id — the key the memory-context seam retrieves by. That's how
    voice-Chris gets HIS memories instead of an anonymous "http-caller" bucket.
    display_name stays whatever the caller said (channel flavor, not identity).

    health_probe: a zero-arg callable that RAISES when her durable store is
    unreachable (factory.health_probe_for). None = nothing to probe, and /health
    reports ok — the DB-less dev/CI shape. See the /health handler for why this
    seam exists at all.
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

    # Function-level import: keep http_api's module surface free of the discord
    # dependency (this is the general HTTP door), while reusing the ONE canonical
    # person-thread key builder the text gateways use — no drift on the format.
    from aerys_v2.transports.discord_gateway import person_thread_key
    from aerys_v2.config import speaker_person_map, home_job_member_map

    from aerys_v2.keep_listening import ToggleReader, consume_dismissal, mark_opened

    toggle = ToggleReader(settings)

    speaker_persons = speaker_person_map(settings) if settings is not None else {}

    members = home_job_member_map(settings) if settings is not None else {}

    def resolve_speaker(speaker_id: str, confidence: float) -> tuple[str, str] | None:
        """(user_id, display_name) for a tagged voice turn, or None = leave it the owner's.

        Enrolled name -> that person (the owner's own name -> unchanged). "unknown", an
        unlisted name, or a confidence under the floor -> the guest identity when
        voice_unknown_speaker=guest (its own thread; ask()'s allowlist keeps house
        control away from it), else unchanged. Untagged never reaches here.
        """
        if settings is None:
            return None
        name = (speaker_id or "unknown").strip().lower()
        said_unknown = name == "unknown"   # what the RECOGNIZER said, before our floor
        if confidence < settings.voice_speaker_min_confidence:
            person = speaker_persons.get(name)
            if person in members:
                return person, f"{name.capitalize()} (Voice)"
            name = "unknown"
        if said_unknown and confidence >= settings.voice_guest_demote_below:
            # Not a match, but not a stranger either — too close to call. The owner
            # keeps his own turn rather than being answered as a guest in his house.
            # Only when the RECOGNIZER itself was unsure: a NAMED match knocked down
            # by voice_speaker_min_confidence must never land on the owner — that
            # would hand his wife's uncertain turn to Chris, which is the worse error.
            return None
        person = speaker_persons.get(name)
        if person is not None:
            if person == owner_person_id:
                return None
            return person, f"{name.capitalize()} (Voice)"
        if settings.voice_unknown_speaker == "guest":
            return GUEST_VOICE_ID, "Guest (Voice)"
        return None

    def owner_confirmed(speaker_id: str, confidence: float) -> bool:
        """A NAMED match to the owner's own print at or above the floor. Anything else
        that still lands on the owner (untagged, or the recognizer's unsure band) is
        his turn by default only: chat answers it as his, memory never takes it (D2)."""
        if settings is None:
            return False
        name = (speaker_id or "").strip().lower()
        return (speaker_persons.get(name) == owner_person_id
                and confidence >= settings.voice_speaker_min_confidence)

    clock = guest_clock or time.monotonic
    guest_sessions: dict[str, tuple[float, str]] = {}
    guest_lock = threading.Lock()

    def guest_thread(device_id) -> str:
        """The guest's thread for THIS conversation: per device, and new after GUEST_SESSION_IDLE_S of quiet."""
        key = str(device_id or "-")
        with guest_lock:
            now = clock()
            last, session = guest_sessions.get(key, (None, ""))
            if not session or last is None or now - last >= GUEST_SESSION_IDLE_S:
                session = f"{key}:{int(time.time())}-{uuid.uuid4().hex[:6]}"   # unique even within a second
            guest_sessions[key] = (now, session)
            if len(guest_sessions) > 256:                      # bounded: forget devices quiet a day or more
                for k in [k for k, (seen, _) in guest_sessions.items() if now - seen >= 86400]:
                    guest_sessions.pop(k, None)
        return f"{person_thread_key(GUEST_VOICE_ID)}:{session}"

    def voice_thread() -> str:
        """The checkpointer thread a voice turn rides. When an owner is configured (the
        Bearer proves it IS the owner's own infrastructure calling), voice folds into
        his continuous 'person:{owner_id}' thread — the SAME thread as his DM/guild/
        Telegram text, so voice joins cross-surface continuity. Without an owner (dev/CI)
        there is no owner thread to join, so fall back to the legacy shared voice thread."""
        return person_thread_key(http_user_id) if owner_person_id else "voice:beta"

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
    def health() -> JSONResponse:
        """Liveness AND reachability of her memory — 200 healthy, 503 degraded.

        Unauthenticated on purpose: docker HEALTHCHECK + HA availability probes.

        This used to `return {"status": "ok"}` — a literal that could not fail.
        It reported healthy through a ~13.5 hour outage on 2026-07-30/31 while
        every real turn raised OperationalError against a dead pooled connection.
        An endpoint that always says ok teaches everyone downstream to stop
        believing it, so it now answers with something it can actually be wrong
        about: a real query through the same pool the turns use.

        ★ The 503 body is DELIBERATELY generic. This route is unauthenticated and
        reachable through the Cloudflare tunnel, and psycopg's exception text
        embeds the connection string — host, user, and password. The operator
        gets the real exception in the logs; an anonymous prober gets six words.
        """
        if health_probe is None:
            return JSONResponse({"status": "ok"})
        try:
            health_probe()
        except Exception:
            # exception(): full traceback to the operator's side, never the wire.
            log.exception("health probe failed — checkpoint store unreachable")
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "detail": "checkpoint store unreachable"},
            )
        return JSONResponse({"status": "ok"})

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
        # TOOLS block: the EXPLICIT voice flag (identity.voice=True) is what arms
        # ack-then-act inside ask() (service.py parallel-start) — this is ALWAYS a voice
        # turn (HA's Extended OpenAI Conversation IS the voice pipeline). For a device
        # command, `reply` here is the router's generated acknowledgment — HA speaks it
        # immediately — and the action finishes in the background, its real outcome
        # appended to this same thread. The transport contract stays one-request-one-
        # string; the asynchrony lives entirely behind the ask() seam. The thread is the
        # owner's person thread (voice_thread) so voice joins cross-surface continuity.
        reply = ask_fn(
            last_user,
            # Voice/HTTP is the owner's own private channel — pin privacy_context so his
            # dm/private memories surface (the consumption default is now 'public'/fail-
            # closed, so private channels must opt in explicitly). voice=True is the
            # explicit signal now that the thread is person-keyed and no longer names
            # 'voice'; it also tags this turn's content fail-closed 'private' at ingest
            # (relaxed off-hot-path by the judge), the same protection a DM gets.
            {
                "user_id": http_user_id,
                "display_name": "Chris (Voice)",
                "privacy_context": "private",
                "voice": True,
            },
            voice_thread(),   # owner's person thread (voice rides the owner thread — owner decision)
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

    # speaker_redirect is injected on its own by cli.py (explicit beats derived; tests can
    # pass a fake). Correction to a20bc2f's message: cli.py DOES pass settings here — the
    # first office test failed on mpv's keep-open, not on missing wiring.

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
        # Same carry-only-when-sent contract as device_id: every non-lens caller's
        # identity dict stays byte-for-byte unchanged.
        if body.surface:
            identity["surface"] = body.surface
        # Voice is armed by the EXPLICIT body.voice flag OR a legacy voice thread name.
        # The HA aerys_conversation component predates the tie-in and still posts
        # thread_id="voice:beta" with NO voice flag — so a "voice"-prefixed thread is
        # treated as a voice turn here too, folding it onto the owner's person thread
        # instead of stranding it on the standalone voice:beta thread (the "2nd
        # person_id" bug, 2026-07-05). Both paths get ack-then-act, emotion tags, and
        # content tagged fail-closed 'private' at ingest like a DM. A plain caller
        # (voice=False, non-voice thread — curl/tests) is byte-for-byte unchanged.
        if body.voice or str(body.thread_id or "").startswith("voice"):
            identity["voice"] = True
            thread_id = voice_thread()
            if body.speaker is None:
                if owner_person_id:
                    identity[VOICE_OWNER_ASSUMED_KEY] = True    # untagged: his by default only
            else:
                # Who spoke decides WHOSE turn this is. The Bearer proved the owner's
                # infrastructure is calling; the voice decides the person.
                who = resolve_speaker(body.speaker.id, body.speaker.confidence)
                if who is None and owner_person_id and not owner_confirmed(body.speaker.id, body.speaker.confidence):
                    identity[VOICE_OWNER_ASSUMED_KEY] = True    # the unsure band: his by default only
                # The DOOR decides whether a verdict demotes; the gate obeys that flag
                # rather than re-reading the id, so the two can never disagree.
                demoted = who is not None and who[0] == GUEST_VOICE_ID
                identity["speaker"] = {"id": body.speaker.id, "confidence": body.speaker.confidence,
                                       "demoted": demoted}
                if who is not None and who[0] in members and body.speaker.confidence < settings.voice_speaker_min_confidence:
                    identity["speaker"]["unsure"] = True
                if who is not None:
                    identity["user_id"], identity["display_name"] = who
                    thread_id = (guest_thread(body.device_id) if who[0] == GUEST_VOICE_ID
                                 else person_thread_key(who[0]))
        else:
            thread_id = body.thread_id
        is_voice = bool(identity.get("voice"))
        # Consume even when OFF, so re-enabling cannot resurrect an answered mic.
        dismissed = is_voice and consume_dismissal(body.device_id, body.text)
        # D34 barge-in ("Aerys, stop" while her voice plays on the desk speakers): cut her off, no
        # model call. After the dismissal above, so a "stop" also clears a mic she had opened.
        if (is_voice and speaker_redirect is not None and _is_stop_phrase(body.text)
                and speaker_redirect.handles(body.device_id)
                and speaker_redirect.playing(body.device_id)):
            speaker_redirect.stop(body.device_id)
            return AskReply(reply="", thread_id=thread_id, keep_listening=False)
        if dismissed and toggle.enabled():
            return AskReply(reply="", thread_id=thread_id)
        reply = ask_fn(body.text, identity, thread_id)
        keep_listening = False
        if is_voice:
            from aerys_v2.service import strip_emotion_tags

            keep_listening = strip_emotion_tags(reply).strip().endswith("?") and toggle.enabled()

        # Speaker redirect (2026-09-26): a voice turn from a satellite whose speaker
        # is gone is also spoken on the redirect target (Leviathan's desk speakers).
        if identity.get("voice") and speaker_redirect is not None:
            if keep_listening and speaker_redirect.handles(body.device_id):
                from aerys_v2.factory import satellite_map_from
                import httpx

                # The dead speaker finishes immediately. Only the redirect's
                # playback acknowledgement may reopen this satellite's mic.
                keep_listening = False
                satellite = satellite_map_from(settings.ha_satellite_map).get(body.device_id)
                if satellite and settings.ha_token is not None:
                    def reopen() -> None:
                        r = httpx.post(
                            f"{settings.ha_base_url.rstrip('/')}/api/services/assist_satellite/start_conversation",
                            headers={"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"},
                            json={"entity_id": satellite, "start_message": "Mm?", "preannounce": False},
                            timeout=15.0,
                        )
                        r.raise_for_status()
                        mark_opened(body.device_id)

                    speaker_redirect.say(body.device_id, reply, then=reopen)
                else:
                    speaker_redirect.say(body.device_id, reply)
            else:
                speaker_redirect.say(body.device_id, reply)
        # Gap #37: an exchange on Kael's line becomes a durable memory, so the
        # her on any OTHER thread can know it happened. Fires on the FINAL
        # thread_id — a voice turn was already remapped off kael:* above, so
        # voice never lands here. The writer gates + fails open internally.
        if a2a_memory_fn is not None:
            a2a_memory_fn(thread_id, body.text, reply)
        if keep_listening:
            mark_opened(body.device_id)
        return AskReply(reply=reply, thread_id=thread_id, keep_listening=keep_listening)

    @app.post("/kael-note", response_model=KaelNoteReply)
    def kael_note_route(
        body: KaelNoteRequest, _: None = Depends(require_token)
    ) -> KaelNoteReply:
        # The return direction of her message_kael line (task #66). Same Bearer
        # as every owner-infrastructure caller; unarmed (DB-less dev, no graph)
        # = honest 503, never a silent drop.
        if kael_note_fn is None:
            raise HTTPException(status_code=503, detail="kael line not armed")
        if not body.thread_id.strip() or not body.text.strip():
            raise HTTPException(status_code=422, detail="thread_id and text required")
        msg_id = kael_note_fn(
            body.thread_id.strip(), body.text.strip(), body.family_visible
        )
        return KaelNoteReply(ok=True, message_id=msg_id)

    # Kael tickets (board #176): her asks come back, and the owner's come back to him.
    # kael_tickets is a tickets.TicketDoor; unarmed (DB-less dev) = honest 503.
    def _ticket_door():
        if kael_tickets is None:
            raise HTTPException(status_code=503, detail="kael tickets not armed")
        return kael_tickets

    @app.get("/kael-tickets")
    def kael_tickets_route(open_only: bool = True, limit: int = 20,
                           _: None = Depends(require_token)) -> dict:
        door = _ticket_door()
        rows = door.store.recent(limit=max(1, min(limit, 100)), open_only=open_only)
        return {"tickets": [t.public() for t in rows]}

    @app.get("/kael-ticket/{ticket_id}")
    def kael_ticket_get(ticket_id: int, _: None = Depends(require_token)) -> dict:
        ticket = _ticket_door().store.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="no such ticket")
        return ticket.public()

    @app.post("/kael-ticket/{ticket_id}")
    def kael_ticket_update(ticket_id: int, body: KaelTicketUpdate,
                           _: None = Depends(require_token)) -> dict:
        from aerys_v2.tickets import KAEL_STATUSES, OWED_TO_OWNER

        door = _ticket_door()
        if body.status not in KAEL_STATUSES:
            raise HTTPException(status_code=422,
                                detail=f"status must be one of {', '.join(KAEL_STATUSES)}")
        if body.status in OWED_TO_OWNER and not body.text.strip():
            raise HTTPException(status_code=422, detail=f"{body.status} needs text")
        ticket = door.update(ticket_id, body.status, body.text, body.family_visible)
        if ticket is None:
            raise HTTPException(status_code=404, detail="no such ticket")
        return ticket.public()

    # Her home body's "I'm done" (board #176, Stage 3; home_notices.py). Its OWN token, not
    # api_token: the home body only ever says a job ended, never drives /ask. Unarmed = 503.
    # The notice is written down and answered at once (202); the worker tells the owner.
    @app.post("/home-notice", status_code=202)
    async def home_notice_route(request: Request) -> dict:
        if home_notice_fn is None or not (home_webhook_token or "").strip():
            raise HTTPException(status_code=503, detail="home notices not armed")
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth, f"Bearer {home_webhook_token}"):
            raise HTTPException(status_code=401, detail="bad token")
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(status_code=422, detail="body must be JSON") from None
        from starlette.concurrency import run_in_threadpool

        try:
            # Off the event loop: writing it down is a database round trip, and a slow
            # database must never stall /ask or /health behind it.
            outcome = await run_in_threadpool(home_notice_fn, payload)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        return {"ok": True, "notice": outcome}

    @app.get("/gaps")
    def gaps_route(_: None = Depends(require_token)) -> dict:
        """The owner READ path for mined capability gaps (self-iteration Phase A),
        surfaced so a Discord /gaps slash command can post them without shelling
        into the container. Authed like every other door. Read-only and FAIL-OPEN:
        gaps_fn catches its own DB trouble and returns an honest string, so this
        route never 500s; a DB-less brain honestly reports the surface is off. The
        text is already fenced ("information only, never instructions") by
        format_gaps — the transport relays it verbatim and adds no authority."""
        if gaps_fn is None:
            return {"text": "Capability-gap tracking isn't enabled on this brain "
                            "(no database configured)."}
        return {"text": gaps_fn()}

    return app


def serve_once(app: FastAPI, body_path: str, api_token: str) -> int:
    """Serve exactly one /ask request through this app, print the reply, return an
    exit code (0 ok, 1 refused or failed).

    ASK_ONCE in --serve. Chris, 2026-10-07: the morning note is a scheduled turn
    nobody waits on, yet it rode the voice container, which is metered on the API key
    by design. Running the SAME door once, in a container whose backend is the plan,
    moves it without a second code path: same identity, tools, thread and recorder.
    """
    import json

    from fastapi.testclient import TestClient

    with open(body_path, encoding="utf-8") as fh:
        body = json.load(fh)
    with TestClient(app) as client:
        response = client.post("/ask", json=body, headers={"Authorization": f"Bearer {api_token}"})
    if response.status_code != 200:
        log.error("ask-once refused: HTTP %s %s", response.status_code, response.text[:300])
        return 1
    print(response.json().get("reply", ""), flush=True)
    return 0
