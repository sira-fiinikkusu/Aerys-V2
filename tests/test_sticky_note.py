"""sticky_note — offline, MockTransport. What these prove: the note lands on
the configured input_text entity via set_value, blank input never touches HA,
over-limit text is trimmed with visible honesty (HA hard-rejects >255), HA
failure comes back as an honest string, and the factory arms the tool only
when the entity knob is set."""

import json

import httpx

from aerys_v2.tools.sticky_note import NOTE_MAX, build_sticky_note_tool

ENTITY = "input_text.sticky_note_from_aerys"


def make_tool(captured, *, fail=False):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(500 if fail else 200, json=[])

    return build_sticky_note_tool(
        base_url="http://ha.test",
        token="tok",
        entity_id=ENTITY,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_note_lands_on_the_configured_entity():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"message": "Good luck at the demo today — you built this. - A"})
    assert "Note is up" in out
    path, body = captured[0]
    assert path == "/api/services/input_text/set_value"
    assert body["entity_id"] == ENTITY
    assert body["value"].startswith("Good luck")


def test_blank_note_never_touches_ha():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"message": "   "})
    assert "nothing to write" in out
    assert captured == []


def test_over_limit_is_trimmed_and_honest():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"message": "x" * 400})
    _, body = captured[0]
    assert len(body["value"]) <= NOTE_MAX
    assert body["value"].endswith("…")
    assert "trimmed" in out


def test_ha_failure_is_honest_string():
    captured = []
    tool = make_tool(captured, fail=True)
    out = tool.invoke({"message": "hello"})
    assert "did NOT reach the display" in out


def test_factory_arms_only_when_entity_set():
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for

    base = dict(_env_file=None, anthropic_api_key="sk-test", ha_token="tok")
    names_without = {t.name for t in action_tools_for(Settings(**base))}
    names_with = {
        t.name
        for t in action_tools_for(Settings(**base, ha_sticky_note_entity=ENTITY))
    }
    assert "sticky_note" not in names_without
    assert "sticky_note" in names_with
