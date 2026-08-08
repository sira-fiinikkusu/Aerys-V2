"""manage_list (owner ask 8/08) — offline, MockTransport. What these prove:
the closed name→entity map is the consistency contract (unknown lists refused,
nothing touches HA), all four verbs hit the right HA todo services, show
renders items with completion marks, a mismatched summary on remove/complete
steers toward exact text instead of failing opaquely, and the factory arms
the tool only when the knob is set."""

import json

import httpx

from aerys_v2.tools.todo_lists import ITEM_MAX, build_todo_tool, lists_map

LISTS = {"shopping": "todo.shopping_list", "tasks": "todo.tasks"}


def make_tool(captured, *, items=None, fail_status=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append((request.url.path, dict(request.url.params), body))
        if fail_status:
            return httpx.Response(fail_status, text="boom")
        if request.url.path.endswith("/get_items"):
            return httpx.Response(
                200,
                json={"service_response": {body["entity_id"]: {"items": items or []}}},
            )
        return httpx.Response(200, json=[])

    return build_todo_tool(
        base_url="http://ha.test",
        token="tok",
        lists=LISTS,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_lists_map_parses_csv():
    assert lists_map("shopping=todo.shopping_list, tasks=todo.tasks") == LISTS
    assert lists_map("") == {}


def test_unknown_list_is_refused_without_touching_ha():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"action": "add", "list_name": "wishlist", "item": "pony"})
    assert "Unknown list" in out and "shopping" in out
    assert captured == []


def test_add_remove_complete_hit_the_right_services():
    captured = []
    tool = make_tool(captured)
    assert "Added" in tool.invoke({"action": "add", "list_name": "shopping", "item": "milk"})
    assert "Removed" in tool.invoke({"action": "remove", "list_name": "shopping", "item": "milk"})
    assert "Checked off" in tool.invoke({"action": "complete", "list_name": "tasks", "item": "mow lawn"})
    paths = [c[0] for c in captured]
    assert paths == [
        "/api/services/todo/add_item",
        "/api/services/todo/remove_item",
        "/api/services/todo/update_item",
    ]
    assert captured[0][2] == {"entity_id": "todo.shopping_list", "item": "milk"}
    assert captured[2][2] == {"entity_id": "todo.tasks", "item": "mow lawn", "status": "completed"}


def test_show_renders_items_with_completion_marks():
    captured = []
    tool = make_tool(
        captured,
        items=[
            {"summary": "milk", "status": "needs_action"},
            {"summary": "eggs", "status": "completed"},
        ],
    )
    out = tool.invoke({"action": "show", "list_name": "shopping"})
    assert "[ ] milk" in out and "[x] eggs" in out
    assert captured[0][1].get("return_response") == "true"


def test_show_empty_list_is_honest():
    tool = make_tool([])
    assert "empty" in tool.invoke({"action": "show", "list_name": "tasks"})


def test_missing_item_text_asks_for_it():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"action": "add", "list_name": "shopping"})
    assert "needs the item text" in out
    assert captured == []


def test_item_length_capped():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"action": "add", "list_name": "shopping", "item": "x" * (ITEM_MAX + 1)})
    assert "characters" in out
    assert captured == []


def test_summary_mismatch_steers_to_exact_text():
    tool = make_tool([], fail_status=500)
    out = tool.invoke({"action": "remove", "list_name": "shopping", "item": "melk"})
    assert "couldn't find" in out and "exact text" in out


def test_factory_arms_only_when_knob_set():
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for

    base = dict(_env_file=None, anthropic_api_key="sk-test", ha_token="tok")
    without = {t.name for t in action_tools_for(Settings(**base))}
    withit = {
        t.name
        for t in action_tools_for(
            Settings(**base, ha_todo_lists="shopping=todo.shopping_list")
        )
    }
    assert "manage_list" not in without
    assert "manage_list" in withit
