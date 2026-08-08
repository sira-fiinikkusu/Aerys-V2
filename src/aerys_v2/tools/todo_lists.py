"""manage_list — the household's real to-do lists, driven by her.

Owner-commissioned 2026-08-08 with two explicit requirements: she must be
CONSISTENT about lists (one canonical shopping list, one canonical task list —
never freelancing items into note slots or invented lists), and the lists must
be visible on phones at the store. Both fall out of using Home Assistant's
native todo entities: the companion app ships a To-do panel (store visibility
for free), and this tool only accepts the configured list names, so there is
nowhere else for an item to go.

Rides the HOME half of the action stack (existing allowlist gates who can
ask). All four verbs map to HA todo services; reads use return_response.

Failure posture: ToolNode contract — honest strings, never raises. Never
claim a list changed unless HA said so.
"""

from __future__ import annotations

import json
import logging

import httpx

log = logging.getLogger(__name__)

ITEM_MAX = 100  # sane cap; todo summaries are short lines, not documents


def lists_map(csv: str) -> dict[str, str]:
    """Parse HA_TODO_LISTS ('shopping=todo.shopping_list,tasks=todo.tasks')."""
    out: dict[str, str] = {}
    for pair in csv.split(","):
        if "=" in pair:
            name, entity = pair.split("=", 1)
            if name.strip() and entity.strip():
                out[name.strip().lower()] = entity.strip()
    return out


def build_todo_tool(
    *,
    base_url: str,
    token: str,
    lists: dict[str, str],
    client: httpx.Client | None = None,
):
    """Close over config and return the manage_list tool (test seam: inject
    an httpx.Client on a MockTransport)."""
    from langchain_core.tools import tool

    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)
    known = ", ".join(sorted(lists))

    def _call(service: str, entity: str, extra: dict, *, respond: bool = False):
        url = f"{base}/api/services/todo/{service}"
        if respond:
            url += "?return_response=true"
        r = http.post(url, headers=headers, json={"entity_id": entity, **extra})
        r.raise_for_status()
        return r

    @tool
    def manage_list(action: str, list_name: str, item: str = "") -> str:
        """Manage the household's to-do lists (shopping list, tasks).

        CALL THIS TOOL whenever the user asks to add something to the
        shopping list or tasks ("add milk to the list", "put batteries on
        the shopping list", "what's on the list?", "cross off eggs",
        "we got the milk"). These lists show on the family's phones and the
        e-ink displays — this tool is the ONLY correct place for list items;
        never put list items in the sticky note instead.

        action: one of "add", "remove", "complete", "show".
        list_name: which list — "shopping" or "tasks".
        item: the item text (required for add/remove/complete). Use the
        exact item text from "show" when removing or completing.
        """
        act = (action or "").strip().lower()
        lname = (list_name or "").strip().lower()
        entity = lists.get(lname)
        if entity is None:
            return f"Unknown list '{list_name}'. The lists I manage are: {known}."

        text = (item or "").strip()
        if len(text) > ITEM_MAX:
            return (
                f"That item is {len(text)} characters — list items are short "
                f"lines (max {ITEM_MAX}). Shorten it, or put long content in a note."
            )

        try:
            if act == "show":
                r = _call("get_items", entity, {}, respond=True)
                items = (
                    (r.json().get("service_response") or {})
                    .get(entity, {})
                    .get("items", [])
                )
                if not items:
                    return f"The {lname} list is empty."
                lines = []
                for it in items:
                    mark = "x" if it.get("status") == "completed" else " "
                    lines.append(f"[{mark}] {it.get('summary', '')}")
                return f"The {lname} list:\n" + "\n".join(lines)

            if not text:
                return f"'{act}' needs the item text — which item on {lname}?"

            if act == "add":
                _call("add_item", entity, {"item": text})
                return f"Added to the {lname} list: {text}"
            if act == "remove":
                _call("remove_item", entity, {"item": text})
                return f"Removed from the {lname} list: {text}"
            if act == "complete":
                _call("update_item", entity, {"item": text, "status": "completed"})
                return f"Checked off on the {lname} list: {text}"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500 and act in ("remove", "complete"):
                # HA 500s when the summary doesn't match an item — the usual
                # cause is paraphrase; steer the model to exact text.
                return (
                    f"Home Assistant couldn't find '{text}' on the {lname} list. "
                    "Call show first and use the item's exact text."
                )
            return f"The {lname} list change FAILED — Home Assistant said: {e}."
        except httpx.HTTPError as e:
            return f"Home Assistant is unreachable right now ({e})."

        return f"Unknown action '{action}'. Valid actions: add, remove, complete, show."

    return manage_list
