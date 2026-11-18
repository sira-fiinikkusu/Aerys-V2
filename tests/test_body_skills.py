"""D67 (Chris 11/18 12:22, "yes please"): her brain can read the skills her bodies write.

12:19 he asked "did you make a skill for maptap?"; her brain said no (its list held one skill) while her
home body had written a fuller one at 11:29. Her bodies' skills reach the house through the door, into
portable_memory_feed (key_label skill.<slug>), after the door's own checks; her brain's list read only its
own shelf. Now that feed is a read-only shelf at /home-skills/, listed beside her own; her own skill wins a
name clash; the bodies stay the only writers of theirs.
"""
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from aerys_v2.skills import BODY_SKILLS, BodySkillsBackend, backend_for, rows_from_feed, skills_middleware
from test_her_skills import CONTENT, OWNER, PATH, call, config, native

WHEN = datetime(2026, 11, 18, 16, 29, tzinfo=timezone.utc)


_SEQ = iter(range(1, 10_000))


def feed_row(slug="maptap-play-with-drag-zoom-and-tap", content="1. Open https://maptap.gg/ with a picture.\nMAPTAP_BODY",
             event="insert", context="skill: Nov 18, 2026 when browse_pointer gained drag", source="home", memory_id=None):
    """(rack_seq, memory_id, key_label, event_type, content, context, source, observed_at), in feed order."""
    return (next(_SEQ), memory_id or f"m-{slug}", f"skill.{slug}", event, content, context, source, WHEN)


def body(rows, **kw):
    return BodySkillsBackend(lambda: rows_from_feed(rows), refresh_s=0, **kw)


def test_her_body_skills_show_in_her_list_and_read_like_her_own():
    store = InMemoryStore()
    backend_for(store).write(PATH, CONTENT)
    graph, log = native([call("read_file", file_path=f"{BODY_SKILLS}maptap-play-with-drag-zoom-and-tap/SKILL.md"),
                         AIMessage(content="Done")], store,
                        middleware=skills_middleware(store, OWNER, body_skills=body([feed_row()])))
    graph.invoke({"messages": [HumanMessage(content="Did you make a skill for MapTap?")]}, config())
    first = str(log[0]["messages"])
    assert "maptap-play-with-drag-zoom-and-tap" in first and "home body" in first   # listed, with where it's from
    assert "sample" in first                                                          # her own shelf still there
    assert "MAPTAP_BODY" not in first and "MAPTAP_BODY" in str(log[1]["messages"])   # body read on demand


def test_her_body_skills_are_read_only():
    store = InMemoryStore()
    shelf = body([feed_row()])
    graph, log = native([call("write_file", file_path=f"{BODY_SKILLS}maptap-play-with-drag-zoom-and-tap/SKILL.md",
                              content="---\nname: maptap-play-with-drag-zoom-and-tap\ndescription: x\n---\nREWRITTEN"),
                         AIMessage(content="ok")], store,
                        middleware=skills_middleware(store, OWNER, body_skills=shelf))
    out = graph.invoke({"messages": [HumanMessage(content="Rewrite the MapTap skill")]}, config())
    refusal = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert refusal.status == "error"
    got = shelf.download_files([f"/maptap-play-with-drag-zoom-and-tap/SKILL.md"])[0].content.decode()
    assert "MAPTAP_BODY" in got and "REWRITTEN" not in got
    assert shelf.write("/x/SKILL.md", "y").error and shelf.edit("/maptap-play-with-drag-zoom-and-tap/SKILL.md", "1.", "2.").error


def test_her_own_skill_wins_a_name_clash():
    store = InMemoryStore()
    backend_for(store).write(PATH, CONTENT)                                   # her own "sample"
    graph, log = native([AIMessage(content="ok")], store,
                        middleware=skills_middleware(store, OWNER, body_skills=body([feed_row(slug="sample")])))
    graph.invoke({"messages": [HumanMessage(content="hi")]}, config())
    first = str(log[0]["messages"])
    assert "A sample workflow" in first and "From my home body" not in first


def test_only_live_well_formed_skills_come_off_the_feed():
    rows = rows_from_feed([feed_row(), feed_row(slug="gone", event="tombstone"), feed_row(slug="Bad Slug!"),
                           feed_row(slug="empty", content="  "),
                           (99, "m-g", "interest.garden", "insert", "roses", "", "stick", WHEN)])
    assert [r["slug"] for r in rows] == ["maptap-play-with-drag-zoom-and-tap"]


def test_a_feed_hiccup_keeps_the_last_view_and_reads_are_throttled():
    calls = []

    def load():
        calls.append(1)
        if len(calls) > 1:
            raise ConnectionError("NAS blip")
        return rows_from_feed([feed_row()])

    clock = {"t": 0.0}
    shelf = BodySkillsBackend(load, refresh_s=30, clock=lambda: clock["t"])
    assert [e["path"] for e in shelf.ls("/").entries] == ["/maptap-play-with-drag-zoom-and-tap/"]
    shelf.ls("/")
    assert len(calls) == 1                                                    # within 30 s: no second read
    clock["t"] = 31.0
    assert [e["path"] for e in shelf.ls("/").entries] == ["/maptap-play-with-drag-zoom-and-tap/"]   # blip: kept
    assert len(calls) == 2



def test_a_revised_skill_stays_and_a_deleted_one_goes():
    """Live 11/18 15:19: the door writes a revision as an insert of the new version and THEN a tombstone of
    the old one, under the same label. A latest-row-per-label read dropped her MapTap and Wordle skills."""
    rows = rows_from_feed([
        feed_row(memory_id="old", content="v1"),                               # the first version
        feed_row(memory_id="new", content="v2 REVISED"),                       # the revision ...
        feed_row(memory_id="old", event="tombstone", content=None),           # ... then the old one retired
        feed_row(slug="gone-skill", memory_id="g1", content="x"),
        feed_row(slug="gone-skill", memory_id="g1", event="tombstone", content=None),   # deleted outright
    ])
    assert [(r["slug"], r["content"]) for r in rows] == [("maptap-play-with-drag-zoom-and-tap", "v2 REVISED")]


def test_nothing_on_the_shelf_changes_by_any_door_async_included():
    import asyncio
    shelf = body([feed_row()])
    path = "/maptap-play-with-drag-zoom-and-tap/SKILL.md"
    assert shelf.download_files([path])[0].content
    assert asyncio.run(shelf.adelete(path)).error and shelf.delete(path).error
    assert shelf.download_files([path])[0].content                         # still there (Codex 11/18)


def test_a_feed_that_never_answered_says_so_instead_of_an_empty_shelf():
    """Codex 11/18: a cold-load failure looked like a body with no skills, the very answer D67 fixes."""
    def down():
        raise ConnectionError("NAS down")
    shelf = BodySkillsBackend(down, refresh_s=0)
    listed = shelf.ls("/")
    assert listed.error and "couldn't" in listed.error and not listed.entries
