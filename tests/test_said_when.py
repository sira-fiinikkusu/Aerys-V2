"""Ticket T-6 (2026-11-11): a line from Monday said "tomorrow", and on Wednesday she read it as tomorrow from
Wednesday. What she's shown of earlier days now carries when it was said."""
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.services.content_privacy import CONTENT_PRIVACY_KEY, gate_for_room
from aerys_v2.services.said_when import SAID_AT_KEY, UNRECORDED, said_label, with_said_dates

MONDAY = "2026-11-09T19:39:00+00:00"            # Mon 11/09 2:39 PM Eastern
WEDNESDAY = datetime(2026, 11, 11, 19, 22, tzinfo=timezone.utc)
THIS_MORNING = "2026-11-11T14:05:00+00:00"      # Wed 9:05 AM Eastern


def human(text, said=None, mid=None, **kw):
    kwargs = {CONTENT_PRIVACY_KEY: "private", **kw}
    if said:
        kwargs[SAID_AT_KEY] = said
    return HumanMessage(content=text, id=mid, additional_kwargs=kwargs)


def test_the_venice_line_is_read_against_monday():
    history = [human("i have a doctor's appointment in venice tomorrow", MONDAY, "h1"),
               AIMessage(content="Tomorrow in the area looks cloudy, 69 to 83.", id="a1"),
               human("I scheduled the tattoo appointment", "2026-11-11T19:22:00+00:00", "h2")]
    shown = with_said_dates(history, WEDNESDAY)
    assert shown[0].content == "[said Mon Nov 9, 2:39 PM] i have a doctor's appointment in venice tomorrow"
    assert shown[0].id == "h1" and shown[0].additional_kwargs == history[0].additional_kwargs
    assert shown[1] is history[1]                     # her own reply: never labelled (she'd copy it)
    assert shown[2] is history[2]                     # the current turn: never labelled
    assert history[0].content.startswith("i have")    # the record itself is untouched


def test_earlier_today_stays_as_it_is():
    history = [human("call me tonight", THIS_MORNING), human("now")]
    assert with_said_dates(history, WEDNESDAY)[0] is history[0]


def test_a_line_from_before_stamping_says_its_date_wasnt_recorded_only_when_it_matters():
    history = [human("the plumber comes tomorrow"), human("I like the blue one"), human("now")]
    shown = with_said_dates(history, WEDNESDAY)
    assert shown[0].content == f"{UNRECORDED} the plumber comes tomorrow"
    assert shown[1] is history[1]


def test_odd_content_and_bad_stamps_are_left_alone():
    blocks = HumanMessage(content=[{"type": "text", "text": "tomorrow"}],
                          additional_kwargs={SAID_AT_KEY: MONDAY})
    bad = human("see you tomorrow", "not-a-time")
    shown = with_said_dates([blocks, bad, human("now")], WEDNESDAY)
    assert shown[0] is blocks
    assert shown[1].content == f"{UNRECORDED} see you tomorrow"


def test_the_label_is_eastern_and_portable():
    assert said_label(datetime(2026, 11, 10, 4, 50, tzinfo=timezone.utc)) == "[said Mon Nov 9, 11:50 PM]"
    assert said_label(datetime(2026, 11, 10, 17, 5, tzinfo=timezone.utc)) == "[said Tue Nov 10, 12:05 PM]"


def test_every_room_path_shows_the_dates():
    history = [human("dentist tomorrow", MONDAY), human("now")]
    private = gate_for_room(history, {"privacy_context": "private"})
    public = gate_for_room([human("dentist tomorrow", MONDAY, **{CONTENT_PRIVACY_KEY: "public"}), human("now")],
                           {"privacy_context": "public"})
    assert private[0].content.startswith("[said ")
    assert public[0].content.startswith("[said ")


def test_each_new_line_is_stamped_when_it_is_said():
    from aerys_v2.service import _human_turn

    m = _human_turn("hello", "private", "m1")
    assert datetime.fromisoformat(m.additional_kwargs[SAID_AT_KEY]).tzinfo is not None


def test_her_time_line_explains_the_labels():
    from aerys_v2.factory import _where_when_line

    line = _where_when_line("person:x", {})
    assert "[said ...]" in line and "against that day" in line and "no recorded date" in line
