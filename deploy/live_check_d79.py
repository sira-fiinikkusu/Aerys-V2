# D79 (one her, not twins): live checks to append to ~/bench/live_checks.py on the Jetson.
# Every line prints "#D79 ...: True|False", and False on any failure (an image without D79 fails to import).
# The deploy expects the switch OFF (it goes on only by his separate go line). If any D79 line is False,
# the snippet ends with exit status 1, so `docker exec ... live_checks.py` fails as well as the grep.
_D79 = []
try:
    from aerys_v2.config import Settings as _SD79
    from aerys_v2.factory import action_allowlist_for as _alD79
    from aerys_v2.service import text_one_loop_eligible as _eD79, text_one_loop_flag as _fD79

    _sD79 = _SD79()
    _ownD79 = str(_sD79.owner_person_id or "")
    _allowD79 = _alD79(_sD79)
    _dmD79 = {"user_id": _ownD79, "privacy_context": "private", "platform": "discord", "channel_kind": "dm"}
    _D79.append(_fD79(_sD79)() is False)
    print("#D79 the one-loop switch is OFF as deployed (it goes on only by his go line):", _D79[-1])
    _D79.append(bool(_ownD79) and _eD79(_dmD79, _allowD79, _ownD79)
                and not _eD79({**_dmD79, "channel_kind": "guild"}, _allowD79, _ownD79)
                and not _eD79({**_dmD79, "user_id": "discord:1"}, _allowD79, _ownD79)
                and not _eD79({**_dmD79, "voice": True}, _allowD79, _ownD79)
                and not _eD79({**_dmD79, "platform": "kael"}, _allowD79, _ownD79))
    print("#D79 only a household DM is eligible: his DM yes; a guild room, a stranger, voice and Kael's lane no:",
          _D79[-1])
except Exception as _e:
    _D79.append(False)
    print("#D79 one-loop switch and eligibility (failed to load):", False)
try:
    from langchain_core.messages import HumanMessage as _HD79
    from aerys_v2 import factory as _facD79

    _buildD79 = _facD79._action_prompt_builder(
        "SOUL", charter="CHARTER", overlay="", context_fn=lambda p, q, v: "PROFILE ONLY",
        conversational_context_fn=lambda p, q, v: "FULL MEMORY", family_notes_fn=None, room_context_fn=None,
        portable_context_fn=None, presence_fn=None, voice_room_fn=None, shared_surface_ids=None,
        context_trailing=False)
    _idD79 = {"user_id": "u", "display_name": "Chris", "privacy_context": "private", "platform": "discord",
              "channel_kind": "dm"}
    _pD79, _specD79, _ = _buildD79([_HD79(content="hi")], {"configurable": {
        "thread_id": "person:u", "identity": _idD79, "specialist": False, "text_one_loop": True,
        "one_loop_correction": "CORRECTION"}})
    _qD79, _, _ = _buildD79([_HD79(content="lights off")], {"configurable": {
        "thread_id": "person:u", "identity": _idD79, "specialist": True}})
    _D79.append(_pD79[0].content.startswith("SOUL") and _facD79.TEXT_BANTER_OVERLAY in _pD79[0].content
                and "FULL MEMORY" in _pD79[0].content and "CORRECTION" in _pD79[0].content and not _specD79
                and [m.type for m in _pD79[1:]] == ["human"]
                and _facD79.TEXT_BANTER_OVERLAY not in _qD79[0].content and "PROFILE ONLY" in _qD79[0].content)
    print("#D79 a one-loop turn gets her soul, the typed banter voice, full memory and its correction in the "
          "system prompt (not as a turn); a specialist turn does not:", _D79[-1])
except Exception as _e:
    _D79.append(False)
    print("#D79 one-loop prompt (failed to load):", False)
try:
    from langchain_core.messages import AIMessage as _AD79, ToolMessage as _TD79
    from aerys_v2.service import (_claims_effect_without_doing_it as _cD79, _one_loop_check as _okD79,
                                  _one_loop_receipts as _rcD79)

    def _turnD79(name, said, args=None):
        return [_AD79(content="", tool_calls=[{"id": "c", "name": name, "args": args or {}}]),
                _TD79(content="ok", name=name, tool_call_id="c"), _AD79(content=said)]

    _rD79 = "I’ve started a home job to update the MapTap skill, Chris."
    _D79.append(_cD79("you are good to update your skill", _rD79)
                and _okD79(_rD79, _rcD79(_turnD79("list_async_tasks", _rD79))) is not None
                and _okD79(_rD79, _rcD79(_turnD79("send_message", _rD79))) is not None
                and _okD79(_rD79, _rcD79(_turnD79("start_async_task", _rD79))) is None
                and _okD79("Let me check the lamp. It is off.", _rcD79(_turnD79(
                    "home_control", "x", {"operation": "get_state"}))) is None)
    print("#D79 claims are matched to receipts: a read or a send doesn't back a start, a start does, a look keeps "
          "a promise to look:", _D79[-1])
except Exception as _e:
    _D79.append(False)
    print("#D79 claim postconditions (failed to load):", False)
if not all(_D79):
    import sys as _sysD79
    print("#D79 live checks: FAILED", file=_sysD79.stderr)
    _sysD79.exit(1)
