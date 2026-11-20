# D79 (one her, not twins): live checks to append to ~/bench/live_checks.py on the Jetson.
# Each line prints "#D79 ...: True|False"; on an image without D79 the imports fail and every line is False.
try:
    from aerys_v2.config import Settings as _SD79
    from aerys_v2.factory import action_allowlist_for as _alD79
    from aerys_v2.service import text_one_loop_eligible as _eD79, text_one_loop_flag as _fD79

    _sD79 = _SD79()
    _ownD79 = str(_sD79.owner_person_id or "")
    _allowD79 = _alD79(_sD79)
    _dmD79 = {"user_id": _ownD79, "privacy_context": "private", "platform": "discord", "channel_kind": "dm"}
    _armedD79 = _fD79(_sD79)()
    print("#D79 the one-loop switch reads (it starts off; on only after his go line):",
          isinstance(_armedD79, bool))
    print("#D79 only a household DM is eligible: his DM yes; a guild room, a stranger, voice and Kael's lane no:",
          bool(_ownD79) and _eD79(_dmD79, _allowD79, _ownD79)
          and not _eD79({**_dmD79, "channel_kind": "guild"}, _allowD79, _ownD79)
          and not _eD79({**_dmD79, "user_id": "discord:1"}, _allowD79, _ownD79)
          and not _eD79({**_dmD79, "voice": True}, _allowD79, _ownD79)
          and not _eD79({**_dmD79, "platform": "kael"}, _allowD79, _ownD79))
except Exception as _e:
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
        "thread_id": "person:u", "identity": _idD79, "specialist": False, "text_one_loop": True}})
    _qD79, _, _ = _buildD79([_HD79(content="lights off")], {"configurable": {
        "thread_id": "person:u", "identity": _idD79, "specialist": True}})
    print("#D79 a one-loop turn gets her soul, the typed banter voice and full memory; a specialist turn does not:",
          _pD79[0].content.startswith("SOUL") and _facD79.TEXT_BANTER_OVERLAY in _pD79[0].content
          and "FULL MEMORY" in _pD79[0].content and not _specD79
          and _facD79.TEXT_BANTER_OVERLAY not in _qD79[0].content and "PROFILE ONLY" in _qD79[0].content)
except Exception as _e:
    print("#D79 one-loop prompt (failed to load):", False)
try:
    from aerys_v2.service import _claims_effect_without_doing_it as _cD79, _one_loop_check as _okD79
    from langchain_core.messages import AIMessage as _AD79, ToolMessage as _TD79

    _rD79 = "I’ve started a home job to update the MapTap skill, Chris."
    _readD79 = [_AD79(content="", tool_calls=[{"id": "c", "name": "list_async_tasks", "args": {}}]),
                _TD79(content="none running", name="list_async_tasks", tool_call_id="c"), _AD79(content=_rD79)]
    print("#D79 the 10:55 wording is caught, and a read alone does not back a claimed start:",
          _cD79("you are good to update your skill", _rD79) and _okD79(_rD79, _readD79) is not None)
except Exception as _e:
    print("#D79 claim postcondition (failed to load):", False)
