"""Suite-wide switch for board #176 (Stage 1): AERYS_TEST_ACTION_LOOP=agent runs every
test that builds the action graph against the create_agent loop instead, with no edits
to the tests. Test modules import build_action_graph by name, so the swap happens here,
before they are collected."""

import os

if os.environ.get("AERYS_TEST_ACTION_LOOP") == "agent":
    import aerys_v2.factory as _factory
    from aerys_v2.action_loop import build_agent_action_graph as _agent

    _factory._original_build_action_graph = _factory.build_action_graph   # parity tests' "old" side
    _factory.build_action_graph = _agent
