"""Suite-wide, two things (board #176):

1. Her house's names are always generic here (house_names.py). The real ones live only in the
   private .env; a test never sees them, and the prompt snapshots render from these. Set FIRST:
   house_names reads the environment when aerys_v2 is first imported (the switch below imports it).
2. Stage 1's switch: AERYS_TEST_ACTION_LOOP=agent runs every test that builds the action graph
   against the create_agent loop instead, with no edits to the tests. Test modules import
   build_action_graph by name, so the swap happens here, before they are collected."""
import json
import os

GENERIC_HOUSE_NAMES = {"wife": "Robin", "friends": "Alex, Jordan, Casey, Riley", "friend": "Alex",
                       "colleague": "Riley", "aside": "Morgan"}
os.environ["AERYS_HOUSE_NAMES"] = json.dumps(GENERIC_HOUSE_NAMES)

if os.environ.get("AERYS_TEST_ACTION_LOOP") == "agent":
    import aerys_v2.factory as _factory
    from aerys_v2.action_loop import build_agent_action_graph as _agent

    _factory._original_build_action_graph = _factory.build_action_graph   # parity tests' "old" side
    _factory.build_action_graph = _agent
