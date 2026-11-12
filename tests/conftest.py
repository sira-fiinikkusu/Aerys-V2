"""Use generic house names before any application import."""
import json
import os

GENERIC_HOUSE_NAMES = {"wife": "Robin", "friends": "Alex, Jordan, Casey, Riley", "friend": "Alex",
                       "colleague": "Riley", "aside": "Morgan"}
os.environ["AERYS_HOUSE_NAMES"] = json.dumps(GENERIC_HOUSE_NAMES)
