"""Suite-wide: her house's names are always generic here (house_names.py). The real ones live only
in the private .env; a test never sees them, and the prompt snapshots render from these."""
import json
import os

GENERIC_HOUSE_NAMES = {"wife": "Robin", "friends": "Alex, Jordan, Casey, Riley", "friend": "Alex",
                       "colleague": "Riley", "aside": "Morgan"}
os.environ["AERYS_HOUSE_NAMES"] = json.dumps(GENERIC_HOUSE_NAMES)
