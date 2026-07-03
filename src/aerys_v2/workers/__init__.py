"""Background workers — batch jobs that run beside the Brain, not inside a turn.

n8n mapping: these are the Schedule Trigger workflows (batch extraction, morning
brief, watchdogs) rebuilt as plain Python jobs. Each worker module exposes a
`run_*` function that takes injected connections/seams (offline-testable, like
every service), and `workers/__main__.py` is the process entrypoint —
`python -m aerys_v2.workers extraction --once` — destined for its own container.

Shadow-mode rule (write side of the writer lease): workers read prod aerys
READ-ONLY and write ONLY to aerys_v2 tables until the lease flips.
"""
