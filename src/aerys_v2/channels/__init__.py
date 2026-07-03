"""Channel-facing helpers — the Output Router reborn as testable functions.

n8n mapping: in Aerys V1 the code that understood platform payloads (Discord guild
adapter, DM adapter, Telegram adapter, Output Router) lived inside Code nodes scattered
across workflows — untestable except by sending real messages. This package collects
that logic as PURE functions: no I/O, no network, no n8n sandbox. The transport layers
(Discord/Telegram clients, webhooks) call in; everything here is exercised by pytest.
"""
