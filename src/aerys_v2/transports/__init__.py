"""Transports — thin adapters between the outside world and ask().

n8n mapping: this package replaces the adapter workflows (02-01 guild, 03-03 DM,
02-02 Telegram, ...). Each transport does exactly three jobs: receive a platform
event, normalize it, call ask(), deliver the reply. No routing, no memory, no
model logic — that all lives behind the ask() seam so every channel gets it
identically.
"""
