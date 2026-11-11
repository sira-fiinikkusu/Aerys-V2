# Messages as Aerys (s4-01)

This ships off. Apply `db/migrations/029_message_sends.sql` before enabling it. Configure
`MESSAGE_PEOPLE` as a comma-separated list of person UUIDs in the deployment environment;
no contact names, handles, or addresses belong in the repository. With the existing bot token,
people database, brain database, owner identity, and approval gate configured,
`MESSAGES_AS_HERSELF=1` enables the tool. Guest tool lists exclude it. Missing dependencies
leave it unavailable and produce an arming reason in the log.

`send_message(to, text, platform="discord")` sends from Aerys's bot and appends `\n-Aerys`
once. It refuses signed messages over 1990 characters rather than allowing the shared DM
helper to truncate them. Email uses `send_email`; Telegram remains parked.

D12's button shows the resolved person, platform, handle/address, and the entire signed
message. The tool resolves once and passes that snapshot to the gate. Namesakes remain
ambiguous even if only one is listed, or one lacks a Discord address. The MESSAGE watcher
executes the stored destination and words after approval. Confirmed destinations go directly
unless the turn read outside text; D28 also applies to messages addressed to the owner.

Migration 029 keeps person ID, platform, SHA-256 of the signed words, attempt time, and
`pending` / `ok` / `failed` status. It is separate from the successful-file ledger and the
owner's reach desk because neither represents per-person message attempts and uncertain
network outcomes. Both watcher and direct sends reserve here before contacting Discord.
A transaction advisory lock serializes reservations across workers. Completed attempts
suppress identical words to the same person/platform for ten minutes, including failed
attempts whose network outcome may be uncertain. Pending attempts suppress retries until
an operator reconciles them; they never expire automatically after a crash.

A missing ledger, failed reservation, or database commit failure prevents delivery. If Discord
accepts a message but the final ledger update fails, the response explains that it was accepted
and must not be resent; it does not claim a completed `Sent:` result. No automatic retry runs.
The switch also removes the MESSAGE executor when off, so queued approvals stay unexecuted.
