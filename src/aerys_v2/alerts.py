"""AlertSink — the one place bad news goes.

n8n mapping: the Central Error Handler workflow (audit_log insert + #echoes post),
except it covers EVERYTHING wired through it (the n8n one was attached to 19 of 33
workflows) and it can never take the brain down: any failure inside the sink itself
degrades to a log line. Alerting about an error must not become the error.

Transport: the existing kael-dm webhook on the Jetson (same one the backup pipeline
uses) — no new infra. stdlib urllib so this adds zero dependencies.
"""

import json
import logging
import urllib.request

log = logging.getLogger("aerys_v2.alerts")


class AlertSink:
    """Fire-and-forget operator alerts with log fallback.

    webhook_url=None (e.g. tests, dev without the LAN) = log-only mode; every alert
    still lands in the journal, nothing raises either way.
    """

    def __init__(self, webhook_url: str | None, *, timeout_s: float = 10.0) -> None:
        self.webhook_url = webhook_url
        self.timeout_s = timeout_s

    def alert(self, message: str, *, source: str = "brain") -> bool:
        """Send one alert. Returns True if the webhook accepted it (False = logged only)."""
        text = f"🚨 **aerys-v2 {source}**: {message}"
        log.error("ALERT [%s] %s", source, message)
        if not self.webhook_url:
            return False
        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps({"text": text[:1900]}).encode(),  # Discord cap headroom
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode() or "{}")
            # The webhook can 200 with message_id=null on oversize — treat as failure
            # so we never believe an alert landed when it didn't (backup-script lesson).
            ok = bool(body.get("message_id"))
            if not ok:
                log.error("alert webhook accepted but message_id null — not delivered")
            return ok
        except Exception as e:  # noqa: BLE001 — the sink must swallow everything
            log.error("alert webhook failed (%s) — logged only", e)
            return False
