#!/usr/bin/env python3
"""Daily capability-gaps digest -> #aerys-debug (out-of-session, via cron).

The self-iteration loop's hourly MINER already runs as the `aerys-gaps` container
(APScheduler, --restart unless-stopped). This is the once-a-day SURFACING nudge the
owner asked for: read the mined gaps and, if any are open, post a fenced digest to
#aerys-debug as the Aerys bot so the review ("pick one to build") happens ~1x/day.

Read-only: the gaps reader never writes; this script only reads + posts. Cron-safe:
absolute docker path, stdlib only, no python-dotenv dependency. The bot token is read
from the deploy .env at run time and never stored here; the channel id is not secret.

Usage (manual test):
    /usr/bin/python3 /home/sira/aerys-brain-src/scripts/gaps_digest.py

Cron (installed 2026-07-09, Jetson `sira` user crontab):
    30 13 * * *  ->  13:30 UTC / 09:30 EDT daily
"""
import json
import subprocess
import sys
import urllib.request

ENV_PATH = "/home/sira/aerys-brain-src/.env"
CHANNEL_ID = "1480365115684688127"  # #aerys-debug (not secret)
DOCKER = "/usr/bin/docker"
IMAGE = "aerys-brain:latest"
DISCORD_API = "https://discord.com/api/v10"
DIGEST_CAP = 1700  # leave room under Discord's 2000-char cap for the fence + header


def env_value(name: str) -> str | None:
    """Pull one KEY=value out of the deploy .env (no python-dotenv dependency)."""
    try:
        with open(ENV_PATH) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def read_gaps() -> str:
    """Run the owner /gaps reader inside the image (same DB env); return its digest."""
    out = subprocess.run(
        [DOCKER, "run", "--rm", "--network", "host", "--env-file", ENV_PATH,
         "--entrypoint", "python", IMAGE, "-m", "aerys_v2.workers", "gaps",
         "--status", "open"],  # only surface OPEN gaps — built/resolved ones drop off
        capture_output=True, text=True, timeout=180,
    )
    return out.stdout.strip()


def has_gaps(digest: str) -> bool:
    """format_gaps prints '  #<id> [..]' rows; the empty case ends with '(none)'."""
    return any(ln.lstrip().startswith("#") for ln in digest.splitlines())


def post_to_discord(token: str, content: str) -> int:
    body = json.dumps({"content": content}).encode()
    req = urllib.request.Request(
        f"{DISCORD_API}/channels/{CHANNEL_ID}/messages",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            # Discord's edge (Cloudflare) 1010-blocks requests with no User-Agent.
            "User-Agent": "aerys-gaps-digest (https://github.com/sira-fiinikkusu/Aerys-V2, 1.0)",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def main() -> int:
    token = env_value("DISCORD_BOT_TOKEN")
    if not token:
        print("gaps-digest: no DISCORD_BOT_TOKEN in .env", file=sys.stderr)
        return 1
    digest = read_gaps()
    if not has_gaps(digest):
        print("gaps-digest: no open gaps — nothing to post")
        return 0
    content = (
        "**\U0001f527 Daily capability-gaps digest** — review with `/gaps`, then flag one to Kael to build.\n"
        "```\n" + digest[:DIGEST_CAP] + "\n```"
    )
    status = post_to_discord(token, content)
    print(f"gaps-digest: posted to #aerys-debug (HTTP {status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
