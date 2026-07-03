#!/usr/bin/env python3
"""Eyeball-parity check: run the read-only services against the LIVE aerys DB.

MANUAL TOOL — never run by CI or an agent. Compares Python service output against
what the n8n workflows (03-01 Identity Resolver, 04-03 Profile API, 04-02 Memory
Retrieval) return for the same inputs. The connection is forced read-only at the
session level, so even a bug here cannot write.

Usage:
    DATABASE_URL='postgresql://sira:<pw>@192.168.1.231:5432/aerys' \\
    OPENROUTER_API_KEY='sk-or-...' \\
    uv run python scripts/parity_check.py \\
        --platform discord --platform-user-id <discord_snowflake> \\
        --person-id <uuid> --query "what do you remember about coffee?" \\
        [--privacy-context private]

OPENROUTER_API_KEY is only needed for the memory leg (it embeds --query the same
way the n8n workflow does); without it, memory retrieval is skipped.
"""

import argparse
import json
import os
import sys

import psycopg

from aerys_v2.services.identity import resolve_identity
from aerys_v2.services.memory import (
    format_memory_context,
    openrouter_embedder,
    retrieve_memories,
)
from aerys_v2.services.profile import get_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="discord")
    parser.add_argument("--platform-user-id", help="platform account id to resolve")
    parser.add_argument("--person-id", help="person uuid for profile + memory")
    parser.add_argument("--query", default="hello", help="text to embed for memory retrieval")
    parser.add_argument("--privacy-context", default="public", choices=["public", "private"])
    args = parser.parse_args()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set — refusing to guess a connection string.", file=sys.stderr)
        return 1

    with psycopg.connect(dsn) as conn:
        conn.read_only = True  # session-level write ban — SELECTs only, enforced by Postgres

        if args.platform_user_id:
            print(f"\n=== resolve_identity({args.platform!r}, {args.platform_user_id!r}) ===")
            print(json.dumps(resolve_identity(conn, args.platform, args.platform_user_id), indent=2))
        else:
            print("\n(no --platform-user-id — skipping identity)")

        if args.person_id:
            print(f"\n=== get_profile({args.person_id!r}, {args.privacy_context!r}) ===")
            print(json.dumps(get_profile(conn, args.person_id, args.privacy_context), indent=2))

            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                print(f"\n=== retrieve_memories(query={args.query!r}, {args.privacy_context!r}) ===")
                rows = retrieve_memories(
                    conn,
                    args.person_id,
                    query_text=args.query,
                    embed=openrouter_embedder(api_key),
                    privacy_context=args.privacy_context,
                )
                for row in rows:
                    print(f"  {row['combined_score']:.4f}  {row['content']!r}  [{row['source_platform']}]")
                print("\n--- memory_context block (compare vs n8n Format Memory Context) ---")
                print(format_memory_context(rows) or "(empty)")
            else:
                print("\n(OPENROUTER_API_KEY not set — skipping memory retrieval)")
        else:
            print("(no --person-id — skipping profile + memory)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
