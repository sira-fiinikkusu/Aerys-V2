# Aerys-V2

A ground-up rebuild of Aerys's agent runtime — migrating the reasoning and orchestration layer out of n8n into a Python **LangGraph "Brain,"** while n8n is refactored into a deterministic **"Hands"** layer (identity resolution, memory pipeline, output routing, credential-touching operations) exposed as REST endpoints.

Built incrementally, in public, as a learning-forward evolution of Aerys V1.

## Status

🚧 Early scaffold. Architecture cross-reviewed (Codex + Gemini, May 2026); phase scoping in progress.

## Architecture (target)

```
Discord / Telegram / Voice ──→  Hands (n8n, refactored)  ←──  Brain (Python / LangGraph)
                                - identity resolution         - agent loop
                                - memory pipeline             - model-tier routing
                                - output routing + PII        - tool orchestration
                                - credential-touching ops     - persona (soul) + reflection
                                  (exposed as REST)
                                                          ←──  Voice (TypeScript, later)
                                                               - sub-4s streaming surface
```

The boundary rule: **Brain may ask, Hands decides.** One canonical owner per concept; no shared business logic across runtimes.

## Secrets

No secrets live in this repo. Tokens, credentials, and channel IDs are loaded from a local `.env` (gitignored). Public repo ≠ public secrets — every credential stays in ignored files by design.

## License

TBD
