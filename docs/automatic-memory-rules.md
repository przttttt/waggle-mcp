# Automatic Memory Rules

Use this rule text in MCP clients that support agent instructions, user rules, or custom instructions.

This is the recommended setup for:
- Codex global/project instructions
- Antigravity User Rules

`waggle-mcp init` writes this rule into a managed `AGENTS.md` block automatically when run from a Codex workspace.

## Rule Text

```text
Use Waggle automatically for conversational memory.

At the start of a new session, if project, agent, or session scope is known, call prime_context.

Before answering questions that may depend on prior decisions, preferences, constraints, project state, or earlier conversation context, call query_graph with the narrowest relevant scope.

After completed turns that contain durable information such as decisions, preferences, constraints, requirements, user corrections, project facts, or meaningful task outcomes, call observe_conversation automatically.

Do not ask the user to trigger Waggle manually. Use it in the background when relevant.
```

## Why This Exists

Registering Waggle in `mcp_config.json` or `config.toml` only makes the tools available.
It does not force the client to use them automatically.

For generic MCP clients today, portable automatic memory requires both:
- MCP server registration
- agent-level instructions telling the model when to call Waggle

## Important Notes

- Use the same `WAGGLE_DB_PATH` across clients if you want Codex and Antigravity to share one local memory graph.
- Keep scope stable across sessions, especially `project`, or recall will look empty even when memory exists.
- `ingest-transcript-handoff` is for rollover/session-handoff ingestion and export.
- Normal live conversational memory still depends on automatic `prime_context`, `query_graph`, and `observe_conversation` usage during chats.
