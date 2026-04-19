# Northstar Q&A Slack bot

A LangGraph agent over a 50-customer SQLite knowledge base (tickets, calls,
internal docs, competitor research), wrapped as a Slack bot using the Assistants
& AI Apps surface.

- `src/agent.py` — plan → research → answer graph with tool use + checkpointing.
- `src/tools.py` — FTS5 search, artifact fetch, customer enumerate/profile.
- `src/db.py` — thread-local read-only SQLite.
- `src/slack_app.py` — Bolt app: assistant thread, `@`-mention, and DM handlers.
- `scripts/run_slack.py` — Socket Mode entry point.
- `scripts/ask.py` — local REPL (no Slack required).
- `scripts/eval.py` + `evals/cases.py` — 7-case eval harness.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env   # then fill in the tokens
```

Tokens needed in `.env`:

- `OPENAI_API_KEY` — any key with access to `gpt-4.1`.
- `SLACK_BOT_TOKEN` — bot token from your Slack app (`xoxb-…`).
- `SLACK_APP_TOKEN` — app-level token with `connections:write` (`xapp-…`),
  required for Socket Mode.

The SQLite database ships in `applied-ai-take-home-database/` and is opened
read-only, so nothing else needs provisioning.

## Running the Slack bot

```bash
uv run python -m scripts.run_slack
```

Then in Slack:

- DM the bot, or open an **Assistant** thread with it — you'll see four seeded
  starter prompts (lifted from the eval cases).
- `@`-mention the bot in any channel you've added it to.

While the agent is working you'll see live status updates
(`Planning the approach…` → `Calling search_artifacts…` →
`Reading evidence (3 artifact(s))…`). The final answer lands in-thread with
thumbs-up / thumbs-down feedback buttons.

Follow-ups in the same thread carry context — the LangGraph checkpointer is
keyed off `channel:thread_ts`.

## Provisioning the Slack app

`manifest.json` at the repo root is the reference app manifest. Paste it into
Slack → your app → **App Manifest** to recreate the app. It enables:

- Socket Mode + Interactivity
- Bot events: `app_mention`, `assistant_thread_started`, `message.*`
- Assistant view with the four suggested prompts
- Scopes: `chat:write`, `app_mentions:read`, `assistant:write`,
  `channels:history`, `im:history`, `mpim:history`, `groups:history`,
  `im:write`, `users:read`

## Trying it without Slack

```bash
uv run python -m scripts.ask                 # REPL
uv run python -m scripts.ask "what's up with BlueHarbor?"   # one-shot
uv run python -m scripts.eval                # 7-case eval
uv run pytest                                # unit tests
```

## Architecture notes

- The Slack layer is intentionally thin — one `build_agent()` at module load,
  one `_answer()` helper, handlers route messages into it. No new abstractions
  over the existing agent.
- Progress streams via `stream_mode="updates"` on the LangGraph graph: each
  node completion maps to one `set_status(…)` call. Typical run fires 5–15
  events, well below Slack's soft rate limit.
- `InMemorySaver` means checkpoints drop on restart — fine for the demo.
  Swap to `SqliteSaver` if you need persistence across restarts.
- `parallel_tool_calls=False` is set in `agent.py`, so tool calls are
  serialized and `Calling <tool>…` statuses make sense one at a time.
