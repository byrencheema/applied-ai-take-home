# Northstar Q&A Slack bot

A LangGraph agent over a 50-customer SQLite knowledge base, wrapped as a Slack
bot using the Assistants surface.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env   # then fill in the tokens
```

## Run

```bash
uv run python -m scripts.run_slack        # Slack bot (Socket Mode)
uv run python -m scripts.ask              # local REPL, no Slack
uv run python -m scripts.ask "question"   # one-shot
uv run python -m evals.eval               # eval suite (logs to evals/runs/)
uv run pytest                             # unit tests
```

In Slack: DM the bot, open an Assistant thread with it, or `@`-mention it in a
channel. Follow-ups in the same thread keep context.

## Slack app

`manifest.json` at the repo root is the reference app manifest. Paste it into
Slack > your app > App Manifest to recreate the app. You need a bot token
(`xoxb-`) and an app-level token (`xapp-`, `connections:write` scope) for
Socket Mode.

## Layout

- `src/agent.py`: plan, research, answer graph.
- `src/tools.py`: FTS5 search, artifact fetch, customer enumerate/profile.
- `src/slack_app.py`: Bolt handlers.
- `scripts/run_slack.py`: Socket Mode entry.
- `evals/cases.py` + `evals/eval.py`: eval harness. Runs written to `evals/runs/` (gitignored).
