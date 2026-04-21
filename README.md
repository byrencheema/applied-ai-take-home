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
uv run slack-agent              # Slack bot (Socket Mode) — the production entry
uv run ask                      # local REPL, no Slack — for prompt iteration
uv run ask "question"           # one-shot, prints the final answer
uv run trace "question"         # prints the full message trajectory — for debugging
uv run python -m evals.eval     # eval suite, writes to evals/runs/ (gitignored)
uv run pytest                   # unit tests
```

In Slack: DM the bot, open an Assistant thread with it, or `@`-mention it in a
channel. Follow-ups in the same thread keep context.

## Slack app

`manifest.json` at the repo root is the reference app manifest. Paste it into
Slack > your app > App Manifest to recreate the app. You need a bot token
(`xoxb-`) and an app-level token (`xapp-`, `connections:write` scope) for
Socket Mode.

## Layout

```
src/
  agent.py            plan → research → answer LangGraph + prompts
  tools.py            FTS5 search, artifact fetch, customer enumerate/profile
  db.py               sqlite connection helper
  slack/              Bolt surface
    __init__.py         build_slack_app() — wires events to runner
    runner.py           placeholder → status → final answer flow
    status.py           in-place status message
    blocks.py           Block Kit builders
    format.py           string helpers and tool-call labels
  cli/                entry points (see pyproject.toml [project.scripts])
    slack.py            uv run slack-agent
    ask.py              uv run ask
    trace.py            uv run trace
evals/                evaluation harness (cases + runner; runs/ is gitignored)
tests/                unit tests
data/                 runtime state (feedback.jsonl) — gitignored
docs/                 take-home instructions + progress log
```
