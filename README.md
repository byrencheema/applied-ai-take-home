<p align="center">
  <img src="assets/agent-orange.jpg" alt="Agent Orange" width="160" />
</p>

<h1 align="center">Agent Orange</h1>

A Slack Q&A bot that answers questions grounded in a 50-customer SQLite knowledge base (support tickets, customer calls, internal comms, internal docs, competitor research). Built on LangGraph and slack-bolt.

- Design and reasoning: [DESIGN.md](DESIGN.md)
- Demo video: https://www.loom.com/share/a08b0c62c5324398ad0b78f79c8ff39f

### Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- A Slack workspace you can install an app into
- An OpenAI API key with `gpt-4.1` access

### Setup

1. Clone this repo, then clone the dataset repo next to the source so the SQLite file lives at `./applied-ai-take-home-database/synthetic_startup.sqlite`:
   ```bash
   git clone https://github.com/langchain-ai/applied-ai-take-home-database
   ```
2. Install dependencies:
   ```bash
   uv sync --extra dev
   ```
3. Copy the env template and fill it in:
   ```bash
   cp .env.example .env
   ```
   Required values:
   - `OPENAI_API_KEY`
   - `SLACK_BOT_TOKEN` (`xoxb-...`)
   - `SLACK_APP_TOKEN` (`xapp-...`, Socket Mode)

### Slack app setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App > From an app manifest**.
2. Pick your workspace, then paste the contents of [`manifest.json`](manifest.json) and create.
3. Under **Basic Information > App-Level Tokens**, generate a token with the `connections:write` scope. This is your `SLACK_APP_TOKEN` (`xapp-...`).
4. Under **Install App**, install to your workspace and copy the **Bot User OAuth Token**. This is your `SLACK_BOT_TOKEN` (`xoxb-...`).
5. Paste both into `.env` alongside `OPENAI_API_KEY`.
6. Start the bot with `uv run slack-agent`. Socket Mode means no public URL is needed.

### Running

```bash
uv run slack-agent              # Slack bot (Socket Mode), the production entry
uv run ask                      # local REPL, no Slack
uv run ask "question"           # one-shot, prints the final answer
uv run trace "question"         # full message trajectory for debugging
uv run python -m evals.eval     # eval suite, writes to evals/runs/
uv run pytest                   # unit tests
```

In Slack: DM the bot, open an Assistant thread with it, or `@`-mention it in a channel. Follow-ups in the same thread keep context.

### Layout

```
src/
  agent.py            plan, research, answer LangGraph + prompts
  tools.py            FTS5 search, artifact fetch, customer enumerate/profile
  db.py               sqlite connection helper
  slack/              Bolt surface
    __init__.py         build_slack_app() wires events to runner
    runner.py           placeholder, status, final answer flow
    status.py           in-place status message
    blocks.py           Block Kit builders
    format.py           string helpers and tool-call labels
  cli/                entry points (see pyproject.toml [project.scripts])
    slack.py            uv run slack-agent
    ask.py              uv run ask
    trace.py            uv run trace
evals/                evaluation harness (cases + runner; runs/ is gitignored)
tests/                unit tests
assets/               README image
data/                 runtime state (feedback.jsonl), gitignored
docs/                 take-home instructions and progress log
```
