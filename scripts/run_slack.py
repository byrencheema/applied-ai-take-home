"""Run the Slack bot in Socket Mode.

Usage:
    uv run python -m scripts.run_slack

Requires SLACK_BOT_TOKEN (xoxb-) and SLACK_APP_TOKEN (xapp-) in .env.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from slack_bolt.adapter.socket_mode import SocketModeHandler

from src.slack_app import build_slack_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def main() -> None:
    app = build_slack_app()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
