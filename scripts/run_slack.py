"""Run the Slack bot in Socket Mode.

Usage:
    uv run python -m scripts.run_slack

Requires SLACK_BOT_TOKEN (xoxb-) and SLACK_APP_TOKEN (xapp-) in .env.
"""
import logging
import os

from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode import SocketModeHandler

from src.slack_app import build_slack_app

load_dotenv()

logging.basicConfig(level=logging.INFO)


def main() -> None:
    app = build_slack_app()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
