"""Run the Slack bot in Socket Mode.

Usage:
    uv run slack-agent

Requires SLACK_BOT_TOKEN (xoxb-) and SLACK_APP_TOKEN (xapp-) in .env.
"""
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from slack_bolt.adapter.socket_mode import SocketModeHandler

from ..slack import build_slack_app

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
LEVEL_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;41m",
}
NAME_COLOR = "\033[35m"      # magenta


class ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        level = record.levelname
        name = record.name
        msg = record.getMessage()
        if self.use_color:
            color = LEVEL_COLORS.get(level, "")
            line = (
                f"{DIM}{ts}{RESET} "
                f"{color}{level:<7}{RESET} "
                f"{NAME_COLOR}{name}{RESET} {BOLD}|{RESET} {msg}"
            )
        else:
            line = f"{ts} {level:<7} {name} | {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


use_color = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(use_color))
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def main() -> None:
    app = build_slack_app()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
