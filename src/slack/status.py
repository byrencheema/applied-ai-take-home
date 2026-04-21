"""Placeholder message that chat.updates in place as the graph progresses."""
import logging
from typing import Any

from slack_sdk import WebClient
from slackify_markdown import slackify_markdown

from .blocks import PLACEHOLDER_TEXT, section_blocks, status_blocks

logger = logging.getLogger(__name__)


class StatusMessage:
    """Posts a placeholder and updates the same message in place as the graph runs.

    chat.update failures are swallowed; the final update (with the answer) is
    attempted unconditionally, with a postMessage fallback if update fails."""

    def __init__(self, client: WebClient, channel: str, thread_ts: str) -> None:
        self.client = client
        self.channel = channel
        self.thread_ts = thread_ts
        self.ts: str | None = None
        self.last_text: str = PLACEHOLDER_TEXT
        self._post()

    def _post(self) -> None:
        try:
            resp = self.client.chat_postMessage(
                channel=self.channel,
                thread_ts=self.thread_ts,
                text=PLACEHOLDER_TEXT,
                blocks=status_blocks(PLACEHOLDER_TEXT),
            )
            self.ts = resp["ts"]
        except Exception:
            logger.exception("initial chat.postMessage failed")

    def set(self, text: str) -> None:
        text = text.strip()
        if not text or text == self.last_text or not self.ts:
            return
        self.last_text = text
        try:
            self.client.chat_update(
                channel=self.channel,
                ts=self.ts,
                text=text,
                blocks=status_blocks(text),
            )
        except Exception:
            logger.debug("chat.update (status) failed", exc_info=True)

    def finalize(self, answer_text: str, feedback_blocks: list[dict[str, Any]]) -> None:
        paragraphs = [p.strip() for p in answer_text.split("\n\n") if p.strip()] or [answer_text]
        formatted = "\n\n".join(slackify_markdown(p).rstrip() for p in paragraphs)
        blocks = [*section_blocks(formatted), *feedback_blocks]
        if self.ts:
            try:
                self.client.chat_update(
                    channel=self.channel,
                    ts=self.ts,
                    text=answer_text,
                    blocks=blocks,
                )
                return
            except Exception:
                logger.exception("final chat.update failed; posting fresh message")
        try:
            self.client.chat_postMessage(
                channel=self.channel,
                thread_ts=self.thread_ts,
                text=answer_text,
                blocks=blocks,
            )
        except Exception:
            logger.exception("fallback chat.postMessage also failed")
