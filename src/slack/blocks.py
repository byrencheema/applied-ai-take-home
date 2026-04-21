"""Slack Block Kit builders: answer sections, feedback buttons, status placeholder."""
from typing import Any

PLACEHOLDER_TEXT = "Thinking…"
SECTION_LIMIT = 2900


def section_blocks(text: str) -> list[dict[str, Any]]:
    """Split text into section blocks under Slack's 3000-char section limit."""
    chunks: list[str] = []
    remaining = text
    while len(remaining) > SECTION_LIMIT:
        cut = remaining.rfind("\n", 0, SECTION_LIMIT)
        if cut == -1:
            cut = remaining.rfind(" ", 0, SECTION_LIMIT)
        if cut == -1:
            cut = SECTION_LIMIT
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return [{"type": "section", "text": {"type": "mrkdwn", "text": c}} for c in chunks]


def feedback_blocks(thread_key: str) -> list[dict[str, Any]]:
    """Slack-native feedback buttons (context_actions + feedback_buttons)."""
    return [
        {
            "type": "context_actions",
            "block_id": "agent_feedback",
            "elements": [
                {
                    "type": "feedback_buttons",
                    "action_id": "agent_feedback",
                    "positive_button": {
                        "text": {"type": "plain_text", "text": "Good", "emoji": True},
                        "value": f"up::{thread_key}",
                        "accessibility_label": "Mark response as helpful",
                    },
                    "negative_button": {
                        "text": {"type": "plain_text", "text": "Bad", "emoji": True},
                        "value": f"down::{thread_key}",
                        "accessibility_label": "Mark response as unhelpful",
                    },
                }
            ],
        }
    ]


def status_blocks(text: str) -> list[dict[str, Any]]:
    """Italic placeholder message that updates in place."""
    return [{"type": "section", "text": {"type": "mrkdwn", "text": f"_{text}_"}}]
