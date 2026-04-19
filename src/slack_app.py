"""Slack wrapper around the LangGraph Q&A agent.

One module-level agent, three handlers:
- assistant_thread_started: seed suggested prompts for the Assistants surface.
- app_mention: answer when @-mentioned in a channel.
- message (DMs + assistant threads): answer in a DM or an assistant thread.

Progress is streamed via set_status (the "Thinking..." shimmer in the Assistant
surface) as LangGraph nodes complete. The final answer is posted with say().
"""
import json
import logging
import os
import re
from typing import Any

from slack_bolt import App, Assistant, SetStatus, SetSuggestedPrompts, Say
from slack_sdk import WebClient

from .agent import build_agent

logger = logging.getLogger(__name__)

AGENT = build_agent()

SUGGESTED_PROMPTS = [
    {
        "title": "BlueHarbor proof plan",
        "message": (
            "Which customer's issue started after the 2026-02-20 taxonomy rollout, "
            "and what proof plan did we propose to get them comfortable with renewal?"
        ),
    },
    {
        "title": "Verdant Bay rollback",
        "message": (
            "For Verdant Bay, what's the approved live patch window, "
            "and exactly how do we roll back if the validation checks fail?"
        ),
    },
    {
        "title": "NA West cohort",
        "message": (
            "Among the North America West Event Nexus accounts, which ones are really "
            "dealing with taxonomy/search semantics problems versus duplicate-action problems?"
        ),
    },
    {
        "title": "Canada bypass pattern",
        "message": (
            "Do we have a recurring Canada approval-bypass pattern across accounts, "
            "or is MapleBridge basically a one-off?"
        ),
    },
]

MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _thread_key(channel: str, thread_ts: str) -> str:
    return f"{channel}:{thread_ts}"


def _clean_text(text: str) -> str:
    return MENTION_RE.sub("", text or "").strip()


def _run_agent(question: str, thread_key: str, set_status: SetStatus | None = None) -> str:
    """Drive the LangGraph agent, nudging set_status as nodes complete.

    stream_mode="updates" fires once per node; we map each to a user-visible status.
    Typical run: plan → research (→ tools → research)* → answer, ~5-15 events.
    """
    cfg = {"configurable": {"thread_id": thread_key}}
    inputs = {"messages": [{"role": "user", "content": question}]}
    tool_count = 0
    final: str | None = None

    for chunk in AGENT.stream(inputs, config=cfg, stream_mode="updates"):
        node = next(iter(chunk))
        if node == "plan" and set_status:
            set_status("Planning the approach...")
        elif node == "research" and set_status:
            last = chunk["research"]["messages"][-1]
            if getattr(last, "tool_calls", None):
                call = last.tool_calls[0]
                set_status(f"Calling {call['name']}...")
            else:
                set_status("Drafting findings...")
        elif node == "tools":
            tool_count += len(chunk["tools"]["messages"])
            if set_status:
                set_status(f"Reading evidence ({tool_count} artifact(s))...")
        elif node == "answer":
            final = chunk["answer"]["messages"][-1].content

    return final or "Sorry — I couldn't produce an answer."


def _feedback_blocks(thread_key: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "actions",
            "block_id": "feedback",
            "elements": [
                {
                    "type": "button",
                    "action_id": "feedback_up",
                    "text": {"type": "plain_text", "text": "👍 Helpful"},
                    "value": thread_key,
                },
                {
                    "type": "button",
                    "action_id": "feedback_down",
                    "text": {"type": "plain_text", "text": "👎 Not helpful"},
                    "value": thread_key,
                },
            ],
        }
    ]


def _answer(
    *,
    question: str,
    channel: str,
    thread_ts: str,
    say: Say,
    client: WebClient,
    set_status: SetStatus | None = None,
) -> None:
    key = _thread_key(channel, thread_ts)
    try:
        answer = _run_agent(question, key, set_status=set_status)
    except Exception:
        logger.exception("agent error for thread %s", key)
        say(text="Sorry — something broke while I was researching that.", thread_ts=thread_ts)
        return

    say(
        text=answer,
        thread_ts=thread_ts,
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": answer}},
            *_feedback_blocks(key),
        ],
    )


def build_slack_app() -> App:
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    assistant = Assistant()

    @assistant.thread_started
    def on_thread_started(say: Say, set_suggested_prompts: SetSuggestedPrompts) -> None:
        say("What can I help you dig up on Northstar customers?")
        set_suggested_prompts(prompts=SUGGESTED_PROMPTS)

    @assistant.user_message
    def on_user_message(
        payload: dict[str, Any],
        client: WebClient,
        say: Say,
        set_status: SetStatus,
    ) -> None:
        channel = payload["channel"]
        thread_ts = payload.get("thread_ts") or payload["ts"]
        _answer(
            question=_clean_text(payload.get("text", "")),
            channel=channel,
            thread_ts=thread_ts,
            say=say,
            client=client,
            set_status=set_status,
        )

    app.use(assistant)

    @app.event("app_mention")
    def on_mention(event: dict[str, Any], say: Say, client: WebClient) -> None:
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        text = _clean_text(event.get("text", ""))
        if not text:
            say(text="Ask me anything about Northstar customers.", thread_ts=thread_ts)
            return
        _answer(question=text, channel=channel, thread_ts=thread_ts, say=say, client=client)

    @app.event("message")
    def on_message(event: dict[str, Any]) -> None:
        # Assistant.user_message already handles assistant-thread DMs. This
        # catch-all just swallows other message events so Bolt doesn't warn.
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

    @app.action(re.compile(r"feedback_(up|down)"))
    def on_feedback(ack, body: dict[str, Any]) -> None:
        ack()
        action = body["actions"][0]
        logger.info("feedback %s for %s by %s", action["action_id"], action["value"], body["user"]["id"])

    @app.error
    def on_error(error: Exception, body: dict[str, Any]) -> None:
        logger.exception("bolt error: %s\n%s", error, json.dumps(body)[:500])

    return app
