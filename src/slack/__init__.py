"""Slack surface: Bolt app factory that wires events to the agent runner.

Handlers:
- assistant.thread_started: seed suggested prompts on the Assistants surface.
- assistant.user_message: answer within an Assistants-panel thread (sets title).
- app_mention: answer when @-mentioned in a channel; adds an :eyes: reaction.
- message: answer in DMs and in channel threads that already have state.
- action agent_feedback: persist up/down votes to data/feedback.jsonl.
"""
import json
import logging
import os
from typing import Any

from slack_bolt import App, Assistant, SetSuggestedPrompts, Say
from slack_sdk import WebClient

from .format import clean_text, thread_key
from .runner import answer, record_feedback, set_assistant_title, thread_has_state

logger = logging.getLogger(__name__)

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


def _parse_allowed_users(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {u.strip() for u in raw.replace(",", " ").split() if u.strip()}


def build_slack_app() -> App:
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    bot_user_id = app.client.auth_test()["user_id"]
    logger.info("bot user id: %s", bot_user_id)

    allowed_users = _parse_allowed_users(os.environ.get("SLACK_ALLOWED_USERS"))
    if allowed_users:
        logger.info("user whitelist active (%d users)", len(allowed_users))
    else:
        logger.info("user whitelist empty — allowing all users")

    def is_allowed(user_id: str | None) -> bool:
        return not allowed_users or (user_id in allowed_users)

    assistant = Assistant()

    @assistant.thread_started
    def on_thread_started(say: Say, set_suggested_prompts: SetSuggestedPrompts) -> None:
        logger.info("assistant thread started")
        say("What can I help you dig up on Northstar customers?")
        set_suggested_prompts(prompts=SUGGESTED_PROMPTS)

    @assistant.user_message
    def on_user_message(payload: dict[str, Any], client: WebClient) -> None:
        channel = payload["channel"]
        thread_ts = payload.get("thread_ts") or payload["ts"]
        user = payload.get("user")
        question = clean_text(payload.get("text", ""))
        logger.info("assistant msg · channel=%s thread=%s user=%s",
                    channel, thread_ts, user)
        if not is_allowed(user):
            logger.info("blocked assistant msg · user=%s not in whitelist", user)
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Sorry, you're not authorized to use Agent Orange.",
            )
            return
        set_assistant_title(client, channel, thread_ts, question)
        answer(question=question, channel=channel, thread_ts=thread_ts, client=client)

    app.use(assistant)

    @app.event("app_mention")
    def on_mention(event: dict[str, Any], client: WebClient) -> None:
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        user = event.get("user")
        text = clean_text(event.get("text", ""))
        logger.info("app_mention · channel=%s thread=%s user=%s text=%r",
                    channel, thread_ts, user, text[:120])
        if not is_allowed(user):
            logger.info("blocked app_mention · user=%s not in whitelist", user)
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Sorry, you're not authorized to use Agent Orange.",
            )
            return
        if not text:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Ask me anything about Northstar customers.",
            )
            return
        answer(
            question=text,
            channel=channel,
            thread_ts=thread_ts,
            client=client,
            react_ts=event["ts"],
        )

    @app.event("message")
    def on_message(event: dict[str, Any], client: WebClient) -> None:
        subtype = event.get("subtype")
        channel_type = event.get("channel_type")
        if event.get("bot_id") or subtype:
            logger.debug("message skip · bot_id=%s subtype=%s",
                         event.get("bot_id"), subtype)
            return
        text = event.get("text", "")
        if f"<@{bot_user_id}>" in text:
            return  # app_mention handler owns this

        channel = event["channel"]
        thread_ts = event.get("thread_ts")
        user = event.get("user")

        if channel_type == "im":
            thread_ts = thread_ts or event["ts"]
            logger.info("dm · channel=%s thread=%s user=%s text=%r",
                        channel, thread_ts, user, text[:120])
            if not is_allowed(user):
                logger.info("blocked dm · user=%s not in whitelist", user)
                client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text="Sorry, you're not authorized to use Agent Orange.",
                )
                return
            answer(
                question=clean_text(text),
                channel=channel,
                thread_ts=thread_ts,
                client=client,
            )
            return

        if not thread_ts:
            return
        key = thread_key(channel, thread_ts)
        if not thread_has_state(key):
            logger.debug("channel thread reply skip · no prior state for %s", key)
            return
        logger.info("thread reply · channel=%s thread=%s user=%s text=%r",
                    channel, thread_ts, user, text[:120])
        if not is_allowed(user):
            logger.info("blocked thread reply · user=%s not in whitelist", user)
            return
        answer(
            question=clean_text(text),
            channel=channel,
            thread_ts=thread_ts,
            client=client,
            react_ts=event["ts"],
        )

    @app.action("agent_feedback")
    def on_feedback(ack, body: dict[str, Any]) -> None:
        ack()
        action = body["actions"][0]
        value = action.get("value", "") or action.get("selected_option", {}).get("value", "")
        vote, _, key = value.partition("::")
        if vote not in ("up", "down") or not key:
            logger.debug("feedback click with unexpected payload: %r", action)
            return
        record_feedback(body["user"]["id"], key, vote)

    @app.error
    def on_error(error: Exception, body: dict[str, Any]) -> None:
        logger.exception("bolt error: %s\n%s", error, json.dumps(body)[:500])

    return app
