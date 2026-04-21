"""Slack wrapper around the LangGraph Q&A agent.

Handlers:
- assistant_thread_started: seed suggested prompts on the Assistants surface.
- assistant.user_message: drives the Assistants-panel surface (set_title + set_status).
- app_mention: answer when @-mentioned in a channel; adds an :eyes: reaction.
- message (DMs + channel thread replies): answer without requiring a mention.

UX: as soon as a question lands we post a placeholder message ("_Agent Orange is
thinking..._"), then chat.update it with status as the graph progresses ("is
researching the evidence..."), and finally replace it in place with the answer
plus feedback buttons. In the Assistants panel we additionally drive
assistant.threads.setStatus (shimmer + rotating loading_messages) and setTitle.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from slack_bolt import App, Assistant, SetSuggestedPrompts, Say
from slack_sdk import WebClient
from slackify_markdown import slackify_markdown

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

PLACEHOLDER_TEXT = "Thinking…"


def _thread_key(channel: str, thread_ts: str) -> str:
    return f"{channel}:{thread_ts}"


def _clean_text(text: str) -> str:
    return MENTION_RE.sub("", text or "").strip()


def _describe_tool_call(call: dict[str, Any]) -> str:
    """Human-readable label for a tool call, with its key argument inlined."""
    name = call.get("name", "")
    args = call.get("args", {}) or {}

    def _clip(s: Any, n: int = 50) -> str:
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    if name == "search_artifacts":
        bits = []
        if q := args.get("query"):
            bits.append(f'"{_clip(q, 30)}"')
        if t := args.get("artifact_type"):
            bits.append(f"type={t}")
        if c := args.get("customer_name"):
            bits.append(f"customer={_clip(c, 24)}")
        return f"Searching artifacts: {', '.join(bits)}" if bits else "Searching artifacts"
    if name == "get_artifact":
        return "Reading an artifact"
    if name == "list_customers":
        flt = ", ".join(f"{k}={v}" for k, v in args.items() if v)
        return f"Listing customers ({flt})" if flt else "Listing customers"
    if name == "get_customer":
        return f"Looking up customer {_clip(args.get('name_or_id', ''), 40)}"
    return f"{name}(…)"


def _summarize_tool_calls(msg: AIMessage) -> list[str]:
    """Render each tool call as `name(k=v,...)` for a log line."""
    out = []
    for call in getattr(msg, "tool_calls", None) or []:
        name = call.get("name", "?")
        args = call.get("args", {}) or {}
        parts = []
        for k, v in args.items():
            s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if len(s) > 60:
                s = s[:57] + "..."
            parts.append(f"{k}={s}")
        out.append(f"{name}({', '.join(parts)})")
    return out


def _summarize_tool_result(msg: ToolMessage) -> str:
    """One-line preview of a tool response content."""
    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
    content = content.replace("\n", " ").strip()
    if len(content) > 180:
        content = content[:177] + "..."
    return content


SECTION_LIMIT = 2900


def _section_blocks(text: str) -> list[dict[str, Any]]:
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


def _feedback_blocks(thread_key: str) -> list[dict[str, Any]]:
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


def _status_blocks(text: str) -> list[dict[str, Any]]:
    """Italic placeholder message that updates in place."""
    return [{"type": "section", "text": {"type": "mrkdwn", "text": f"_{text}_"}}]


class _StatusMessage:
    """Posts a placeholder and updates the same message in place as the graph runs.

    chat.update failures are swallowed; the final update (with the answer) is
    attempted unconditionally."""

    def __init__(
        self,
        client: WebClient,
        channel: str,
        thread_ts: str,
    ) -> None:
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
                blocks=_status_blocks(PLACEHOLDER_TEXT),
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
                blocks=_status_blocks(text),
            )
        except Exception:
            logger.debug("chat.update (status) failed", exc_info=True)

    def finalize(self, answer_text: str, feedback_blocks: list[dict[str, Any]]) -> None:
        formatted = slackify_markdown(answer_text)
        blocks = [*_section_blocks(formatted), *feedback_blocks]
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


def _run_agent(
    question: str,
    thread_key: str,
    status_msg: _StatusMessage | None,
) -> str:
    """Drive the LangGraph agent; log every node and tool call; advance status."""
    cfg = {"configurable": {"thread_id": thread_key}}
    inputs = {"messages": [{"role": "user", "content": question}]}
    tool_count = 0
    final: str | None = None
    logger.info("[%s] question: %s", thread_key, question)
    t0 = time.perf_counter()

    def set_status(text: str) -> None:
        if status_msg:
            status_msg.set(text)

    set_status("Planning the investigation…")

    for mode, payload in AGENT.stream(inputs, config=cfg, stream_mode=["updates"]):
        if mode != "updates":
            continue
        node, update = next(iter(payload.items()))
        msgs = update.get("messages", []) if isinstance(update, dict) else []

        if node == "plan":
            plan_text = (update.get("plan", "") or "").strip()
            first_line = plan_text.splitlines()[0] if plan_text else ""
            cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", first_line).strip()
            logger.info("[%s] plan complete (%d lines) | %s",
                        thread_key, len(plan_text.splitlines()), first_line[:120])
            if cleaned:
                preview = cleaned if len(cleaned) <= 140 else cleaned[:139] + "…"
                set_status(f"Plan: {preview}")

        elif node == "research":
            for m in msgs:
                if isinstance(m, AIMessage):
                    calls = m.tool_calls or []
                    if calls:
                        logger.info("[%s] research tool_calls: %s",
                                    thread_key,
                                    " | ".join(_summarize_tool_calls(m)))
                        for call in calls:
                            set_status(_describe_tool_call(call))
                    elif m.content:
                        preview = m.content[:120].replace("\n", " ")
                        logger.info("[%s] research no_tools preview=%s",
                                    thread_key, preview)

        elif node == "tools":
            tool_count += len(msgs)
            for m in msgs:
                if isinstance(m, ToolMessage):
                    logger.info("[%s]   tool_result %s: %s",
                                thread_key,
                                getattr(m, "name", "tool"),
                                _summarize_tool_result(m))

        elif node == "answer":
            set_status("Writing the answer…")
            if msgs:
                final = msgs[-1].content
                logger.info("[%s] answer produced (%d chars)",
                            thread_key, len(final or ""))

    elapsed = time.perf_counter() - t0
    in_tok, out_tok = _token_totals(thread_key)
    logger.info(
        "[%s] done in %.1fs, %d tool call(s), %d in / %d out tokens, %d chars",
        thread_key, elapsed, tool_count, in_tok, out_tok, len(final or ""),
    )
    return final or "Sorry, I couldn't produce an answer."


def _token_totals(thread_key: str) -> tuple[int, int]:
    """Sum usage_metadata across AIMessages in the checkpointed state."""
    state = AGENT.get_state({"configurable": {"thread_id": thread_key}})
    in_tok = out_tok = 0
    for m in state.values.get("messages", []):
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
            in_tok += m.usage_metadata.get("input_tokens", 0)
            out_tok += m.usage_metadata.get("output_tokens", 0)
    return in_tok, out_tok


def _set_assistant_title(client: WebClient, channel: str, thread_ts: str, question: str) -> None:
    title = question.strip().splitlines()[0] if question.strip() else "Agent Orange"
    if len(title) > 200:
        title = title[:197] + "..."
    try:
        client.assistant_threads_setTitle(
            channel_id=channel, thread_ts=thread_ts, title=title,
        )
    except Exception:
        logger.debug("assistant setTitle failed", exc_info=True)


def _react(client: WebClient, channel: str, ts: str, name: str, add: bool) -> None:
    try:
        if add:
            client.reactions_add(channel=channel, timestamp=ts, name=name)
        else:
            client.reactions_remove(channel=channel, timestamp=ts, name=name)
    except Exception as e:
        logger.debug("reaction %s %s failed: %s", "add" if add else "remove", name, e)


def _answer(
    *,
    question: str,
    channel: str,
    thread_ts: str,
    client: WebClient,
    react_ts: str | None = None,
) -> None:
    """Post a placeholder, run the agent, update the placeholder in place."""
    key = _thread_key(channel, thread_ts)
    if react_ts:
        _react(client, channel, react_ts, "eyes", add=True)

    status_msg = _StatusMessage(client, channel, thread_ts)

    try:
        answer = _run_agent(question, key, status_msg=status_msg)
    except Exception:
        logger.exception("[%s] agent error", key)
        status_msg.finalize(
            "Sorry, something broke while I was researching that.",
            [],
        )
        if react_ts:
            _react(client, channel, react_ts, "eyes", add=False)
        return

    status_msg.finalize(answer, _feedback_blocks(key))
    if react_ts:
        _react(client, channel, react_ts, "eyes", add=False)


def build_slack_app() -> App:
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    bot_user_id = app.client.auth_test()["user_id"]
    logger.info("bot user id: %s", bot_user_id)

    assistant = Assistant()

    @assistant.thread_started
    def on_thread_started(say: Say, set_suggested_prompts: SetSuggestedPrompts) -> None:
        logger.info("assistant thread started")
        say("What can I help you dig up on Northstar customers?")
        set_suggested_prompts(prompts=SUGGESTED_PROMPTS)

    @assistant.user_message
    def on_user_message(
        payload: dict[str, Any],
        client: WebClient,
    ) -> None:
        channel = payload["channel"]
        thread_ts = payload.get("thread_ts") or payload["ts"]
        question = _clean_text(payload.get("text", ""))
        logger.info("assistant msg · channel=%s thread=%s user=%s",
                    channel, thread_ts, payload.get("user"))
        _set_assistant_title(client, channel, thread_ts, question)
        _answer(
            question=question,
            channel=channel,
            thread_ts=thread_ts,
            client=client,
        )

    app.use(assistant)

    @app.event("app_mention")
    def on_mention(event: dict[str, Any], client: WebClient) -> None:
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        text = _clean_text(event.get("text", ""))
        logger.info("app_mention · channel=%s thread=%s user=%s text=%r",
                    channel, thread_ts, event.get("user"), text[:120])
        if not text:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Ask me anything about Northstar customers.",
            )
            return
        _answer(
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

        # Native 1:1 DM: answer every message, thread under the first one.
        if channel_type == "im":
            thread_ts = thread_ts or event["ts"]
            logger.info("dm · channel=%s thread=%s user=%s text=%r",
                        channel, thread_ts, event.get("user"), text[:120])
            _answer(
                question=_clean_text(text),
                channel=channel,
                thread_ts=thread_ts,
                client=client,
            )
            return

        # Channel thread replies: only answer if we have prior state for this thread.
        if not thread_ts:
            return
        key = _thread_key(channel, thread_ts)
        state = AGENT.get_state({"configurable": {"thread_id": key}})
        if not state.values.get("messages"):
            logger.debug("channel thread reply skip · no prior state for %s", key)
            return
        logger.info("thread reply · channel=%s thread=%s user=%s text=%r",
                    channel, thread_ts, event.get("user"), text[:120])
        _answer(
            question=_clean_text(text),
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
        vote, _, thread_key = value.partition("::")
        if vote not in ("up", "down") or not thread_key:
            logger.debug("feedback click with unexpected payload: %r", action)
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": body["user"]["id"],
            "thread_key": thread_key,
            "vote": vote,
        }
        path = Path("data/feedback.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("feedback %s · thread=%s user=%s",
                    vote, thread_key, body["user"]["id"])

    @app.error
    def on_error(error: Exception, body: dict[str, Any]) -> None:
        logger.exception("bolt error: %s\n%s", error, json.dumps(body)[:500])

    return app
