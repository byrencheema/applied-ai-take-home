"""Drive the LangGraph agent from a Slack event: placeholder → status → final answer."""
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from slack_sdk import WebClient

from ..agent import build_agent
from .blocks import feedback_blocks
from .format import (
    describe_tool_call,
    summarize_tool_calls,
    summarize_tool_result,
    thread_key,
)
from .status import StatusMessage

logger = logging.getLogger(__name__)

AGENT = build_agent()
FEEDBACK_PATH = Path("data/feedback.jsonl")


def run_agent(question: str, key: str, status_msg: StatusMessage | None) -> str:
    """Drive the LangGraph agent; log every node and tool call; advance status."""
    cfg = {"configurable": {"thread_id": key}}
    inputs = {"messages": [{"role": "user", "content": question}]}
    tool_count = 0
    final: str | None = None
    logger.info("[%s] question: %s", key, question)
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
                        key, len(plan_text.splitlines()), first_line[:120])
            if cleaned:
                preview = cleaned if len(cleaned) <= 140 else cleaned[:139] + "…"
                set_status(f"Plan: {preview}")

        elif node == "research":
            for m in msgs:
                if isinstance(m, AIMessage):
                    calls = m.tool_calls or []
                    if calls:
                        logger.info("[%s] research tool_calls: %s",
                                    key, " | ".join(summarize_tool_calls(m)))
                        for call in calls:
                            set_status(describe_tool_call(call))
                    elif m.content:
                        preview = m.content[:120].replace("\n", " ")
                        logger.info("[%s] research no_tools preview=%s", key, preview)

        elif node == "tools":
            tool_count += len(msgs)
            for m in msgs:
                if isinstance(m, ToolMessage):
                    logger.info("[%s]   tool_result %s: %s",
                                key,
                                getattr(m, "name", "tool"),
                                summarize_tool_result(m))

        elif node == "answer":
            set_status("Writing the answer…")
            if msgs:
                final = msgs[-1].content
                logger.info("[%s] answer produced (%d chars)", key, len(final or ""))

    elapsed = time.perf_counter() - t0
    in_tok, out_tok = token_totals(key)
    logger.info(
        "[%s] done in %.1fs, %d tool call(s), %d in / %d out tokens, %d chars",
        key, elapsed, tool_count, in_tok, out_tok, len(final or ""),
    )
    return final or "Sorry, I couldn't produce an answer."


def token_totals(key: str) -> tuple[int, int]:
    """Sum usage_metadata across AIMessages in the checkpointed state."""
    state = AGENT.get_state({"configurable": {"thread_id": key}})
    in_tok = out_tok = 0
    for m in state.values.get("messages", []):
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
            in_tok += m.usage_metadata.get("input_tokens", 0)
            out_tok += m.usage_metadata.get("output_tokens", 0)
    return in_tok, out_tok


def set_assistant_title(client: WebClient, channel: str, thread_ts: str, question: str) -> None:
    title = question.strip().splitlines()[0] if question.strip() else "Agent Orange"
    if len(title) > 200:
        title = title[:197] + "..."
    try:
        client.assistant_threads_setTitle(
            channel_id=channel, thread_ts=thread_ts, title=title,
        )
    except Exception:
        logger.debug("assistant setTitle failed", exc_info=True)


def react(client: WebClient, channel: str, ts: str, name: str, add: bool) -> None:
    try:
        if add:
            client.reactions_add(channel=channel, timestamp=ts, name=name)
        else:
            client.reactions_remove(channel=channel, timestamp=ts, name=name)
    except Exception as e:
        logger.debug("reaction %s %s failed: %s", "add" if add else "remove", name, e)


def answer(
    *,
    question: str,
    channel: str,
    thread_ts: str,
    client: WebClient,
    react_ts: str | None = None,
) -> None:
    """Post a placeholder, run the agent, update the placeholder in place."""
    key = thread_key(channel, thread_ts)
    if react_ts:
        react(client, channel, react_ts, "eyes", add=True)

    status_msg = StatusMessage(client, channel, thread_ts)

    try:
        result = run_agent(question, key, status_msg=status_msg)
    except Exception:
        logger.exception("[%s] agent error", key)
        status_msg.finalize("Sorry, something broke while I was researching that.", [])
        if react_ts:
            react(client, channel, react_ts, "eyes", add=False)
        return

    status_msg.finalize(result, feedback_blocks(key))
    if react_ts:
        react(client, channel, react_ts, "eyes", add=False)


def record_feedback(user_id: str, key: str, vote: str) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user": user_id,
        "thread_key": key,
        "vote": vote,
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("feedback %s · thread=%s user=%s", vote, key, user_id)


def thread_has_state(key: str) -> bool:
    state = AGENT.get_state({"configurable": {"thread_id": key}})
    return bool(state.values.get("messages"))
