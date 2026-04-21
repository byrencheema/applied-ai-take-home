"""Pure string helpers: identifiers, label rendering, log summaries."""
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def thread_key(channel: str, thread_ts: str) -> str:
    return f"{channel}:{thread_ts}"


def clean_text(text: str) -> str:
    return MENTION_RE.sub("", text or "").strip()


def describe_tool_call(call: dict[str, Any]) -> str:
    """Human-readable status label for a tool call, with its key argument inlined."""
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


def summarize_tool_calls(msg: AIMessage) -> list[str]:
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


def summarize_tool_result(msg: ToolMessage) -> str:
    """One-line preview of a tool response content."""
    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
    content = content.replace("\n", " ").strip()
    if len(content) > 180:
        content = content[:177] + "..."
    return content
