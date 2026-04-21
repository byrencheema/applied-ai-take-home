"""Run one question and print the full message trajectory for debugging.

Usage:
    uv run trace "your question here"
"""
import sys
import textwrap

from dotenv import load_dotenv

from ..agent import build_agent

load_dotenv()

TRUNC = 600


def short(s: str, n: int = TRUNC) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + f"… [+{len(s)-n} chars]"


def indent(s: str, n: int = 4) -> str:
    return textwrap.indent(s, " " * n)


def main() -> None:
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("need a question")
        sys.exit(1)

    agent = build_agent()
    cfg = {"configurable": {"thread_id": "trace"}}
    out = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)

    step = 0
    for m in out["messages"]:
        step += 1
        kind = m.__class__.__name__
        name = getattr(m, "name", None)
        tool_calls = getattr(m, "tool_calls", None) or []

        if kind == "HumanMessage":
            print(f"\n[{step}] HUMAN")
            print(indent(short(m.content)))
        elif kind == "AIMessage":
            print(f"\n[{step}] AI")
            if m.content:
                print(indent(short(m.content)))
            for tc in tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(indent(f"→ call: {tc['name']}({args})"))
        elif kind == "ToolMessage":
            print(f"\n[{step}] TOOL {name}")
            print(indent(short(m.content)))
        else:
            print(f"\n[{step}] {kind}")
            print(indent(short(str(m.content))))

    print("\n---")
    tool_count = sum(1 for m in out["messages"] if m.__class__.__name__ == "ToolMessage")
    ai_count = sum(1 for m in out["messages"] if m.__class__.__name__ == "AIMessage")
    print(f"total messages={len(out['messages'])}  ai_turns={ai_count}  tool_calls={tool_count}")


if __name__ == "__main__":
    main()
