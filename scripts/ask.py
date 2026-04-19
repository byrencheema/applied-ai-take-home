"""Tiny REPL to chat with the agent. Reuses one thread_id so multi-turn works.

Usage:
    uv run python -m scripts.ask
    uv run python -m scripts.ask "one-shot question"
"""
import sys

from dotenv import load_dotenv

from src.agent import build_agent

load_dotenv()


def main() -> None:
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "dev"}}

    one_shot = " ".join(sys.argv[1:]).strip() or None
    if one_shot:
        out = agent.invoke({"messages": [{"role": "user", "content": one_shot}]}, config=cfg)
        print(out["messages"][-1].content)
        return

    print("Ask the bot anything. Ctrl-C to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        out = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)
        print(out["messages"][-1].content)


if __name__ == "__main__":
    main()
