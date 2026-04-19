"""Live test — hits the OpenAI API. Skipped if OPENAI_API_KEY is not set."""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-REPLACE"),
    reason="OPENAI_API_KEY not set",
)


def test_blueharbor_proof_plan():
    from src.agent import build_agent

    agent = build_agent()
    q = (
        "Which customer's issue started after the 2026-02-20 taxonomy rollout, "
        "and what proof plan did we propose to get them comfortable with renewal?"
    )
    out = agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={"configurable": {"thread_id": "test-blueharbor"}},
    )
    answer = out["messages"][-1].content.lower()
    assert "blueharbor" in answer
    assert "7-10" in answer or "7 to 10" in answer
