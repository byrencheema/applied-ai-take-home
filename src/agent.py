from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

from .tools import get_artifact, get_customer, list_customers, search_artifacts

ALL_TOOLS = [search_artifacts, get_artifact, list_customers, get_customer]

PLAN_SYSTEM = """You are the planner. Given the user's question, write a short numbered plan for answering it.

- Single-target question ("for customer X, what is Y") → one or two steps.
- Superlative ("which customer is MOST X") → (a) enumerate plausible candidates, (b) check the criterion per candidate, (c) compare and pick.
- Enumeration ("which customers have Y", "compare across …") → list every candidate in the cohort and mark what to check for each.

Use list_customers (region/industry/product/health) when the cohort is metadata-based (e.g. "Canada accounts", "NA West Event Nexus customers").
Use search_artifacts with short keyword queries when the cohort is content-based (e.g. "customers evaluating NoiseGuard", "accounts with duplicate-action issues") to enumerate every plausible candidate.
Do NOT read full artifacts and do NOT answer the question — the researcher does that. Output only the numbered plan."""

RESEARCH_SYSTEM = """You gather evidence to execute the plan below.

Tools:
- search_artifacts(query, artifact_type, customer_name, limit): FTS (OR-semantics, bm25, per-customer capped). Use short keyword queries (1-3 terms).
- get_artifact(artifact_id): full content of one artifact.
- list_customers(region, industry, product_name, account_health): enumerate customers.
- get_customer(name_or_id): full customer profile.

Rules:
- Execute the plan step by step. For every candidate the plan names, search and read evidence.
- Broaden queries if they return nothing; read more than the top hit on compare/enumerate steps.
- Never output a customer name, date, command, or artifact_id that did not come back from a tool.
- Stop calling tools only when every plan step is backed by retrieved text."""

SYNTHESIZE_SYSTEM = """Produce the final answer to the user's question, using the plan and the retrieved evidence in the conversation.

- Ground every claim in the retrieved tool results.
- Cite the artifact_id(s) you used.
- For superlatives, pick ONE best candidate and justify with concrete evidence.
- For enumerations, list every matching customer and state the shared pattern in plain English.
- If the evidence is incomplete, say exactly what is missing rather than guessing."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    question: str


def build_agent(model: str = "gpt-4.1"):
    llm = ChatOpenAI(model=model, temperature=0)
    llm_tools = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    plan_agent = create_react_agent(
        model=llm, tools=[list_customers, search_artifacts], prompt=PLAN_SYSTEM
    )

    def plan_node(state: State) -> dict:
        q = state["messages"][-1].content
        out = plan_agent.invoke({"messages": [{"role": "user", "content": q}]})
        return {"plan": out["messages"][-1].content, "question": q}

    def research_node(state: State) -> dict:
        sys = SystemMessage(
            f"{RESEARCH_SYSTEM}\n\nQUESTION: {state['question']}\n\nPLAN:\n{state['plan']}"
        )
        msgs = [sys, *state["messages"]]
        return {"messages": [llm_tools.invoke(msgs)]}

    def synthesize_node(state: State) -> dict:
        sys = SystemMessage(
            f"{SYNTHESIZE_SYSTEM}\n\nQUESTION: {state['question']}\n\nPLAN:\n{state['plan']}"
        )
        msgs = [sys, *state["messages"]]
        return {"messages": [llm.invoke(msgs)]}

    def route(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "synthesize"

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("research", research_node)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_conditional_edges("research", route, {"tools": "tools", "synthesize": "synthesize"})
    g.add_edge("tools", "research")
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=InMemorySaver())
