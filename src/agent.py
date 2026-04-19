from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import TOOLS

SCHEMA_REFERENCE = """# Database reference

## Tables
- **customers** (50 rows): one row per account.
  - customer_id (`cus_<hex>`), scenario_id (1:1 with scenarios), name, industry, subindustry, region, country, size_band, employee_count, annual_revenue_band, account_health, crm_stage, primary_contact_name/email, tech_stack_summary, notes, contacts_json.
- **scenarios** (50 rows): one blueprint per customer.
  - scenario_id (`scn_<hex>`), industry, region, company_size_band, primary_product_id, secondary_product_id, primary_competitor_id, trigger_event, pain_point, scenario_summary, blueprint_json, status, created_at.
- **artifacts** (250 rows, 5 per scenario — exactly one of each artifact_type per customer).
  - artifact_id (`art_<hex>`), scenario_id, customer_id, product_id, competitor_id, artifact_type, title, created_at (ISO 8601 with tz), summary, content_text, token_estimate, content_fingerprint, metadata_json.
- **artifacts_fts**: FTS5 over (artifact_id, title, summary, content_text), unicode61 tokenizer, ranked by bm25.
- **products** (4 rows): Event Nexus, Orchestrator, Signal Ingest, Signal Insights.
- **competitors**: e.g. NoiseGuard, EdgeCollector, MetricLens, SignalFlow, Patchway. Each scenario has one primary competitor.
- **implementations**: customer_id × product_id with deployment_model, status, kickoff/go-live dates, contract_value.
- **employees**, **company_profile**: seed data for dialogue voices and Northstar positioning.

## Enum values (use these literals for list_customers filters)
- region: {Canada, Nordics, ANZ, North America West}
- industry: {Education, Energy, Financial Services, Healthcare, Hospitality, Logistics, Manufacturing, Public Sector, Retail, SaaS}
- account_health: {at risk, watch list, recovering, healthy, expanding}
- crm_stage: {active pilot, escalation recovery, expansion cycle, implementation, new logo pursuit, renewal review}
- product_name: {Event Nexus, Orchestrator, Signal Ingest, Signal Insights}

## Artifact types — EXACTLY 5 per customer/scenario, one of each
- **support_ticket**: problem statement with error codes (e.g. SI-SCHEMA-REG), impact counts, customer's ask.
- **customer_call**: multi-speaker transcript. Has CSM + customer eng/ops + Northstar engineers. Contains agenda, technical deep-dive, action items with owner/deadline, acceptance criteria, and frequently the concrete renewal commitments and customer escalation quotes.
- **internal_communication**: Slack-thread style — short lines, @mentions, decisions, owner assignments, ETAs.
- **internal_document**: playbook/retro/memo. Sections like Background, Root Cause, Remediation, Rollback, Timeline, Contact Roster. Numbered action items with owner/ETA.
- **competitor_research**: Strengths/Weaknesses/Implications/Tactical Approach/Objections. Names the cheaper/tactical competitor and the counter-positioning plan. This is where "what's the milestone that keeps them from defecting" usually lives.

## Scenario archetypes (stories are templated, not unique)
- Every at-risk customer tends to have a parallel story: a migration/rollout broke something, a competitor is circling, a renewal is gated on a specific proof. Multiple customers will look similar on the surface. Differentiate by the *specificity* of the commitments (day counts, %s, named targets) and which competitor is named.
- Common archetypes: taxonomy drift / search relevance regression, approval-workflow bypass after migration, duplicate-action alert storms, schema drift on ingest, regional rollout (Canada/Quebec, Nordics, ANZ).

## Content markers to look for
- Day counts: "7-10 business days", "72-hour window", "six weeks".
- Percentages: "80% accuracy", "40% reduction", "10% ACV credit".
- Named targets: "top 20 saved searches", "top 10 customers", "per-collector mTLS certs".
- Named competitors: NoiseGuard, EdgeCollector, MetricLens, SignalFlow, Patchway.
- Renewal language: "renewal", "milestone", "credit", "SOW", "conditional", "POC", "proof plan".

## Tool behavior gotchas
- `search_artifacts` OR-rewrites multi-term queries and caps hits at 2 per customer (over-fetched at 100, ranked by bm25). Heavily-documented customers still tend to surface first — don't trust rank alone.
- Use `artifact_type` to target: competitor_research for "what's the competitor", customer_call for quoted commitments, internal_document for rollback/playbook steps.
- `created_at` is ISO 8601 with customer-local timezone. All artifacts are in March 2026.
- `get_customer` and `list_customers` are the ONLY sources of account_health, region, industry — these don't appear in artifact content reliably.
"""

PLAN_PROMPT = """You are the planner. Look at the type of the user's question and come up with a good plan for how to search — implicitly, not explicitly. Sketch the strategic ordering: first we should do this, then we should do this, then we should do this. Don't name specific tool calls or customer names. Just the shape of the approach.

Recognize the query type and let that dictate the plan:
- Single-target lookup ("for customer X, what is Y")
- Superlative / ranking ("which customer is MOST X")
- Enumeration / pattern ("which customers have Y", "is there a recurring X across accounts") — the plan must direct the researcher to check EACH cohort member's artifacts individually, because customers often describe the same underlying pattern with different words; a single aggregate FTS search will miss members whose terminology differs.
- Comparison ("compare X across …")

Output only a short numbered plan. Do not answer the question."""

RESEARCH_PROMPT = f"""You gather evidence to execute the plan below. You know this data deeply — here is the schema and content reference:

{SCHEMA_REFERENCE}

Tools:
- search_artifacts(query, artifact_type, customer_name, limit): FTS (OR-semantics, bm25, per-customer capped). Use short keyword queries (1-3 terms).
- get_artifact(artifact_id): full content of one artifact.
- list_customers(region, industry, product_name, account_health): enumerate customers by metadata.
- get_customer(name_or_id): full customer profile.

You take a broad approach. You know that if you miss a small detail, it's bad news — so you check every outcome and weigh every option. Every scenario has exactly 5 artifacts (one of each type), so once you've surfaced a candidate customer, pull the other artifact types on that customer too — the full answer almost always spans multiple artifact types on the same account.

When the question asks about a pattern "across accounts" or "recurring" within a cohort, FTS is not enough — different customers describe the same underlying failure with different words (e.g. "bypass" vs "stuck" vs "routing to wrong approvers" vs "rules ordering"). After enumerating the cohort, iterate through EVERY cohort member and query their artifacts individually (e.g. `search_artifacts(query="approval OR rule OR override OR policy", customer_name=<each one>, artifact_type="support_ticket")`). Do not stop after the first few matches — stopping early will miss members whose terminology differs from your initial FTS query.

Rules:
- Execute the plan step by step. For every candidate implied by the plan, search and read evidence across multiple artifact types before moving on.
- Broaden queries if they return nothing. Read more than the top hit on superlative/enumerate/compare steps.
- Never output a customer name, date, command, or artifact_id that did not come back from a tool.
- Stop calling tools only when every plan step is backed by retrieved text AND you've weighed every plausible candidate."""

ANSWER_PROMPT = """Produce the final answer to the user's question, using the plan and the retrieved evidence in the conversation.

- Ground every claim in the retrieved tool results.
- Cite the artifact_id(s) you used.
- Name the specific Northstar product(s) involved when they appear in evidence (Event Nexus, Orchestrator, Signal Ingest, Signal Insights).
- For superlatives, pick ONE best candidate and justify with concrete evidence.
- For enumerations, list every matching customer and state the shared pattern in plain English.
- If the evidence is incomplete, say exactly what is missing rather than guessing."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    question: str


def build_agent(model: str = "gpt-4.1", temperature: float = 0.5):
    llm = ChatOpenAI(model=model, temperature=temperature)
    llm_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    def _framed(prompt: str, state: State) -> list:
        sys = SystemMessage(f"{prompt}\n\nQUESTION: {state['question']}\n\nPLAN:\n{state['plan']}")
        return [sys, *state["messages"]]

    def plan(state: State) -> dict:
        q = state["messages"][-1].content
        out = llm.invoke([SystemMessage(PLAN_PROMPT), HumanMessage(q)])
        return {"plan": out.content, "question": q}

    def research(state: State) -> dict:
        return {"messages": [llm_tools.invoke(_framed(RESEARCH_PROMPT, state))]}

    def answer(state: State) -> dict:
        return {"messages": [llm.invoke(_framed(ANSWER_PROMPT, state))]}

    def route(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "answer"

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("answer", answer)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_conditional_edges("research", route, {"tools": "tools", "answer": "answer"})
    g.add_edge("tools", "research")
    g.add_edge("answer", END)
    return g.compile(checkpointer=InMemorySaver())
