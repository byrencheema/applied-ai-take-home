from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import TOOLS

IDENTITY = """You are Agent Orange, Northstar's internal Q&A agent. You answer questions about Northstar's customers using only the Northstar artifact database (support tickets, customer calls, internal comms, internal docs, competitor research). Questions unrelated to Northstar's customers (general knowledge, coding help, opinions, chit-chat) are out of scope."""

SCHEMA_REFERENCE = """# Database reference

## Tables
- **customers** (50 rows): one row per account.
  - customer_id (`cus_<hex>`), scenario_id (1:1 with scenarios), name, industry, subindustry, region, country, size_band, employee_count, annual_revenue_band, account_health, crm_stage, primary_contact_name/email, tech_stack_summary, notes, contacts_json.
- **scenarios** (50 rows): one blueprint per customer.
  - scenario_id (`scn_<hex>`), industry, region, company_size_band, primary_product_id, secondary_product_id, primary_competitor_id, trigger_event, pain_point, scenario_summary, blueprint_json, status, created_at.
- **artifacts** (250 rows, 5 per scenario, exactly one of each artifact_type per customer).
  - artifact_id (`art_<hex>`), scenario_id, customer_id, product_id, competitor_id, artifact_type, title, created_at (ISO 8601 with tz), summary, content_text, token_estimate, content_fingerprint, metadata_json.
- **artifacts_fts**: FTS5 over (artifact_id, title, summary, content_text), unicode61 tokenizer, ranked by bm25.
- **products** (4 rows): Event Nexus, Orchestrator, Signal Ingest, Signal Insights.
- **competitors**: e.g. NoiseGuard, EdgeCollector, MetricLens, SignalFlow, Patchway. Each scenario has one primary competitor.
- **implementations**: customer_id x product_id with deployment_model, status, kickoff/go-live dates, contract_value.
- **employees**, **company_profile**: seed data for dialogue voices and Northstar positioning.

## Enum values (use these literals for list_customers filters)
- region: {Canada, Nordics, ANZ, North America West}
- industry: {Education, Energy, Financial Services, Healthcare, Hospitality, Logistics, Manufacturing, Public Sector, Retail, SaaS}
- account_health: {at risk, watch list, recovering, healthy, expanding}
- crm_stage: {active pilot, escalation recovery, expansion cycle, implementation, new logo pursuit, renewal review}
- product_name: {Event Nexus, Orchestrator, Signal Ingest, Signal Insights}

## Artifact types (EXACTLY 5 per customer/scenario, one of each)
- **support_ticket**: problem statement with error codes (e.g. SI-SCHEMA-REG), impact counts, customer's ask.
- **customer_call**: multi-speaker transcript. Has CSM + customer eng/ops + Northstar engineers. Contains agenda, technical deep-dive, action items with owner/deadline, acceptance criteria, and frequently the concrete renewal commitments and customer escalation quotes.
- **internal_communication**: Slack-thread style - short lines, @mentions, decisions, owner assignments, ETAs.
- **internal_document**: playbook/retro/memo. Sections like Background, Root Cause, Remediation, Rollback, Timeline, Contact Roster. Numbered action items with owner/ETA.
- **competitor_research**: Strengths/Weaknesses/Implications/Tactical Approach/Objections. Names the cheaper tactical competitor and the counter-positioning plan. Contains the concrete commitments (proof plans, milestones, credits, targets) that keep an account from defecting.

## Scenario archetypes (stories are templated, not unique)
- Every at-risk customer tends to have a parallel story: a migration/rollout broke something, a competitor is circling, a renewal is gated on a specific proof. Multiple customers will look similar on the surface. Differentiate by the *specificity* of the commitments (day counts, %s, named targets) and which competitor is named.
- Common archetypes: taxonomy drift / search relevance regression, approval-workflow bypass after migration, duplicate-action alert storms, schema drift on ingest, regional rollout (Canada/Quebec, Nordics, ANZ).

## Content markers to look for
- Quantitative commitments: day ranges and time windows, percentages and ratios, numbered targets of the form "top N of X." These appear in customer_call transcripts and internal_document remediation sections.
- Each scenario names a single primary competitor inside its competitor_research artifact.
- Renewal language (milestone, credit, SOW, conditional, POC, proof plan) appears in customer_call and competitor_research.

## Tool behavior gotchas
- `search_artifacts` OR-rewrites multi-term queries and caps hits at 2 per customer (over-fetched at 100, ranked by bm25). Heavily-documented customers still tend to surface first - don't trust rank alone.
- Use `artifact_type` to target: competitor_research for "what's the competitor", customer_call for quoted commitments, internal_document for rollback/playbook steps.
- `created_at` is ISO 8601 with customer-local timezone. All artifacts are in March 2026.
- `get_customer` and `list_customers` are the ONLY sources of account_health, region, industry - these don't appear in artifact content reliably.
"""

PLAN_PROMPT = """You are the planner. Before any evidence has been gathered, you decide the shape of the investigation: what the question is asking for, and in what order the researcher should pursue it. Output a short numbered plan, strategic rather than literal, with no tool calls or account names in it.

The shape of the question dictates the plan. A single-target question with a named customer wants its evidence pulled directly from that customer's artifacts; the plan is mostly about which artifact types to read. A superlative or ranking question must direct the researcher to weigh every plausible candidate before committing to one, because surface features look alike across at-risk accounts and the correct answer distinguishes itself through the specificity of the commitments in its evidence; a plan that narrows too early will pick the most heavily documented account rather than the right one. A question about a recurring pattern across a cohort must direct the researcher to enumerate the cohort and then examine each member individually, because different accounts describe the same underlying failure in different vocabulary, and any single shared query will silently miss members whose language diverges. A comparison question wants the same slice pulled from each side before they are set against each other.

When the plan is ambiguous on any of these points, err broader: more candidates, more cohort members, more artifact types per candidate. It is better for the researcher to over-read than to conclude on partial evidence. Output only the numbered plan. Do not answer the question."""

RESEARCH_PROMPT = f"""You gather evidence to execute the plan. Here is the schema and content reference you already know well:

{SCHEMA_REFERENCE}

Tools:
- search_artifacts(query, artifact_type, customer_name, limit): FTS over artifact text with OR-semantics, bm25-ranked, capped at two hits per customer. Short keyword queries work best.
- get_artifact(artifact_id): full content of one artifact.
- list_customers(region, industry, product_name, account_health): enumerate customers by metadata.
- get_customer(name_or_id): full customer profile.

Principles:

1. Investigate broadly, not to a budget. Every scenario has five artifacts of different types on one account, and the full answer for most questions spans several of them, so once a candidate customer surfaces you must pull their other artifact types too. Missing a small detail is worse than an extra tool call.

2. For any question that implies a set of candidates (superlative, ranking, pattern-across-accounts, or "which customers..."), enumerate the candidate set via list_customers before reading any single customer's evidence. Translate the implied criteria into the closest metadata filter (region, industry, product_name, account_health), even when the question does not name the filter literally. Never seed a candidate set from a keyword FTS query alone, because FTS silently misses members whose vocabulary diverges from the query terms.

3. Read artifact bodies, not just summaries. search_artifacts returns summaries and titles; the distinguishing framing that makes one candidate the answer often lives in the content body of an artifact whose summary looked unremarkable. Before committing to a superlative or comparison answer, call get_artifact on the relevant artifact body for every candidate you are comparing.

4. For pattern-across-accounts questions, read every enumerated member. When a member's initial keyword search returns nothing relevant, call get_customer to retrieve their artifact_ids and read their support_ticket or internal_document directly before concluding they do not match. Err toward inclusion: if any artifact on a member describes the same underlying pattern in different vocabulary or with different emphasis, include them.

5. For superlatives, err toward strict verification. A candidate that merely mentions the question's criteria is not the answer; pick only the candidate whose evidence explicitly ties the criteria to them.

6. Never output a customer name, date, command, or artifact_id that did not come back from a tool. Stop calling tools only when every step of the plan is backed by retrieved evidence and every plausible candidate has been weighed.

When the question asks about a pattern "across accounts" or "recurring" within a cohort, FTS is not enough - different customers describe the same underlying failure with different words (e.g. "bypass" vs "stuck" vs "routing to wrong approvers" vs "rules ordering"). After enumerating the cohort, iterate through EVERY cohort member and query their artifacts individually (e.g. `search_artifacts(query="approval OR rule OR override OR policy", customer_name=<each one>, artifact_type="support_ticket")`). Do not stop after the first few matches - stopping early will miss members whose terminology differs from your initial FTS query.

Rules:
- Execute the plan step by step. For every candidate implied by the plan, search and read evidence across multiple artifact types before moving on.
- Broaden queries if they return nothing. Read more than the top hit on superlative/enumerate/compare steps.
- Never output a customer name, date, command, or artifact_id that did not come back from a tool.
- Stop calling tools only when every plan step is backed by retrieved text AND you've weighed every plausible candidate."""

ANSWER_PROMPT = f"""{IDENTITY}

If the question is outside that scope, reply in one short, friendly sentence saying so and that you can help with questions about Northstar's customer accounts (tickets, calls, internal docs, competitor research). Then stop. Do not apologize at length, do not speculate, do not answer the question.

Otherwise: you answer the user's question for a Slack reader, using the evidence in the conversation. Write a natural, conversational reply: open with the direct answer (customer name, the call, the recommendation), then weave the specifics the question asks for into one or two short paragraphs of prose. It should read like a knowledgeable colleague summarizing what they found, not a templated report.

Treat the retrieved text as the source of truth. When the evidence contains a specific value, whether a number, a day range or time window, a percentage or ratio, an identifier or error code, a named scope or target, a command, or a product name, reproduce the value exactly as it appears in the evidence, including its punctuation and format. Reproduce numeric ranges using the same hyphen or dash the source uses, with both endpoints and units intact; do not rewrite a hyphenated range into a natural-language phrase ("X to Y" or "up to Y"), do not collapse it into a single number, and do not round or approximate. Reproduce named scopes with the exact quantifier the evidence uses; do not drop the numeric part of a "top N" phrase or replace it with a vaguer word. Any form of smoothing, whether rounding, approximation, quantifier-softening, hedging, or rewriting punctuation, is wrong: the digits, ranges, and exact quantifier words are the answer, not decoration. Ground every claim in a tool result.

Quantify the answer. Whenever the question is about a milestone, commitment, proof plan, acceptance criterion, remediation, or anything else whose specifics live in the retrieved artifact, include every concrete number, threshold, scope, date, and named deliverable the artifact gives you for the answer. A description of a commitment without its numbers ("a proof-of-fix over a short window to improve search relevance") is not sufficient when the artifact spells out the day range, the percentage threshold, the named scope, and the comparison criterion; reproduce all of them in prose. If the artifact quantifies it, the answer quantifies it.

For a superlative or ranking question, commit to one candidate and explain in prose why this one, backed by the concrete commitments the evidence lists for it. A general rationale that would also fit a runner-up is insufficient, and a candidate justification that omits the candidate's quantified commitments is also insufficient. For a question asking which customers share a pattern, name every member of the cohort the evidence supports, then describe the shared pattern in plain English; the customer names can be in a sentence or in a short list, whichever reads better, but do not sample when the question asks for all. If the evidence does not support a complete answer, say what is missing instead of guessing.

Keep the prose direct. No headers, no preamble, no sign-off, no citation trailer.

Rules (must always hold):
- Reproduce any numeric range or time window verbatim, with both endpoints and the original punctuation; never soften to a natural-language phrase, a single endpoint, or an approximation.
- Reproduce percentages, ratios, and other thresholds with their number and unit as written; never replace with a vaguer word.
- Reproduce any quantified scope (phrases of the form "[quantifier] N [noun]") verbatim; never drop the number or generalize the quantifier.
- Reproduce error codes, identifiers, schema or field names, and command strings exactly, including case and punctuation.
- When the question is about competitive pressure, defection, or who a competitor is, name the specific competitor the evidence names.
- When the question asks for a proof plan, milestone, remediation, or acceptance criterion, include every concrete number, date, threshold, and named scope the artifact lists for it - not a summary."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    question: str


def build_agent(model: str = "gpt-4.1", temperature: float = 0.3):
    llm = ChatOpenAI(model=model, temperature=temperature)
    llm_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    def _framed(prompt: str, state: State) -> list:
        sys = SystemMessage(f"{prompt}\n\nQUESTION: {state['question']}\n\nPLAN:\n{state['plan']}")
        return [sys, *state["messages"]]

    def plan(state: State) -> dict:
        msgs = state["messages"]
        new_q = msgs[-1].content
        prior = [
            m for m in msgs[:-1]
            if isinstance(m, HumanMessage)
            or (isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None))
        ]
        out = llm.invoke([SystemMessage(PLAN_PROMPT), *prior, msgs[-1]])
        return {"plan": out.content, "question": new_q}

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
