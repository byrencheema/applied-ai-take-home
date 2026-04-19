import sqlite3

from langchain_core.tools import tool

from .db import conn, customer_suggestions, resolve_customer


_FTS_OPERATOR_TOKENS = {"OR", "AND", "NOT", "NEAR"}
_FTS_OPERATOR_CHARS = ('"', "(", ")", ":")


def _or_rewrite(query: str) -> str:
    """Turn `a b c` into `a OR b OR c` so any term matches (bm25 still ranks multi-term hits higher).
    Pass the query through unchanged if it already uses FTS operators (quotes, OR/AND/NOT, parens, column filters)."""
    if any(ch in query for ch in _FTS_OPERATOR_CHARS):
        return query
    tokens = query.split()
    if len(tokens) <= 1:
        return query
    if any(t.upper() in _FTS_OPERATOR_TOKENS for t in tokens):
        return query
    return " OR ".join(tokens)

ARTIFACT_TYPES = {
    "customer_call",
    "support_ticket",
    "internal_communication",
    "internal_document",
    "competitor_research",
}
REGIONS = {"Canada", "Nordics", "ANZ", "North America West"}
HEALTH_STATES = {"at risk", "watch list", "recovering", "healthy", "expanding"}
MAX_LIMIT = 10
PER_CUSTOMER_CAP = 2
OVERFETCH_ROWS = MAX_LIMIT * 10


@tool(parse_docstring=True)
def search_artifacts(
    query: str,
    artifact_type: str = "",
    customer_name: str = "",
    limit: int = 5,
) -> str:
    """Search Northstar artifacts (calls, tickets, internal docs/comms, competitor research) by content.

    Use this first to find specific facts, quotes, incidents, decisions, plans, or commitments.
    Results are ranked by relevance; follow up with get_artifact to read the full text.

    Args:
        query: Keyword(s). Space-separated words are ORed and ranked by bm25 (hits matching more terms rank higher). Use double quotes for exact phrases ("taxonomy rollout") — phrases are not split. Use * for prefix (taxono*). Explicit OR/AND/NOT and parentheses are passed through.
        artifact_type: Optional filter. One of customer_call, support_ticket, internal_communication, internal_document, competitor_research.
        customer_name: Optional filter. Fuzzy substring match on customer name (e.g. "BlueHarbor" matches "BlueHarbor Logistics").
        limit: Max hits to return, default 5, capped at 10.
    """
    limit = max(1, min(limit, MAX_LIMIT))

    if artifact_type and artifact_type not in ARTIFACT_TYPES:
        return f"Unknown artifact_type '{artifact_type}'. Valid: {', '.join(sorted(ARTIFACT_TYPES))}."

    customer_id = None
    canonical_customer = None
    if customer_name:
        resolved = resolve_customer(customer_name)
        if not resolved:
            hints = customer_suggestions(customer_name)
            hint_str = f" Closest: {', '.join(hints)}." if hints else ""
            return f"No customer matches '{customer_name}'.{hint_str} Call list_customers to enumerate."
        customer_id, canonical_customer = resolved

    sql = """
        SELECT a.artifact_id, a.title, a.artifact_type, a.created_at,
               a.customer_id,
               c.name AS customer_name,
               snippet(artifacts_fts, 3, '[', ']', '…', 18) AS snip
        FROM artifacts_fts f
        JOIN artifacts a ON a.artifact_id = f.artifact_id
        LEFT JOIN customers c ON c.customer_id = a.customer_id
        WHERE artifacts_fts MATCH ?
    """
    fts_query = _or_rewrite(query)
    params: list = [fts_query]
    if artifact_type:
        sql += " AND a.artifact_type = ?"
        params.append(artifact_type)
    if customer_id:
        sql += " AND a.customer_id = ?"
        params.append(customer_id)
    sql += " ORDER BY bm25(artifacts_fts) LIMIT ?"
    params.append(OVERFETCH_ROWS)  # over-fetch so per-customer capping has room to diversify

    try:
        raw = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        return f"FTS parse error: {e}. Quote phrases: \"your phrase\". Avoid bare operators."

    # Cap hits per customer so one heavily-documented customer can't dominate top-K.
    # Skip when a single customer is already filtered (nothing to diversify).
    cap = limit if customer_id else PER_CUSTOMER_CAP
    seen: dict[str, int] = {}
    rows = []
    for r in raw:
        key = r["customer_id"] or r["artifact_id"]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= cap:
            rows.append(r)
        if len(rows) >= limit:
            break

    if not rows:
        filters = []
        if artifact_type:
            filters.append(f"type={artifact_type}")
        if canonical_customer:
            filters.append(f"customer={canonical_customer}")
        filt = f" (filters: {', '.join(filters)})" if filters else ""
        return f"No artifacts match {query!r}{filt}. Try broader terms or remove filters."

    lines = [f"Found {len(rows)} hit(s) for {query!r}" + (
        f" · customer={canonical_customer}" if canonical_customer else ""
    ) + (f" · type={artifact_type}" if artifact_type else "") + ":"]
    for i, r in enumerate(rows, 1):
        meta = f"{r['artifact_type']} · {r['customer_name'] or 'no customer'} · {r['created_at'][:10]}"
        lines.append(f"\n[{i}] {r['artifact_id']} ({meta})")
        lines.append(f"    {r['title']}")
        lines.append(f"    {r['snip']}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def get_artifact(artifact_id: str) -> str:
    """Fetch the full content of one artifact by id. Follow up to a search_artifacts result.

    Args:
        artifact_id: An id like "art_bd3560dfe194", taken from a search_artifacts hit.
    """
    row = conn.execute(
        """
        SELECT a.artifact_id, a.title, a.artifact_type, a.created_at, a.summary, a.content_text,
               c.name AS customer_name
        FROM artifacts a
        LEFT JOIN customers c ON c.customer_id = a.customer_id
        WHERE a.artifact_id = ?
        """,
        (artifact_id,),
    ).fetchone()
    if not row:
        return f"No artifact with id {artifact_id!r}. Verify the id from a search_artifacts hit."

    meta = f"{row['artifact_type']} · {row['customer_name'] or 'no customer'} · {row['created_at'][:10]}"
    return (
        f"{row['artifact_id']} ({meta})\n"
        f"Title: {row['title']}\n\n"
        f"Summary: {row['summary']}\n\n"
        f"{row['content_text']}"
    )


@tool(parse_docstring=True)
def list_customers(
    region: str = "",
    industry: str = "",
    account_health: str = "",
    product_name: str = "",
) -> str:
    """Enumerate Northstar customers by structured filters (region, industry, account health, deployed product).

    Use this when the question asks "which customers ..." or needs a cohort, before searching their artifacts.
    All filters ANDed; empty means no filter.

    Args:
        region: One of Canada, Nordics, ANZ, North America West.
        industry: Case-insensitive substring match on industry (e.g. "Logistics").
        account_health: One of at risk, watch list, recovering, healthy, expanding.
        product_name: Exact product name the customer has deployed (e.g. "Event Nexus"). Matched via implementations.
    """
    if region and region not in REGIONS:
        return f"Unknown region '{region}'. Valid: {', '.join(sorted(REGIONS))}."
    if account_health and account_health not in HEALTH_STATES:
        return f"Unknown account_health '{account_health}'. Valid: {', '.join(sorted(HEALTH_STATES))}."

    sql = """
        SELECT DISTINCT c.customer_id, c.name, c.industry, c.region, c.account_health, c.crm_stage
        FROM customers c
        LEFT JOIN implementations i ON i.customer_id = c.customer_id
        LEFT JOIN products p ON p.product_id = i.product_id
        WHERE 1=1
    """
    params: list = []
    if region:
        sql += " AND c.region = ?"
        params.append(region)
    if industry:
        sql += " AND lower(c.industry) LIKE lower(?)"
        params.append(f"%{industry}%")
    if account_health:
        sql += " AND c.account_health = ?"
        params.append(account_health)
    if product_name:
        sql += " AND lower(p.name) = lower(?)"
        params.append(product_name)
    sql += " ORDER BY c.name"

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return "No customers match those filters."

    filters = []
    if region:
        filters.append(f"region={region}")
    if industry:
        filters.append(f"industry~{industry}")
    if account_health:
        filters.append(f"health={account_health}")
    if product_name:
        filters.append(f"product={product_name}")
    header = f"{len(rows)} customer(s)" + (f" ({', '.join(filters)})" if filters else "") + ":"
    body = "\n".join(
        f"- {r['name']} · {r['industry']} · {r['region']} · {r['account_health']} · {r['crm_stage']}"
        for r in rows
    )
    return f"{header}\n{body}"


@tool(parse_docstring=True)
def get_customer(name_or_id: str) -> str:
    """Fetch full profile for one customer: account, deployed product, implementation status, scope, risks.

    Use before diving into a customer's artifacts when you need context (size, product, health, kickoff dates).

    Args:
        name_or_id: Customer name (fuzzy match, e.g. "BlueHarbor") or customer_id (e.g. "cust_...").
    """
    resolved = resolve_customer(name_or_id)
    if not resolved:
        hints = customer_suggestions(name_or_id)
        hint_str = f" Closest: {', '.join(hints)}." if hints else ""
        return f"No customer matches {name_or_id!r}.{hint_str} Call list_customers to enumerate."
    customer_id, _ = resolved

    row = conn.execute(
        """
        SELECT c.name, c.industry, c.subindustry, c.region, c.country, c.size_band,
               c.employee_count, c.annual_revenue_band, c.crm_stage, c.account_health,
               c.primary_contact_name, c.primary_contact_email, c.tech_stack_summary, c.notes,
               s.scenario_summary, s.trigger_event, s.pain_point,
               i.status AS impl_status, i.deployment_model, i.kickoff_date, i.go_live_date,
               i.contract_value, i.scope_summary, i.success_metrics_json, i.risks_json,
               p.name AS product_name
        FROM customers c
        LEFT JOIN scenarios s ON s.scenario_id = c.scenario_id
        LEFT JOIN implementations i ON i.customer_id = c.customer_id
        LEFT JOIN products p ON p.product_id = i.product_id
        WHERE c.customer_id = ?
        """,
        (customer_id,),
    ).fetchone()

    return (
        f"{row['name']} · {row['industry']} / {row['subindustry']} · {row['region']} ({row['country']})\n"
        f"Size: {row['size_band']} · {row['employee_count']} employees · revenue band {row['annual_revenue_band']}\n"
        f"CRM stage: {row['crm_stage']} · Account health: {row['account_health']}\n"
        f"Primary contact: {row['primary_contact_name']} <{row['primary_contact_email']}>\n"
        f"Tech stack: {row['tech_stack_summary']}\n\n"
        f"Scenario: {row['scenario_summary']}\n"
        f"Trigger: {row['trigger_event']}\n"
        f"Pain point: {row['pain_point']}\n\n"
        f"Deployed product: {row['product_name']} ({row['deployment_model']}) · status={row['impl_status']}\n"
        f"Kickoff {row['kickoff_date']} · Go-live {row['go_live_date'] or 'TBD'} · Contract ${row['contract_value']:,}\n"
        f"Scope: {row['scope_summary']}\n"
        f"Success metrics: {row['success_metrics_json']}\n"
        f"Risks: {row['risks_json']}\n\n"
        f"Notes: {row['notes']}"
    )


TOOLS = [search_artifacts, get_artifact, list_customers, get_customer]
