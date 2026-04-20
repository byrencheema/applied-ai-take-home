"""Eval cases for the Slack Q&A agent.

Each case has:
- id: short label
- question: the user's question (stripped of @bot)
- must_include: substrings (case-insensitive) that must appear in the answer
- must_not_include: optional — substrings that must NOT appear

Keyword lists come straight from the take-home's expected answers."""

CASES = [
    {
        "id": "Q1_blueharbor_proof_plan",
        "question": (
            "which customer's issue started after the 2026-02-20 taxonomy rollout, "
            "and what proof plan did we propose to get them comfortable with renewal?"
        ),
        "must_include": [
            "BlueHarbor",
            "7-10",
            "index weighting",
            ("mapping layer", "taxonomy mapping", "mapping rules"),
            "top 20",
            ("80%", "80 percent"),
        ],
    },
    {
        "id": "Q2_verdant_bay_rollback",
        "question": (
            "for Verdant Bay, what's the approved live patch window, "
            "and exactly how do we roll back if the validation checks fail?"
        ),
        "must_include": [
            ("2026-03-24", "March 24, 2026", "March 24 2026"),
            "02:00", "04:00",
            "orchestrator rollback --target ruleset", "invalidation",
        ],
    },
    {
        "id": "Q3_maplehavest_quebec",
        "question": (
            "in the MapleHarvest Quebec pilot, what temporary field mappings are we planning "
            "in the router transform, and what is the March 23 workshop supposed to produce?"
        ),
        "must_include": [
            "txn_id", "transaction_id", "total_amount", "amount_cents",
            "store_id", "register_id", "SI-SCHEMA-REG",
        ],
    },
    {
        "id": "Q4_aureum_scim",
        "question": (
            "what SCIM fields were conflicting at Aureum, "
            "and what fast fix did Jin propose so we don't have to wait on Okta change control?"
        ),
        "must_include": ["department", "businessUnit", "Signal Ingest", "Jin", "hot"],
    },
    {
        "id": "H1_defection_risk",
        "question": (
            "which customer looks most likely to defect to a cheaper tactical competitor "
            "if we miss the next promised milestone, and what exactly is that milestone?"
        ),
        "must_include": [
            "BlueHarbor",
            "NoiseGuard",
            "7-10",
            ("80%", "80 percent"),
            "top 20",
        ],
    },
    {
        "id": "H2_na_west_cohort",
        "question": (
            "among the North America West Event Nexus accounts, which ones are really dealing "
            "with taxonomy/search semantics problems versus duplicate-action problems?"
        ),
        "must_include": [
            # taxonomy cohort
            "Arcadia Cloudworks", "BlueHarbor Logistics", "CedarWind Renewables",
            "HelioFab Systems", "Pacific Health Network", "Pioneer Freight Solutions",
            # duplicate-action cohort
            "Helix Assemblies", "LedgerBright", "LedgerPeak",
            "MedLogix", "Peregrine", "Pioneer Grid Retail",
        ],
    },
    {
        "id": "Z1_out_of_scope",
        "question": "what's the weather like in Tokyo today?",
        "must_include": [
            ("outside", "don't cover", "can't help", "only help", "not something"),
            ("Northstar", "customer accounts"),
        ],
        "must_not_include": ["Tokyo", "art_", "cus_"],
    },
    {
        "id": "H3_canada_bypass_pattern",
        "question": (
            "do we have a recurring Canada approval-bypass pattern across accounts, "
            "or is MapleBridge basically a one-off? Give me the customer names and the shared "
            "failure pattern in plain English."
        ),
        "must_include": [
            "MapleBridge", "Verdant Bay", "Maple Regional Transit",
            "MapleBay", "MapleFork", "MaplePath", "MapleWest",
        ],
    },
]
