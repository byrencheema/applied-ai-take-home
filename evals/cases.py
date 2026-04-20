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
    {
        "id": "H4_nordfryst_reduction",
        "question": (
            "for NordFryst, by how much did alert volume jump since Oct 2025 and what "
            "reduction target did we commit to at renewal, plus the MTTA goal?"
        ),
        "must_include": [
            "NordFryst",
            ("45%", "45 percent"),
            ("60%", "60 percent"),
            "90 days",
            ("24m", "24 minutes"),
            ("<10m", "10 minutes", "10m"),
        ],
    },
    {
        "id": "H5_drs_hysteresis",
        "question": (
            "what's the exact hysteresis rule proposed for DRS auto-routing, "
            "the safe-mode trigger, and the renewal credit amount?"
        ),
        "must_include": [
            "0.62",
            "3 consecutive",
            "10-minute",
            ("40%", "40 percent"),
            "72",
            ("AUD 9,500", "9,500"),
        ],
    },
    {
        "id": "H6_laurentia_reject",
        "question": (
            "at Province of Laurentia, what percentage of events was SI-ETL-FILTER "
            "rejecting after the regional launch, what quarantined topic are we "
            "routing unknown fields to as the permissive fix, and what commercial "
            "concession did we attach to a 12-month renewal?"
        ),
        "must_include": [
            "Laurentia",
            ("28%", "28 percent"),
            "160",
            "2026-04-15",
            ("laurentia.unknown", "quarantin"),
        ],
    },
    {
        "id": "H7_helix_canonical",
        "question": (
            "how exactly is Helix's canonical_id computed on the Plant D collectors, "
            "and what duplicate-reduction target did we commit to within 7 days?"
        ),
        "must_include": [
            "canonical_id",
            "sha256",
            "orderId",
            "lineId",
            "deviceId",
            ("75%", "75 percent"),
            "7 days",
        ],
    },
    {
        "id": "H8_harbourline_scim_sla",
        "question": (
            "for Harbourline Regional Transit Authority, what's the OR-ROLE-BASED "
            "approval timer change, and what median/p95 provisioning latency SLA did we propose?"
        ),
        "must_include": [
            ("15m", "15 minutes", "15-minute"),
            ("45m", "45 minutes", "45-minute"),
            ("+30m", "30 minutes", "30-minute"),
            "2 hours", "8 hours",
            "60 days", "90 days",
        ],
    },
    {
        "id": "H9_peregrine_coalesce",
        "question": (
            "what SI-ETL-FILTER coalesce rule does Peregrine's remediation use for the "
            "BU C shipment id mismatch, and what's the duplicates acceptance criteria?"
        ),
        "must_include": [
            "coalesce",
            "shipment_id",
            "shipment_ref",
            "BUC-",
            ("5%", "5 percent"),
            "10 consecutive",
            ("60-day", "60 day"),
        ],
    },
    {
        "id": "H10_fyrkrona_thresholds",
        "question": (
            "for Fyrkrona Renewables, what's the EN-DEDUPE window change, the "
            "dynamic-threshold formula, and the renewal-review alert-reduction and "
            "MTTA targets by 2026-04-30?"
        ),
        "must_include": [
            ("5-minute", "5 minute"),
            ("low_priority", "low-priority", "low priority"),
            "median",
            "IQR",
            ("14d", "14-day", "14 day"),
            ("60%", "60 percent"),
            ("<10m", "10m", "10 minutes"),
        ],
    },
    {
        "id": "H11_arcadia_tf_weight",
        "question": (
            "what TF-weight change did we make for Arcadia's tenant_tag, and what are "
            "the post-swap search relevance and playbook trigger miss rate acceptance metrics?"
        ),
        "must_include": [
            "tenant_tag",
            "1.0",
            "0.2",
            "0.85",
            "500",
            ("28%", "28 percent"),
            ("5%", "5 percent"),
            "7 days",
        ],
    },
    {
        "id": "H12_noiseguard_cohort",
        "question": (
            "which customers have NoiseGuard named as their primary competitor, and "
            "what's the shared failure pattern across that cohort?"
        ),
        "must_include": [
            "Arcadia Cloudworks",
            "BlueHarbor Logistics",
            "CedarWind Renewables",
            "HelioFab Systems",
            "Pacific Health Network",
            "Pioneer Freight Solutions",
            ("taxonomy", "search relevance"),
        ],
    },
    {
        "id": "H13_harborhome_overrides",
        "question": (
            "at HarborHome, what confidence-score band drives the audit-escape overrides, "
            "and what adoption, MTTRoute, and override-log retention targets does the pilot plan commit to?"
        ),
        "must_include": [
            "0.55",
            "0.75",
            ("50%", "50 percent"),
            "6 weeks",
            ("<8", "8 minutes", "8m"),
            ("18 months", "18-month", "18mo"),
        ],
    },
]
