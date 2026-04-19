from src.tools import get_artifact, get_customer, list_customers, search_artifacts


def invoke(tool, **kwargs):
    return tool.invoke(kwargs)


def test_search_finds_blueharbor_retro():
    out = invoke(search_artifacts, query="taxonomy rollout", customer_name="BlueHarbor")
    assert "art_bd3560dfe194" in out
    assert "Retro" in out


def test_search_respects_artifact_type_filter():
    out = invoke(
        search_artifacts,
        query="taxonomy rollout",
        artifact_type="support_ticket",
    )
    assert "support_ticket" in out
    assert "internal_document" not in out


def test_search_unknown_artifact_type_errors():
    out = invoke(search_artifacts, query="anything", artifact_type="bogus")
    assert "Unknown artifact_type" in out


def test_search_unknown_customer_errors():
    out = invoke(search_artifacts, query="x", customer_name="NotARealCustomerNameHere")
    assert "No customer matches" in out
    assert "list_customers" in out


def test_search_empty_result_is_actionable():
    out = invoke(search_artifacts, query="zzzzz_nomatchhere_qqqq")
    assert "No artifacts match" in out


def test_get_artifact_blueharbor_retro_contains_proof_plan():
    out = invoke(get_artifact, artifact_id="art_bd3560dfe194")
    assert "A/B test" in out
    assert "80%" in out
    assert "7-10" in out


def test_get_artifact_verdant_bay_rollback():
    out = invoke(get_artifact, artifact_id="art_fff67d92fe41")
    assert "orchestrator rollback --target ruleset" in out
    assert "2026-03-24" in out


def test_get_artifact_missing_is_actionable():
    out = invoke(get_artifact, artifact_id="art_does_not_exist")
    assert "No artifact" in out


def test_list_customers_regions():
    out = invoke(list_customers, region="North America West")
    assert "12 customer(s)" in out
    assert "BlueHarbor Logistics" in out

    out = invoke(list_customers, region="Canada")
    assert "13 customer(s)" in out


def test_list_customers_unknown_region():
    out = invoke(list_customers, region="Mars")
    assert "Unknown region" in out


def test_list_customers_by_product():
    out = invoke(list_customers, region="North America West", product_name="Event Nexus")
    assert "BlueHarbor Logistics" in out


def test_get_customer_blueharbor():
    out = invoke(get_customer, name_or_id="BlueHarbor")
    assert "BlueHarbor Logistics" in out
    assert "Logistics" in out
    assert "North America West" in out
    assert "Deployed product" in out


def test_get_customer_unknown_suggests():
    out = invoke(get_customer, name_or_id="NotARealCustomer")
    assert "No customer matches" in out
