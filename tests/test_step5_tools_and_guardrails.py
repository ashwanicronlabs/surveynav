"""Test suite for Step 5: Investigator Tool Layer and Guardrails."""
import sys
import json
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESPONSES_N150_PATH
from src.flattener import flatten_dataset
from src.tools import (
    get_schema,
    query_data,
    sample_respondents,
    walk_survey,
    conclude,
    InvestigationGuardrails,
    INVESTIGATOR_TOOL_DEFINITIONS
)


def test_get_schema_lookup():
    """Verify get_schema tool correctly inspects Confirmit variable types and limits."""
    # Test allocation slider Q7Sk
    s_q7sk = get_schema("Q7Sk")
    assert s_q7sk["found"] is True
    assert s_q7sk["variableType"] == "numeric"
    assert s_q7sk["upperNumericLimit"] == 100.0
    assert len(s_q7sk["fields"]) == 4

    # Test numeric rating S5
    s_s5 = get_schema("S5")
    assert s_s5["found"] is True
    assert s_s5["variableType"] == "numeric"
    assert s_s5["upperNumericLimit"] == 100.0

    # Test grid T14
    s_t14 = get_schema("T14")
    assert s_t14["found"] is True
    assert len(s_t14["fields"]) == 7

    # Test unknown question
    s_unknown = get_schema("NON_EXISTENT_VAR")
    assert s_unknown["found"] is False


def test_query_data_aggregation_and_truncation():
    """Verify query_data aggregates metrics and enforces 20-row context truncation ceiling."""
    with open(RESPONSES_N150_PATH) as f:
        records = flatten_dataset(json.load(f)["items"])

    # Test grouping by language for Q7Sk missing rate
    res_lang = query_data(records, target_variable="Q7Sk", group_by="l", metric="missing_rate")
    assert res_lang["total_records"] == 150
    assert len(res_lang["rows"]) == 5  # 5 languages
    assert any(row["group"] in ("1046", "2058") for row in res_lang["rows"])

    # Test grouping by device
    res_dev = query_data(records, target_variable="Q7Sk", group_by="lastdevicetype", metric="missing_rate")
    assert res_dev["total_records"] == 150

    # Test Context Truncation Guardrail: Create 35 unique synthetic groups
    mock_large_dataset = [{"group_id": f"group_{i}", "val": i} for i in range(35)]
    res_large = query_data(mock_large_dataset, group_by="group_id", metric="count")
    assert res_large["total_groups"] == 35
    assert len(res_large["rows"]) == 20, "Result rows must be truncated to exactly 20 rows"
    assert "truncation_notice" in res_large
    assert "15 additional rows omitted" in res_large["truncation_notice"]


def test_sample_respondents_clamping():
    """Verify sample_respondents limits record count to 5 to protect context window."""
    with open(RESPONSES_N150_PATH) as f:
        records = flatten_dataset(json.load(f)["items"])

    # Request 10 records -> must clamp to max 5
    sampled = sample_respondents(records, n=10)
    assert sampled["sampled_count"] <= 5
    assert len(sampled["records"]) <= 5
    assert "respid" in sampled["records"][0]


def test_walk_survey_integration():
    """Verify walk_survey connects directly to the survey state machine and detects F01."""
    res_walk = walk_survey(language="1046", device="mobile", target_question="Q7Sk")
    assert "NONE" in res_walk["target_state"]["rendered_controls"]
    assert any("F01" in b for b in res_walk["bugs_observed"])


def test_conclude_schema_enforcement():
    """Verify conclude tool enforces structured conclusion dictionary."""
    finding = conclude(
        cause="Q7Sk touch slider widget fails to load on mobile devices in Portuguese and Spanish markets",
        confidence="high",
        evidence=["query_data confirms missing rates concentrated on mobile for languages 1046/2058", "walk_survey reproduces missing slider control"],
        severity="high",
        affected_markets=["1046", "2058"],
        reproduction_steps="Open survey on mobile with l=1046; navigate to Q7Sk; observe missing slider control."
    )
    assert finding["is_concluded"] is True
    assert finding["confidence"] == "high"
    assert finding["severity"] == "high"
    assert len(finding["affected_markets"]) == 2
    assert len(finding["evidence"]) == 2


def test_guardrails_loop_detection_and_call_ceiling():
    """Verify guardrails halt runaway execution on loops or exceeding 25 calls."""
    guard = InvestigationGuardrails(max_calls=25, loop_limit=3)

    # 1. Test Loop Detection: 3 identical calls trigger abort
    tool_args = {"target_variable": "Q7Sk", "group_by": "l"}
    
    # Call 1: Allowed
    ok1, _ = guard.record_and_validate("query_data", tool_args)
    assert ok1 is True
    
    # Call 2: Allowed
    ok2, _ = guard.record_and_validate("query_data", tool_args)
    assert ok2 is True

    # Call 3: Loop detected -> Blocked
    ok3, reason3 = guard.record_and_validate("query_data", tool_args)
    assert ok3 is False
    assert guard.is_aborted is True
    assert "Loop detected" in reason3

    # 2. Test Call Ceiling: Exceeding 25 calls
    ceiling_guard = InvestigationGuardrails(max_calls=25, loop_limit=10)
    for i in range(25):
        ok, _ = ceiling_guard.record_and_validate("query_data", {"step": i})
        assert ok is True

    # 26th call: Blocked by ceiling
    ok_blocked, reason_blocked = ceiling_guard.record_and_validate("query_data", {"step": 26})
    assert ok_blocked is False
    assert ceiling_guard.is_aborted is True
    assert "Hard cap of 25 tool calls reached" in reason_blocked


def test_tool_definitions_schema_valid():
    """Verify investigator tool definitions comply with standard function calling schema."""
    assert len(INVESTIGATOR_TOOL_DEFINITIONS) == 5
    tool_names = [t["name"] for t in INVESTIGATOR_TOOL_DEFINITIONS]
    assert tool_names == ["query_data", "get_schema", "sample_respondents", "walk_survey", "conclude"]


if __name__ == "__main__":
    test_get_schema_lookup()
    test_query_data_aggregation_and_truncation()
    test_sample_respondents_clamping()
    test_walk_survey_integration()
    test_conclude_schema_enforcement()
    test_guardrails_loop_detection_and_call_ceiling()
    test_tool_definitions_schema_valid()
    print("All Step 5 tool layer and guardrails tests passed successfully!")

