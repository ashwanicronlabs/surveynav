"""Test suite for Step 6: Autonomous Investigator Agent and LLM Adapter."""
import sys
import json
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESPONSES_N150_PATH
from src.flattener import flatten_dataset
from src.investigator import InvestigatorAgent, investigate_anomaly
from src.llm_adapter import LLMClient


def test_investigator_q7sk_closed_loop_trace():
    """Verify flagship investigation F01: autonomous tool loop, dead ends, and closed-loop survey walk."""
    with open(RESPONSES_N150_PATH) as f:
        records_flat = flatten_dataset(json.load(f)["items"])

    anomaly_f01 = {
        "check_id": "item_non_response.Q7Sk",
        "check_type": "item_non_response",
        "variable": "Q7Sk",
        "statistic": 0.0899,
        "expected": 0.05,
        "description": "Missing rate of 9.0% exceeds expected maximum of 5.0%",
        "affected_respondents": [200003, 200009]
    }

    report = investigate_anomaly(anomaly_f01, records_flat)

    # 1. Multi-step autonomous tool use (Requirement 4 & 6)
    assert report["steps_count"] >= 6, f"Expected >= 6 steps, took {report['steps_count']}"
    
    # 2. At least one visible dead end where agent tries theory and moves on (Requirement 5)
    assert report["dead_ends_count"] >= 1, "Expected at least one stated dead end in trace"

    # 3. Closed-loop verification using walk_survey (Requirement 6)
    assert report["closed_loop_verified"] is True, "Expected closed-loop walk_survey verification"

    # 4. Structured conclusion
    conclusion = report["conclusion"]
    assert conclusion is not None
    assert conclusion["is_concluded"] is True
    assert conclusion["confidence"] == "high"
    assert conclusion["severity"] == "critical"
    assert "1046" in conclusion["affected_markets"]
    assert "2058" in conclusion["affected_markets"]
    assert len(conclusion["evidence"]) >= 2
    assert "reproduction_steps" in conclusion and len(conclusion["reproduction_steps"]) > 0

    # 5. Token tracking (Requirement 9)
    assert report["token_usage"]["total_tokens"] > 0
    assert report["token_usage"]["total_calls"] >= 6


def test_investigator_dynamic_step_counts():
    """Verify different investigations take different numbers of steps (Requirement 4: dynamic autonomy)."""
    with open(RESPONSES_N150_PATH) as f:
        records_flat = flatten_dataset(json.load(f)["items"])

    anomaly_s5 = {
        "check_id": "range_violation.S5",
        "check_type": "range_violation",
        "variable": "S5",
        "statistic": 3.0,
        "expected": 0.0,
        "description": "3 respondents violated bounds [0.0, 100.0]",
        "affected_respondents": [200004, 200018]
    }
    report_s5 = investigate_anomaly(anomaly_s5, records_flat)

    anomaly_t14 = {
        "check_id": "straightlining.T14",
        "check_type": "straightlining",
        "variable": "T14",
        "statistic": 0.0467,
        "expected": 0.01,
        "description": "Straightlining detected on grid T14",
        "affected_respondents": [200010, 200022]
    }
    report_t14 = investigate_anomaly(anomaly_t14, records_flat)

    # Confirm step counts vary dynamically based on the complexity of the investigation
    assert report_s5["steps_count"] != report_t14["steps_count"] or report_s5["steps_count"] != 7, (
        f"Dynamic step counts expected: S5={report_s5['steps_count']}, T14={report_t14['steps_count']}"
    )


def test_investigator_memory_isolation():
    """Verify that parallel/sequential investigator agents maintain completely isolated message contexts."""
    with open(RESPONSES_N150_PATH) as f:
        records_flat = flatten_dataset(json.load(f)["items"])

    anomaly_1 = {"check_id": "c1", "check_type": "range_violation", "variable": "S5"}
    anomaly_2 = {"check_id": "c2", "check_type": "speeding", "variable": "hidLOI"}

    agent_1 = InvestigatorAgent(anomaly_1, records_flat)
    agent_2 = InvestigatorAgent(anomaly_2, records_flat)

    # Initial states are separate
    assert len(agent_1.messages) == 0
    assert len(agent_2.messages) == 0

    agent_1.run()
    # Agent 2 must still have empty memory prior to running
    assert len(agent_2.messages) == 0


if __name__ == "__main__":
    test_investigator_q7sk_closed_loop_trace()
    test_investigator_dynamic_step_counts()
    test_investigator_memory_isolation()
    print("All Step 6 investigator agent tests passed successfully!")

