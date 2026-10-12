"""Test suite for Step 3: Check Generator and Statistical Power Gate (The 339 Rules Engine)."""
import sys
import json
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESPONSES_N45_PATH, RESPONSES_N150_PATH
from src.check_generator import generate_checks_from_schema, get_blocked_categories
from src.power_gate import apply_power_gate, evaluate_check_power, format_power_gate_statement
from src.check_runner import run_checks


def test_check_generator_exact_339():
    """Verify that exactly 339 QC checks are automatically derived from the Confirmit schema across all 11 categories."""
    checks = generate_checks_from_schema()
    assert len(checks) == 339, f"Expected exactly 339 checks, got {len(checks)}"

    by_type = {}
    for c in checks:
        by_type[c.check_type] = by_type.get(c.check_type, 0) + 1

    expected_counts = {
        "item_non_response": 195,
        "degenerate_distribution": 83,
        "range_violation": 23,
        "straightlining": 22,
        "allocation_sum": 8,
        "duplicate_source": 2,
        "device_anomaly": 2,
        "speeding": 1,
        "disposition_mix": 1,
        "script_errors": 1,
        "dropout_point": 1
    }

    for ct, expected_count in expected_counts.items():
        actual_count = by_type.get(ct, 0)
        assert actual_count == expected_count, (
            f"Category count mismatch for {ct}: expected {expected_count}, got {actual_count}"
        )

    # Verify blocked categories declaration
    blocked = get_blocked_categories()
    assert "routing_conformance" in blocked, "routing_conformance must be declared blocked"
    assert "quota_conformance" in blocked, "quota_conformance must be declared blocked"
    assert blocked["routing_conformance"]["status"] == "BLOCKED"
    assert blocked["quota_conformance"]["status"] == "BLOCKED"


def test_power_gate_suppression_logic():
    """Verify that statistical power gate suppresses underpowered checks at n=45 vs n=150 with stated reasons."""
    checks = generate_checks_from_schema()

    # Evaluation at n=45
    exec_45, supp_45 = apply_power_gate(checks, 45)
    # At n=45, 83 degenerate_distribution checks (n=200) + 1 disposition_mix check (n=50) are suppressed = 84
    assert len(supp_45) == 84, f"Expected 84 suppressed checks at n=45, got {len(supp_45)}"
    assert len(exec_45) == 255, f"Expected 255 executable checks at n=45, got {len(exec_45)}"

    # Evaluation at n=150
    exec_150, supp_150 = apply_power_gate(checks, 150)
    # At n=150, disposition_mix runs; 83 degenerate_distribution checks remain suppressed = 83
    assert len(supp_150) == 83, f"Expected 83 suppressed checks at n=150, got {len(supp_150)}"
    assert len(exec_150) == 256, f"Expected 256 executable checks at n=150, got {len(exec_150)}"

    # Verify suppression statement formatting
    stmt = format_power_gate_statement(
        passed_count=250,
        flagged_count=5,
        suppressed_count=84,
        current_n=45,
        next_n=100
    )
    assert stmt == "250 checks passed, 5 flagged, 84 not yet valid at n=45 — re-run at n=100."


def test_check_runner_execution_and_anomalies():
    """Verify that check_runner produces standardized anomaly records matching client specification."""
    with open(RESPONSES_N150_PATH) as f:
        records_150 = json.load(f)["items"]

    res150 = run_checks(records_150)
    anomalies = res150["anomalies"]
    flagged_ids = [a["check_id"] for a in anomalies]

    # Verify standard anomaly record shape
    for a in anomalies:
        assert "check_id" in a
        assert "check_type" in a
        assert "variable" in a
        assert "statistic" in a
        assert "expected" in a
        assert "n" in a
        assert "affected_respondents" in a
        assert a["valid_at_n"] is True
        assert len(a["affected_respondents"]) > 0

    # Verify key planted faults are flagged at n=150
    assert "item_non_response.Q7Sk" in flagged_ids, "F01 Q7Sk non-response missing from flagged"
    assert "straightlining.T14" in flagged_ids, "F02 T14 straightlining missing from flagged"
    assert "speeding.hidLOI" in flagged_ids, "F03 speeding missing from flagged"
    assert "range_violation.S5" in flagged_ids, "F04 S5 range violation missing from flagged"
    assert "allocation_sum.Q7Mx" in flagged_ids, "F05 Q7Mx allocation sum missing from flagged"
    assert "disposition_mix.status" in flagged_ids, "F08 disposition mix missing from flagged"

    # Evaluation at n=45: Confirm disposition_mix is suppressed by power gate
    with open(RESPONSES_N45_PATH) as f:
        records_45 = json.load(f)["items"]

    res45 = run_checks(records_45)
    flagged_ids_45 = [a["check_id"] for a in res45["anomalies"]]
    assert "disposition_mix.status" not in flagged_ids_45, (
        "disposition_mix should be suppressed at n=45 by power gate"
    )
    assert "item_non_response.Q7Sk" in flagged_ids_45


if __name__ == "__main__":
    test_check_generator_exact_339()
    test_power_gate_suppression_logic()
    test_check_runner_execution_and_anomalies()
    print("All Step 3 check generator and power gate tests passed successfully!")

