"""Test suite for Step 7: Triage, Adjudicator, and Go/No-Go Decision Pipeline."""
import sys
import json
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESPONSES_N45_PATH, RESPONSES_N150_PATH
from src.llm_adapter import LLMClient
from src.triage import triage_anomalies, TriageReport
from src.adjudicator import adjudicate_findings, run_soft_launch_qc_pipeline, AdjudicationReport


def test_triage_single_call_and_prioritization():
    """Verify Triage makes strictly 1 model call and prioritizes into P1, P2, P3 bands."""
    sample_anomalies = [
        {"check_id": "item_non_response.Q7Sk", "check_type": "item_non_response", "variable": "Q7Sk", "statistic": 0.09, "expected": 0.05, "description": "High missing rate"},
        {"check_id": "range_violation.S5", "check_type": "range_violation", "variable": "S5", "statistic": 3.0, "expected": 0.0, "description": "Bounds violated"},
        {"check_id": "allocation_sum.Q7Mx", "check_type": "allocation_sum", "variable": "Q7Mx", "statistic": 3.0, "expected": 0.0, "description": "Sum != 100"},
        {"check_id": "speeding.hidLOI", "check_type": "speeding", "variable": "hidLOI", "statistic": 0.25, "expected": 0.02, "description": "Speeding cluster"},
        {"check_id": "disposition_mix.status", "check_type": "disposition_mix", "variable": "status", "statistic": 0.85, "expected": 0.25, "description": "High screenout rate"},
        {"check_id": "straightlining.T14", "check_type": "straightlining", "variable": "T14", "statistic": 0.05, "expected": 0.01, "description": "Straightliner cluster"}
    ]

    client = LLMClient()
    report: TriageReport = triage_anomalies(sample_anomalies, sample_size_n=150, llm_client=client)

    # 1. Strictly 1 model call (Requirement 3: Check runner 0, Triage 1, Adjudicator 1)
    assert report.model_calls == 1, f"Expected strictly 1 model call in Triage, got {report.model_calls}"
    assert len(report.tasks) == len(sample_anomalies)

    # 2. Priority bands P1, P2, P3
    assert report.priority_counts["P1"] >= 1, "Expected at least 1 P1 task"
    assert report.priority_counts["P2"] >= 1, "Expected at least 1 P2 task"
    assert report.priority_counts["P3"] >= 1, "Expected at least 1 P3 task"

    # Verify critical integrity checks are classified as P1
    p1_variables = [t.target_variable for t in report.tasks if t.priority == "P1"]
    assert "S5" in p1_variables or "Q7Mx" in p1_variables or "Q7Sk" in p1_variables

    # Verify each task has an initial hypothesis formulated
    for task in report.tasks:
        assert task.initial_hypothesis != ""
        assert task.dispatch_id.startswith("TASK-")


def test_triage_empty_anomalies():
    """Verify Triage gracefully handles 0 anomalies without unnecessary model calls."""
    client = LLMClient()
    report = triage_anomalies([], sample_size_n=150, llm_client=client)
    assert report.total_anomalies == 0
    assert report.model_calls == 0
    assert len(report.tasks) == 0


def test_adjudicator_deduplication_and_severity():
    """Verify Adjudicator deduplicates multiple symptoms (F01) and assigns valid severities."""
    client = LLMClient()

    # Synthetic investigator reports where F01 produced two symptoms: Q7Sk missing and device anomaly
    mock_reports = [
        {
            "check_id": "item_non_response.Q7Sk",
            "check_type": "item_non_response",
            "variable": "Q7Sk",
            "steps_count": 7,
            "closed_loop_verified": True,
            "conclusion": {
                "cause": "Allocation slider widget fails to render on mobile devices in languages 1046 and 2058",
                "severity": "critical",
                "affected_markets": ["1046", "2058"]
            }
        },
        {
            "check_id": "device_anomaly.lastdevicetype",
            "check_type": "device_anomaly",
            "variable": "lastdevicetype",
            "steps_count": 4,
            "closed_loop_verified": True,
            "conclusion": {
                "cause": "Mobile device completion anomaly directly tied to Q7Sk slider rendering failure",
                "severity": "critical",
                "affected_markets": ["1046", "2058"]
            }
        },
        {
            "check_id": "range_violation.S5",
            "check_type": "range_violation",
            "variable": "S5",
            "steps_count": 4,
            "closed_loop_verified": True,
            "conclusion": {
                "cause": "Numeric range boundary check missing on S5, accepting > 100",
                "severity": "critical",
                "affected_markets": ["all"]
            }
        }
    ]

    adj_report: AdjudicationReport = adjudicate_findings(
        investigation_reports=mock_reports,
        power_gate_summary={"statement": "250 passed, 3 flagged, 83 suppressed", "suppressed_checks": 83},
        sample_size_n=150,
        llm_client=client
    )

    # 1. Strictly 1 model call in Adjudicator
    assert adj_report.model_calls == 1, f"Expected strictly 1 model call in Adjudicator, got {adj_report.model_calls}"

    # 2. Deduplication check: 3 input symptoms should be merged into 2 deduplicated findings
    # (Q7Sk + device_anomaly merged into single F01 finding)
    assert len(adj_report.findings) == 2, (
        f"Expected 2 deduplicated findings after merging F01 symptoms, got {len(adj_report.findings)}"
    )

    f01_finding = next((f for f in adj_report.findings if "Q7Sk" in f.symptom_checks or "FINDING-F01" in f.finding_id), None)
    assert f01_finding is not None
    assert "device_anomaly.lastdevicetype" in f01_finding.symptom_checks
    assert f01_finding.severity == "critical"
    assert f01_finding.closed_loop_verified is True


def test_adjudicator_per_market_go_no_go():
    """Verify Adjudicator produces granular per-market decisions (Go for 3, Hold 2)."""
    with open(RESPONSES_N150_PATH) as f:
        data150 = json.load(f)["items"]

    client = LLMClient()
    pipeline_res = run_soft_launch_qc_pipeline(data150, sample_size_n=150, llm_client=client)
    adj = pipeline_res["adjudication_report"]

    # Acceptance Criterion: Granular per-market recommendation
    assert "Go for 3 markets" in adj["headline"]
    assert "Hold 2 markets" in adj["headline"]
    assert adj["overall_recommendation"] == "HOLD"

    # Per-market breakdown
    decisions = adj["market_decisions"]
    assert len(decisions) == 5, f"Expected 5 markets in decision map, got {len(decisions)}"

    # Clean markets (English, French, German) must be GO
    assert decisions["9"]["status"] == "GO"
    assert decisions["18"]["status"] == "GO"
    assert decisions["31"]["status"] == "GO"

    # Blocked markets (Portuguese 1046, Spanish 2058) must be HOLD
    assert decisions["1046"]["status"] == "HOLD"
    assert decisions["2058"]["status"] == "HOLD"


def test_adjudicator_power_gate_and_blocked_categories_audit():
    """Verify Adjudicator transparently audits power-gate suppressions and blocked categories."""
    with open(RESPONSES_N150_PATH) as f:
        data150 = json.load(f)["items"]

    client = LLMClient()
    pipeline_res = run_soft_launch_qc_pipeline(data150, sample_size_n=150, llm_client=client)
    adj = pipeline_res["adjudication_report"]

    # 1. Power Gate Audit stated out loud
    audit = adj["power_gate_audit"]
    assert "statement" in audit
    assert "250 checks passed, 6 flagged, 83 not yet valid at n=150" in audit["statement"]
    assert audit["suppressed_checks_count"] == 83

    # 2. Blocked Categories Audit
    blocked = adj["blocked_categories_audit"]
    assert "routing_conformance" in blocked
    assert "quota_conformance" in blocked
    assert blocked["routing_conformance"]["status"] == "BLOCKED"
    assert "Word questionnaire" in blocked["routing_conformance"]["reason"]
    assert blocked["quota_conformance"]["status"] == "BLOCKED"
    assert "quota" in blocked["quota_conformance"]["reason"].lower()


def test_pipeline_comparison_n45_vs_n150():
    """Verify side-by-side run demonstrating dynamic power gate behavior (5 vs 6 flagged)."""
    with open(RESPONSES_N45_PATH) as f:
        data45 = json.load(f)["items"]
    with open(RESPONSES_N150_PATH) as f:
        data150 = json.load(f)["items"]

    # Run n=45 pipeline
    client45 = LLMClient()
    res45 = run_soft_launch_qc_pipeline(data45, sample_size_n=45, llm_client=client45)
    triage45 = res45["triage_report"]

    # Run n=150 pipeline
    client150 = LLMClient()
    res150 = run_soft_launch_qc_pipeline(data150, sample_size_n=150, llm_client=client150)
    triage150 = res150["triage_report"]

    # At n=45: disposition_mix is suppressed (84 suppressed, 5 flagged)
    assert res45["check_runner_summary"]["suppressed_checks"] == 84
    assert len(triage45["tasks"]) == 5
    assert "250 checks passed, 5 flagged, 84 not yet valid at n=45" in res45["power_gate_statement"]

    # At n=150: disposition_mix runs (83 suppressed, 6 flagged)
    assert res150["check_runner_summary"]["suppressed_checks"] == 83
    assert len(triage150["tasks"]) == 6
    assert "250 checks passed, 6 flagged, 83 not yet valid at n=150" in res150["power_gate_statement"]

    # Architecture verification: Check runner has 0 calls; Triage has 1; Adjudicator has 1
    assert triage45["model_calls"] == 1
    assert triage150["model_calls"] == 1
    assert res45["adjudication_report"]["model_calls"] == 1
    assert res150["adjudication_report"]["model_calls"] == 1


if __name__ == "__main__":
    test_triage_single_call_and_prioritization()
    test_triage_empty_anomalies()
    test_adjudicator_deduplication_and_severity()
    test_adjudicator_per_market_go_no_go()
    test_adjudicator_power_gate_and_blocked_categories_audit()
    test_pipeline_comparison_n45_vs_n150()
    print("All Step 7 Triage and Adjudicator tests passed successfully!")

