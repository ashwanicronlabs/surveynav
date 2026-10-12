"""Test suite for Step 8: Ground Truth Benchmark Scoring and CLI Runner."""
import sys
import json
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESPONSES_N45_PATH, RESPONSES_N150_PATH, RUNS_DIR
from src.llm_adapter import LLMClient
from src.adjudicator import run_soft_launch_qc_pipeline
from src.scorer import score_adjudication_report, ScoringReport
from src.main import run_single_evaluation


def test_scorer_n150_benchmark_metrics():
    """Verify ground truth scoring metrics at n=150 (Detection rate 75%, Diagnosis 100%, 0 FP)."""
    with open(RESPONSES_N150_PATH) as f:
        data150 = json.load(f)["items"]

    client = LLMClient()
    pipeline_res = run_soft_launch_qc_pipeline(data150, sample_size_n=150, llm_client=client)
    adj_report = pipeline_res["adjudication_report"]

    score_report: ScoringReport = score_adjudication_report(
        adjudication_report=adj_report,
        sample_size_n=150,
        token_usage=pipeline_res["total_token_usage"]
    )

    # 1. Total planted faults must be 8
    assert score_report.total_planted_faults == 8, f"Expected 8 planted faults, got {score_report.total_planted_faults}"

    # 2. Detection Rate at n=150 (6 of 8 = 75.0%)
    assert score_report.detected_count == 6, f"Expected 6 detected faults at n=150, got {score_report.detected_count}"
    assert score_report.detection_rate == 0.75

    # 3. Diagnosis Accuracy (6 of 6 = 100.0%)
    assert score_report.diagnosed_count == 6, f"Expected 6 correctly diagnosed faults, got {score_report.diagnosed_count}"
    assert score_report.diagnosis_accuracy == 1.0

    # 4. Zero False Positives
    assert score_report.false_positives_count == 0, f"Expected 0 false positives, got {score_report.false_positives_count}"

    # 5. Power Gate Correctness (100%)
    assert score_report.power_gate_correctness == 1.0

    # 6. Verify detected faults include F01, F02, F03, F04, F05, F08
    detected_ids = [ev.fault_id for ev in score_report.fault_evaluations if ev.is_detected]
    for fid in ["F01", "F02", "F03", "F04", "F05", "F08"]:
        assert fid in detected_ids, f"Expected {fid} to be detected at n=150"

    # 7. Transparent Miss Reporting: Exactly F06 and F07 missed with stated reasons
    missed_ids = [m["fault_id"] for m in score_report.misses_breakdown]
    assert "F06" in missed_ids, "F06 must be in misses breakdown"
    assert "F07" in missed_ids, "F07 must be in misses breakdown"
    assert len(missed_ids) == 2, f"Expected 2 misses at n=150, got {len(missed_ids)}"

    f06_miss = next(m for m in score_report.misses_breakdown if m["fault_id"] == "F06")
    assert "BLOCKED" in f06_miss["reason"]
    assert "Word questionnaire" in f06_miss["reason"]

    f07_miss = next(m for m in score_report.misses_breakdown if m["fault_id"] == "F07")
    assert "SUPPRESSED" in f07_miss["reason"]
    assert "200" in f07_miss["reason"]


def test_scorer_n45_benchmark_metrics():
    """Verify ground truth scoring metrics at n=45 (5 of 8 detected, disposition_mix suppressed)."""
    with open(RESPONSES_N45_PATH) as f:
        data45 = json.load(f)["items"]

    client = LLMClient()
    pipeline_res = run_soft_launch_qc_pipeline(data45, sample_size_n=45, llm_client=client)
    adj_report = pipeline_res["adjudication_report"]

    score_report: ScoringReport = score_adjudication_report(
        adjudication_report=adj_report,
        sample_size_n=45,
        token_usage=pipeline_res["total_token_usage"]
    )

    # At n=45: 5 of 8 detected (62.5%)
    assert score_report.detected_count == 5, f"Expected 5 detected faults at n=45, got {score_report.detected_count}"
    assert score_report.detection_rate == 0.625
    assert score_report.diagnosed_count == 5
    assert score_report.diagnosis_accuracy == 1.0

    # Misses at n=45: F06 (blocked), F07 (suppressed at n<200), F08 (suppressed at n<50)
    missed_ids = [m["fault_id"] for m in score_report.misses_breakdown]
    assert len(missed_ids) == 3, f"Expected 3 misses at n=45, got {len(missed_ids)}"
    assert "F06" in missed_ids
    assert "F07" in missed_ids
    assert "F08" in missed_ids

    f08_miss = next(m for m in score_report.misses_breakdown if m["fault_id"] == "F08")
    assert "50" in f08_miss["reason"]


def test_main_evaluation_and_audit_traces():
    """Verify that run_single_evaluation executes cleanly and creates complete JSON audit files in runs/."""
    client = LLMClient()
    result = run_single_evaluation(sample_size_n=45, llm_client=client, save_trace=True)

    assert "trace_file" in result
    trace_path = Path(result["trace_file"])
    assert trace_path.exists(), f"Trace file was not created at {trace_path}"

    with open(trace_path) as f:
        trace_data = json.load(f)

    assert trace_data["sample_size_n"] == 45
    assert "timestamp" in trace_data
    assert "pipeline" in trace_data
    assert "score_report" in trace_data
    assert "adjudication_report" in trace_data["pipeline"]


if __name__ == "__main__":
    test_scorer_n150_benchmark_metrics()
    test_scorer_n45_benchmark_metrics()
    test_main_evaluation_and_audit_traces()
    print("All Step 8 Ground Truth Scoring and Main CLI tests passed successfully!")
