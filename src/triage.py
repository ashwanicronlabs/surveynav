"""Triage layer for Soft-Launch Quality Control anomalies.

Executes exactly 1 model call to analyze and prioritize flagged anomalies
before spawning autonomous investigator agents. Ranks anomalies into P1, P2, P3
priority bands, formulates initial exploratory hypotheses, and determines
investigator dispatch order.
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm_adapter import LLMClient, LLMResponse


TRIAGE_SYSTEM_PROMPT = """You are the Senior QC Triage Specialist at Inizio Ignite.
Your job is to review anomalous quality control checks flagged by the rules engine
and organize them into an actionable, prioritized dispatch plan for autonomous investigators.

MANDATE & CONSTRAINTS:
1. Make EXACTLY 1 decision call (you are not an autonomous multi-step agent).
2. Rank each anomaly into priority bands:
   - P1 (Critical): Survey runtime defects, validation failures, or core data corruption
     (e.g., allocation sums != 100, out-of-range numeric entries, core item non-response).
   - P2 (High): Sample composition, vendor quality, screening distortions, or device skew
     (e.g., speeding clusters, screener over-rejections, device discrepancies).
   - P3 (Medium): Inattentive respondent patterns (e.g., straightlining on item grids).
3. Formulate an initial hypothesis for each anomaly to give the investigator a starting point.
4. Return your output as clean, valid JSON matching the requested schema.
"""


class TriageItem:
    """Represents a single prioritized anomaly ready for investigator dispatch."""

    def __init__(
        self,
        dispatch_id: str,
        priority: str,
        anomaly: Dict[str, Any],
        target_variable: str,
        check_type: str,
        initial_hypothesis: str,
        rationale: str
    ):
        """Initialize a triaged task item.
        
        Args:
            dispatch_id: Unique task identifier (e.g. 'TASK-001').
            priority: 'P1', 'P2', or 'P3'.
            anomaly: Raw anomaly dictionary from check_runner.
            target_variable: Name of variable under test.
            check_type: Category of QC check.
            initial_hypothesis: Starting theory for the investigator.
            rationale: Explanation of assigned priority level.
        """
        self.dispatch_id = dispatch_id
        self.priority = priority
        self.anomaly = anomaly
        self.check_id = anomaly.get("check_id", "")
        self.target_variable = target_variable
        self.check_type = check_type
        self.initial_hypothesis = initial_hypothesis
        self.rationale = rationale

    def to_dict(self) -> Dict[str, Any]:
        """Convert triage item to dictionary."""
        return {
            "dispatch_id": self.dispatch_id,
            "priority": self.priority,
            "target_variable": self.target_variable,
            "check_type": self.check_type,
            "check_id": self.anomaly.get("check_id"),
            "initial_hypothesis": self.initial_hypothesis,
            "rationale": self.rationale,
            "anomaly": self.anomaly
        }


class TriageReport:
    """Standardized report produced by the Triage phase."""

    def __init__(
        self,
        total_anomalies: int,
        tasks: List[TriageItem],
        summary: str,
        priority_counts: Dict[str, int],
        token_usage: Dict[str, Any],
        model_calls: int = 1
    ):
        """Initialize triage summary report.
        
        Args:
            total_anomalies: Total number of input anomalies.
            tasks: Prioritized list of TriageItem instances.
            summary: Textual executive summary of triage outcome.
            priority_counts: Breakdown of tasks by priority level.
            token_usage: Token and cost accounting dictionary.
            model_calls: Number of LLM calls made (strictly 1).
        """
        self.total_anomalies = total_anomalies
        self.tasks = tasks
        self.summary = summary
        self.priority_counts = priority_counts
        self.token_usage = token_usage
        self.model_calls = model_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert triage report to dictionary."""
        return {
            "total_anomalies": self.total_anomalies,
            "model_calls": self.model_calls,
            "priority_counts": self.priority_counts,
            "summary": self.summary,
            "token_usage": self.token_usage,
            "tasks": [t.to_dict() for t in self.tasks]
        }


def triage_anomalies(
    anomalies: List[Dict[str, Any]],
    sample_size_n: int,
    llm_client: Optional[LLMClient] = None
) -> TriageReport:
    """Analyze and rank anomalies using a single model call.
    
    Args:
        anomalies: Flagged anomaly records from check_runner.py.
        sample_size_n: Sample size n of current soft-launch batch (e.g. 45 or 150).
        llm_client: LLM client for executing the single model call.

    Returns:
        TriageReport containing prioritized tasks and initial hypotheses.
    """
    client = llm_client or LLMClient()

    # If no anomalies were flagged, return an empty triage report
    if not anomalies:
        return TriageReport(
            total_anomalies=0,
            tasks=[],
            summary=f"No anomalies flagged at n={sample_size_n}. All active QC checks passed.",
            priority_counts={"P1": 0, "P2": 0, "P3": 0},
            token_usage=client.tracker.summary(),
            model_calls=0
        )

    # Format user prompt for the single model call
    prompt_payload = {
        "sample_size_n": sample_size_n,
        "anomalies_count": len(anomalies),
        "anomalies": [
            {
                "check_id": a.get("check_id"),
                "check_type": a.get("check_type"),
                "variable": a.get("variable"),
                "statistic": a.get("statistic"),
                "expected": a.get("expected"),
                "description": a.get("description")
            }
            for a in anomalies
        ]
    }
    user_prompt = (
        f"Soft-launch batch has n={sample_size_n} completes. The following {len(anomalies)} anomalies "
        f"were flagged by QC checks:\n{json.dumps(prompt_payload, indent=2)}\n\n"
        "Rank these into P1, P2, P3 bands, formulate initial hypotheses, and return JSON."
    )

    # Execute exactly 1 model call
    response: LLMResponse = client.generate_single_call(
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        task_type="triage",
        context={"anomalies": anomalies, "sample_size_n": sample_size_n}
    )

    # Parse response content
    parsed_json: Dict[str, Any] = {}
    try:
        parsed_json = json.loads(response.content)
    except Exception:
        # Fallback parsing in case of non-JSON text output
        parsed_json = {}

    tasks_data = parsed_json.get("tasks", [])
    triaged_items: List[TriageItem] = []

    # Map parsed tasks back to full anomaly records
    anomaly_by_cid = {a.get("check_id"): a for a in anomalies}

    if tasks_data:
        for t_info in tasks_data:
            cid = t_info.get("check_id")
            orig_anomaly = anomaly_by_cid.get(cid, {})
            # Fallback to variable match if check_id differs
            if not orig_anomaly:
                orig_anomaly = next((a for a in anomalies if a.get("variable") == t_info.get("target_variable")), anomalies[0])

            triaged_items.append(TriageItem(
                dispatch_id=t_info.get("dispatch_id", f"TASK-{len(triaged_items)+1:03d}"),
                priority=t_info.get("priority", "P2"),
                anomaly=orig_anomaly,
                target_variable=t_info.get("target_variable") or orig_anomaly.get("variable", ""),
                check_type=t_info.get("check_type") or orig_anomaly.get("check_type", ""),
                initial_hypothesis=t_info.get("initial_hypothesis", "Investigate potential survey or data inconsistency."),
                rationale=t_info.get("rationale", "Prioritized based on soft-launch QC impact.")
            ))
    else:
        # Deterministic fallback triage logic if model output was unparseable
        for idx, a in enumerate(anomalies, 1):
            ct = a.get("check_type", "")
            var = a.get("variable", "")
            if ct in ("allocation_sum", "range_violation", "item_non_response"):
                prio = "P1"
                rat = f"Critical questionnaire data integrity defect on {var}."
                hyp = f"Question {var} may have defective validation or client-side UI failure."
            elif ct in ("speeding", "disposition_mix", "device_anomaly"):
                prio = "P2"
                rat = f"Significant sample quality or market composition anomaly on {var}."
                hyp = f"Distribution anomaly on {var} indicates panel vendor quality or quota mismatch."
            else:
                prio = "P3"
                rat = f"Behavioral or inattentive respondent pattern on {var}."
                hyp = f"Straightlining on {var} requires inattention review."

            triaged_items.append(TriageItem(
                dispatch_id=f"TASK-{idx:03d}",
                priority=prio,
                anomaly=a,
                target_variable=var,
                check_type=ct,
                initial_hypothesis=hyp,
                rationale=rat
            ))

    # Priority counts
    p_counts = {
        "P1": sum(1 for t in triaged_items if t.priority == "P1"),
        "P2": sum(1 for t in triaged_items if t.priority == "P2"),
        "P3": sum(1 for t in triaged_items if t.priority == "P3")
    }

    summary_text = parsed_json.get(
        "summary",
        f"Triaged {len(anomalies)} anomalies into {len(triaged_items)} prioritized investigative tasks ({p_counts['P1']} P1, {p_counts['P2']} P2, {p_counts['P3']} P3)."
    )

    return TriageReport(
        total_anomalies=len(anomalies),
        tasks=triaged_items,
        summary=summary_text,
        priority_counts=p_counts,
        token_usage=client.tracker.summary(),
        model_calls=1
    )


if __name__ == "__main__":
    from src.config import RESPONSES_N150_PATH
    from src.check_runner import run_checks

    print("=================================================================")
    print("STEP 7: TRIAGE TEST RUN (1 MODEL CALL)")
    print("=================================================================")

    with open(RESPONSES_N150_PATH) as f:
        data150 = json.load(f)["items"]

    res = run_checks(data150)
    anomalies = res["anomalies"]
    print(f"Input: {len(anomalies)} anomalies flagged by rules engine.")

    report = triage_anomalies(anomalies, sample_size_n=150)
    print(f"\nTriage Summary: {report.summary}")
    print(f"Model Calls:    {report.model_calls} (strictly single-call)")
    print(f"Priority Breakdown: {report.priority_counts}")
    print("\nDispatch Tasks:")
    for task in report.tasks:
        print(f"  [{task.priority}] {task.dispatch_id}: {task.check_id} ({task.target_variable})")
        print(f"       Hypothesis: {task.initial_hypothesis}")
        print(f"       Rationale:  {task.rationale}")

    print("\nToken Consumption:")
    print(report.token_usage)
