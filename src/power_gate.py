"""Statistical Power Gate for Soft-Launch Quality Control.

Enforces sample size validity gates prior to check execution to avoid noise
and false alarms on underpowered sample sizes (e.g. n=45 vs n=150).
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import POWER_GATE_THRESHOLDS
from src.check_generator import CheckDefinition


class PowerGateResult:
    """Evaluation result from the Power Gate for a specific check at sample size n."""

    def __init__(
        self,
        check: CheckDefinition,
        current_n: int,
        is_valid: bool,
        reason: str
    ):
        self.check = check
        self.current_n = current_n
        self.is_valid = is_valid
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert power gate evaluation to dictionary format."""
        return {
            "check_id": self.check.check_id,
            "check_type": self.check.check_type,
            "variable": self.check.variable,
            "current_n": self.current_n,
            "min_n": self.check.min_n,
            "valid_at_n": self.is_valid,
            "reason": self.reason
        }


def evaluate_check_power(check: CheckDefinition, current_n: int) -> PowerGateResult:
    """Evaluate whether a single QC check has sufficient statistical power at sample size n.
    
    Rules derived from survey research statistical constraints:
    - n = 1: Single-respondent factual rules (range violations, allocation sums, straightlining).
    - n = 30: Median comparisons & baseline proportion bounds (speeding, item non-response rate).
    - n = 50: Categorical proportions (disposition mix across 5 categories).
    - n = 200: Complex distribution shifts / degenerate choice distributions.

    Args:
        check: CheckDefinition instance to evaluate.
        current_n: Current count of collected completes/respondents.

    Returns:
        PowerGateResult specifying validity status and rationale.
    """
    min_required_n = check.min_n

    if current_n >= min_required_n:
        return PowerGateResult(
            check=check,
            current_n=current_n,
            is_valid=True,
            reason=f"Sample size n={current_n} >= {min_required_n} minimum required for {check.check_type}."
        )
    else:
        return PowerGateResult(
            check=check,
            current_n=current_n,
            is_valid=False,
            reason=(
                f"Statistically underpowered at n={current_n} (requires n>={min_required_n}). "
                f"Evaluating distribution or proportions below n={min_required_n} produces noisy false alarms."
            )
        )


def apply_power_gate(
    checks: List[CheckDefinition],
    current_n: int
) -> Tuple[List[CheckDefinition], List[PowerGateResult]]:
    """Partition checks into executable checks and suppressed checks based on power gating.
    
    Args:
        checks: Full battery of 339 CheckDefinition rules.
        current_n: Current sample size of the survey run.

    Returns:
        Tuple of:
        - executable_checks: List of CheckDefinition objects valid at current_n.
        - suppressed_results: List of PowerGateResult objects held back with stated reasons.
    """
    executable: List[CheckDefinition] = []
    suppressed: List[PowerGateResult] = []

    for c in checks:
        eval_result = evaluate_check_power(c, current_n)
        if eval_result.is_valid:
            executable.append(c)
        else:
            suppressed.append(eval_result)

    return executable, suppressed


def format_power_gate_statement(
    passed_count: int,
    flagged_count: int,
    suppressed_count: int,
    current_n: int,
    next_n: int = 100
) -> str:
    """Generate the client-required status sentence stating suppressions out loud.
    
    Example:
        '27 checks passed, 4 flagged, 9 not yet valid at n=45 — re-run at n=100.'
    """
    return (
        f"{passed_count} checks passed, {flagged_count} flagged, "
        f"{suppressed_count} not yet valid at n={current_n} — re-run at n={next_n}."
    )

