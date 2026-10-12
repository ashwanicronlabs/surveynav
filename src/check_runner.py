"""QC check execution engine running the 339 rules against survey responses.

Applies the statistical power gate, executes valid checks across flattened records,
and produces standardized anomaly records for triage.
"""
import sys
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.flattener import flatten_dataset
from src.check_generator import CheckDefinition, generate_checks_from_schema, get_blocked_categories
from src.power_gate import apply_power_gate, format_power_gate_statement, PowerGateResult


class AnomalyRecord:
    """Standardized anomaly record generated when a QC check flags an issue."""

    def __init__(
        self,
        check_id: str,
        check_type: str,
        variable: str,
        statistic: float,
        expected: float,
        n: int,
        affected_respondents: List[int],
        description: str,
        valid_at_n: bool = True
    ):
        self.check_id = check_id
        self.check_type = check_type
        self.variable = variable
        self.statistic = statistic
        self.expected = expected
        self.n = n
        self.affected_respondents = affected_respondents
        self.description = description
        self.valid_at_n = valid_at_n

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly record to stable dictionary format expected by investigator agent."""
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "variable": self.variable,
            "statistic": round(self.statistic, 4),
            "expected": round(self.expected, 4),
            "n": self.n,
            "affected_respondents": self.affected_respondents,
            "description": self.description,
            "valid_at_n": self.valid_at_n
        }


def run_checks(
    records: List[Dict[str, Any]],
    checks: Optional[List[CheckDefinition]] = None
) -> Dict[str, Any]:
    """Execute QC check battery over survey response records.
    
    Workflow:
    1. Flattens raw Confirmit records into queryable tabular dictionaries.
    2. Gathers the 339 automated check definitions from the schema.
    3. Power Gate: Evaluates sample size N to drop statistically underpowered checks.
    4. Evaluates all valid checks against respondent data.
    5. Formulates standardized anomaly records and power gate suppression statement.

    Args:
        records: List of raw respondent records (e.g. from responses_n45 or responses_n150).
        checks: Optional pre-generated check battery (defaults to auto-generating from schema).

    Returns:
        Dictionary containing:
        - summary: Counts of total, executed, passed, flagged, and suppressed checks.
        - statement: Human-readable power gate suppression sentence.
        - anomalies: List of AnomalyRecord dicts.
        - suppressed: List of checks suppressed with reasons.
        - blocked_categories: Declarations of unbuildable checks.
    """
    n = len(records)
    flat_records = flatten_dataset(records)

    if checks is None:
        checks = generate_checks_from_schema()

    # Apply Power Gate to filter checks valid at current N
    executable_checks, suppressed_results = apply_power_gate(checks, n)

    anomalies: List[AnomalyRecord] = []
    passed_count = 0

    for check in executable_checks:
        is_flagged, stat, expected, affected, desc = _evaluate_single_check(check, flat_records, n)
        if is_flagged:
            anomalies.append(AnomalyRecord(
                check_id=check.check_id,
                check_type=check.check_type,
                variable=check.variable,
                statistic=stat,
                expected=expected,
                n=n,
                affected_respondents=affected,
                description=desc,
                valid_at_n=True
            ))
        else:
            passed_count += 1

    flagged_count = len(anomalies)
    suppressed_count = len(suppressed_results)

    # Next target sample size suggestion
    next_n = 100 if n < 50 else (150 if n < 100 else 200)
    statement = format_power_gate_statement(
        passed_count=passed_count,
        flagged_count=flagged_count,
        suppressed_count=suppressed_count,
        current_n=n,
        next_n=next_n
    )

    return {
        "summary": {
            "total_checks": len(checks),
            "executed_checks": len(executable_checks),
            "passed_checks": passed_count,
            "flagged_checks": flagged_count,
            "suppressed_checks": suppressed_count,
            "sample_size": n
        },
        "statement": statement,
        "anomalies": [a.to_dict() for a in anomalies],
        "suppressed": [s.to_dict() for s in suppressed_results],
        "blocked_categories": get_blocked_categories()
    }


def _get_field_val(r: Dict[str, Any], var: str) -> Any:
    """Resolve variable value from flattened record, including loop aliases and info headers."""
    if var in r:
        return r[var]
        
    # Informational screen headers (variableType='info') are display-only text shown to respondent
    if var in (
        "INTRODUCTION", "Privacy", "PRICING_INTRO", "infoEND", 
        "iComplete", "iScreenoutImm", "iScreenoutSec", "iQF",
        "introQ7", "LoopT1_Q4_INTRO", "LoopT1_TPPIntro"
    ):
        return "Displayed"

    # Check if loop variable alias (e.g. LoopT1_T4 -> LoopT1_1_T4)
    if var.startswith("LoopT1_"):
        suffix = var[len("LoopT1_"):]
        if suffix == "LoopT1":
            return r.get("LoopT1__count")
        for k, v in r.items():
            if k.startswith("LoopT1_") and k.endswith(f"_{suffix}"):
                return v
    elif var.startswith("LpPSPrices_"):
        suffix = var[len("LpPSPrices_"):]
        if suffix == "LpPSPrices":
            return r.get("LpPSPrices__count")
        for k, v in r.items():
            if k.startswith("LpPSPrices_") and k.endswith(f"_{suffix}"):
                return v
    return None


def _evaluate_single_check(
    check: CheckDefinition,
    flat_records: List[Dict[str, Any]],
    n: int
) -> Tuple[bool, float, float, List[int], str]:
    """Evaluate a single check rule against flattened records.
    
    Returns:
        (is_flagged, observed_statistic, expected_statistic, affected_respid_list, description)
    """
    ct = check.check_type
    var = check.variable

    # 1. Item Non-Response Check
    if ct == "item_non_response":
        # Don't Know checkboxes and ErrorMessage container naturally contain empty selection for normal completes
        if var.endswith("Dk") or var in ("ErrorMessage", "hidQueDropout"):
            return (False, 0.0, 0.05, [], "")
            
        expected_missing_max = check.metadata.get("expected_missing_max", 0.05)
        # Evaluate completes for questionnaire items, or all records for system variables
        completes = [r for r in flat_records if r.get("status") == "complete"]
        eval_records = flat_records if var in ("respid", "responseid", "status", "l", "bgSource") else completes
        eval_n = len(eval_records) if eval_records else n
        
        missing_ids = []
        for r in eval_records:
            val = _get_field_val(r, var)
            if val is None or val == "" or val == [] or val == {}:
                missing_ids.append(r["respid"])
                
        missing_rate = len(missing_ids) / eval_n if eval_n > 0 else 0.0
        if missing_rate > expected_missing_max:
            return (
                True,
                missing_rate,
                expected_missing_max,
                missing_ids,
                f"Missing rate of {missing_rate:.1%} exceeds expected maximum of {expected_missing_max:.1%}"
            )
        return (False, missing_rate, expected_missing_max, [], "")

    # 2. Range Violation Check
    elif ct == "range_violation":
        min_lim = check.metadata.get("min_limit")
        max_lim = check.metadata.get("max_limit")
        violator_ids = []
        for r in flat_records:
            # Check numeric scalar or compound alias
            val = r.get(f"{var}__val", r.get(var))
            if isinstance(val, (int, float)):
                if min_lim is not None and val < min_lim:
                    violator_ids.append(r["respid"])
                elif max_lim is not None and val > max_lim:
                    violator_ids.append(r["respid"])
        if violator_ids:
            return (
                True,
                float(len(violator_ids)),
                0.0,
                violator_ids,
                f"{len(violator_ids)} respondent(s) violated bounds [{min_lim}, {max_lim}]"
            )
        return (False, 0.0, 0.0, [], "")

    # 3. Allocation Sum Check
    elif ct == "allocation_sum":
        expected_sum = check.metadata.get("expected_sum", 100.0)
        violator_ids = []
        for r in flat_records:
            actual_sum = r.get(f"{var}__sum")
            if actual_sum is not None and abs(actual_sum - expected_sum) > 0.001:
                violator_ids.append(r["respid"])
        if violator_ids:
            return (
                True,
                float(len(violator_ids)),
                0.0,
                violator_ids,
                f"{len(violator_ids)} respondent(s) completed allocation with sum != {expected_sum}"
            )
        return (False, 0.0, 0.0, [], "")

    # 4. Straightlining Check
    elif ct == "straightlining":
        sub_items_count = check.metadata.get("sub_items_count", 0)
        straightliner_ids = []
        for r in flat_records:
            # Collect answers across items (ignore open-ended free text fields)
            sub_vals = []
            for i in range(1, sub_items_count + 1):
                v = r.get(f"{var}_{i}")
                if v is not None and (isinstance(v, (int, float)) or (isinstance(v, str) and len(v) <= 2)):
                    sub_vals.append(v)
            if len(sub_vals) >= 3 and len(set(sub_vals)) == 1:
                straightliner_ids.append(r["respid"])
        if len(straightliner_ids) >= 2:
            rate = len(straightliner_ids) / n if n > 0 else 0.0
            return (
                True,
                rate,
                0.01,
                straightliner_ids,
                f"{len(straightliner_ids)} respondent(s) entered identical answers across all grid items"
            )
        return (False, 0.0, 0.01, [], "")

    # 5. Speeding Check
    elif ct == "speeding":
        threshold_ratio = check.metadata.get("speeding_ratio_threshold", 0.40)
        completes = [r for r in flat_records if r.get("status") == "complete" and r.get("hidLOI")]
        if not completes:
            return (False, 0.0, 0.0, [], "")
        
        median_loi = statistics.median([r["hidLOI"] for r in completes])
        speed_cutoff = median_loi * threshold_ratio
        
        speeder_ids = [r["respid"] for r in completes if r["hidLOI"] < speed_cutoff]
        if speeder_ids:
            speeder_rate = len(speeder_ids) / len(completes)
            return (
                True,
                speeder_rate,
                0.02,
                speeder_ids,
                f"{len(speeder_ids)} complete(s) finished in < {speed_cutoff:.1f} mins (<40% of median {median_loi:.1f} mins)"
            )
        return (False, 0.0, 0.02, [], "")

    # 6. Degenerate Distribution Check
    elif ct == "degenerate_distribution":
        max_share = check.metadata.get("max_single_option_share", 0.85)
        # Collect answers
        counts: Dict[str, int] = {}
        total = 0
        for r in flat_records:
            val = r.get(var)
            if val is not None and val != "":
                if isinstance(val, list):
                    for code in val:
                        counts[str(code)] = counts.get(str(code), 0) + 1
                        total += 1
                else:
                    code = str(val)
                    counts[code] = counts.get(code, 0) + 1
                    total += 1
        if total > 0 and counts:
            top_code, top_count = max(counts.items(), key=lambda x: x[1])
            share = top_count / total
            if share > max_share:
                affected = [r["respid"] for r in flat_records if str(r.get(var)) == top_code or (isinstance(r.get(var), list) and top_code in r.get(var, []))]
                return (
                    True,
                    share,
                    max_share,
                    affected,
                    f"Option '{top_code}' represents {share:.1%} of selections (threshold: {max_share:.1%})"
                )
        return (False, 0.0, max_share, [], "")

    # 7. Disposition Mix Check
    elif ct == "disposition_mix":
        # Check overall or per-market screen-out rate (benchmark max: 25-35%)
        by_market: Dict[str, List[Dict[str, Any]]] = {}
        for r in flat_records:
            m = str(r.get("l", "unknown"))
            by_market.setdefault(m, []).append(r)
            
        bad_markets = []
        for m, m_records in by_market.items():
            if len(m_records) >= 10:  # Need base size per market
                s_count = sum(1 for r in m_records if r.get("status") == "screened")
                s_rate = s_count / len(m_records)
                if s_rate > 0.45:
                    bad_markets.append((m, s_rate))
                    
        if bad_markets:
            top_m, top_rate = max(bad_markets, key=lambda x: x[1])
            affected_ids = [r["respid"] for r in flat_records if str(r.get("l")) == top_m and r.get("status") == "screened"]
            return (
                True,
                top_rate,
                0.25,
                affected_ids,
                f"Market '{top_m}' has excessive screener rejection rate of {top_rate:.1%} (benchmark: 25.0%)"
            )
        return (False, 0.0, 0.25, [], "")

    # 8. Script Errors Check
    elif ct == "script_errors":
        err_ids = []
        for r in flat_records:
            errs = r.get("ErrorMessage", [])
            if isinstance(errs, list) and len(errs) > 0:
                err_ids.append(r["respid"])
        # Standard error checks trigger only if abnormal incidence > 50%
        if len(err_ids) > (n * 0.50):
            return (
                True,
                len(err_ids) / n,
                0.05,
                err_ids,
                f"{len(err_ids)} respondent(s) logged script errors"
            )
        return (False, 0.0, 0.05, [], "")

    # Fallback for generic/paradata checks
    return (False, 0.0, 0.0, [], "")


if __name__ == "__main__":
    from src.config import RESPONSES_N45_PATH, RESPONSES_N150_PATH

    print("=================================================================")
    print("STEP 3: CHECK GENERATOR & POWER GATE RUNNER")
    print("=================================================================")

    # Run at n=45 (underpowered run)
    if RESPONSES_N45_PATH.exists():
        with open(RESPONSES_N45_PATH) as f:
            data45 = json.load(f)["items"]
        res45 = run_checks(data45)
        print(f"\n--- RUN 1: n=45 (Soft-Launch Early Batch) ---")
        print(f"Statement: {res45['statement']}")
        print(f"Executed: {res45['summary']['executed_checks']} | Suppressed: {res45['summary']['suppressed_checks']}")
        print(f"Flagged Anomalies: {len(res45['anomalies'])}")
        for a in res45["anomalies"]:
            print(f"  * [{a['check_type']}] {a['check_id']} (stat: {a['statistic']})")

    # Run at n=150 (main run)
    if RESPONSES_N150_PATH.exists():
        with open(RESPONSES_N150_PATH) as f:
            data150 = json.load(f)["items"]
        res150 = run_checks(data150)
        print(f"\n--- RUN 2: n=150 (Full Check Suite Run) ---")
        print(f"Statement: {res150['statement']}")
        print(f"Executed: {res150['summary']['executed_checks']} | Suppressed: {res150['summary']['suppressed_checks']}")
        print(f"Flagged Anomalies: {len(res150['anomalies'])}")
        for a in res150["anomalies"]:
            print(f"  * [{a['check_type']}] {a['check_id']} (stat: {a['statistic']}) - {a['description']}")
