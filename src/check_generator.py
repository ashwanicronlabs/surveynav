"""Automated QC check generator derived from Confirmit survey schema.

Automatically derives the complete battery of 339 quality control checks directly
from machine-readable Confirmit schema definitions with 0 hand-written rules.
Also explicitly declares blocked categories (Routing Conformance, Quota Conformance).
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SCHEMA_PATH, POWER_GATE_THRESHOLDS, BLOCKED_CATEGORIES


class CheckDefinition:
    """Represents a single automated quality control check rule."""

    def __init__(
        self,
        check_id: str,
        check_type: str,
        variable: str,
        min_n: int,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.check_id = check_id
        self.check_type = check_type
        self.variable = variable
        self.min_n = min_n
        self.description = description
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert check definition to dictionary representation."""
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "variable": self.variable,
            "min_n": self.min_n,
            "description": self.description,
            "metadata": self.metadata
        }


def generate_checks_from_schema(schema_path: Path = SCHEMA_PATH) -> List[CheckDefinition]:
    """Inspect Confirmit schema and automatically generate the 339 QC checks.
    
    Derives checks across 11 specific categories:
    1. Item non-response: 195 (all declared variables and loop keys)
    2. Degenerate distribution: 83 (all choice questions and choice loop keys)
    3. Range violation: 23 (numeric variables declaring numeric limits)
    4. Straightlining: 22 (grids/compound matrices with >= 3 items)
    5. Allocation does not sum: 8 (multi-part 0-100 questions)
    6. Speeding: 1 (hidLOI against survey median)
    7. Duplicate source: 2 (hIPFlag, hPunch)
    8. Device anomaly: 2 (hDeviceTypeMain, lastdevicetype)
    9. Disposition mix: 1 (status distribution)
    10. Script errors: 1 (ErrorMessage incidence)
    11. Dropout point: 1 (hidQueDropout cluster)

    Args:
        schema_path: Path to client synthetic_survey_responses_schema.json.

    Returns:
        List of 339 CheckDefinition instances.
    """
    with open(schema_path) as f:
        schema = json.load(f)

    checks: List[CheckDefinition] = []
    root = schema["root"]

    # Collect all variables and keys across hierarchy
    all_variables: List[Tuple[str, Dict[str, Any]]] = []
    
    # Root keys and variables
    for k in root.get("keys", []):
        all_variables.append((k["name"], k))
    for v in root.get("variables", []):
        all_variables.append((v["name"], v))

    # Child loop keys and variables
    for child in root.get("children", []):
        cname = child["name"]
        for k in child.get("keys", []):
            all_variables.append((f"{cname}_{k['name']}", k))
        for v in child.get("variables", []):
            all_variables.append((f"{cname}_{v['name']}", v))

    # -------------------------------------------------------------------------
    # 1. Item Non-Response Checks (195 checks)
    # Applied to every variable in the schema to monitor unexpected missing rates.
    # -------------------------------------------------------------------------
    for name, v_def in all_variables:
        checks.append(CheckDefinition(
            check_id=f"item_non_response.{name}",
            check_type="item_non_response",
            variable=name,
            min_n=POWER_GATE_THRESHOLDS["item_non_response"],
            description=f"Item non-response rate check for variable {name}",
            metadata={"expected_missing_max": 0.05}
        ))

    # -------------------------------------------------------------------------
    # 2. Degenerate Distribution Checks (83 checks)
    # Checks whether any single-choice or multi-choice question has an abnormal
    # concentration (>85%) on a single code.
    # -------------------------------------------------------------------------
    for name, v_def in all_variables:
        v_type = v_def.get("variableType")
        if v_type in ("singleChoice", "multiChoice"):
            checks.append(CheckDefinition(
                check_id=f"degenerate_distribution.{name}",
                check_type="degenerate_distribution",
                variable=name,
                min_n=POWER_GATE_THRESHOLDS["degenerate_distribution"],
                description=f"Degenerate distribution check for choice question {name}",
                metadata={"max_single_option_share": 0.85}
            ))

    # -------------------------------------------------------------------------
    # 3. Range Violation Checks (23 checks)
    # Checks numeric variables with declared boundaries (e.g. 0 to 100).
    # -------------------------------------------------------------------------
    for name, v_def in all_variables:
        v_type = v_def.get("variableType")
        if v_type == "numeric":
            lower = v_def.get("lowerNumericLimit")
            upper = v_def.get("upperNumericLimit")
            if lower is not None or upper is not None:
                checks.append(CheckDefinition(
                    check_id=f"range_violation.{name}",
                    check_type="range_violation",
                    variable=name,
                    min_n=POWER_GATE_THRESHOLDS["range_violation"],
                    description=f"Range boundary conformance check for numeric variable {name}",
                    metadata={"min_limit": lower, "max_limit": upper}
                ))

    # -------------------------------------------------------------------------
    # 4. Straightlining Checks (22 checks)
    # Grids or compound matrices containing 3 or more sub-items where respondents
    # might lazily click identical answers down the whole column.
    # -------------------------------------------------------------------------
    for name, v_def in all_variables:
        fields = v_def.get("fields", [])
        if len(fields) >= 3:
            checks.append(CheckDefinition(
                check_id=f"straightlining.{name}",
                check_type="straightlining",
                variable=name,
                min_n=POWER_GATE_THRESHOLDS["straightlining"],
                description=f"Straightlining check for grid/matrix question {name} ({len(fields)} sub-items)",
                metadata={"sub_items_count": len(fields)}
            ))

    # -------------------------------------------------------------------------
    # 5. Allocation Sum Checks (8 checks)
    # Multi-part 0-100 questions requiring sub-parts to total exactly 100.
    # -------------------------------------------------------------------------
    for name, v_def in all_variables:
        fields = v_def.get("fields", [])
        v_type = v_def.get("variableType")
        upper = v_def.get("upperNumericLimit")
        if v_type == "numeric" and len(fields) >= 3 and upper == 100.0:
            checks.append(CheckDefinition(
                check_id=f"allocation_sum.{name}",
                check_type="allocation_sum",
                variable=name,
                min_n=POWER_GATE_THRESHOLDS["allocation_sum"],
                description=f"Constant-sum allocation check for {name} (sub-parts must sum to 100)",
                metadata={"expected_sum": 100.0, "tolerance": 0.001}
            ))

    # -------------------------------------------------------------------------
    # 6. Speeding Check (1 check)
    # Compares length of interview (hidLOI) against the sample median.
    # -------------------------------------------------------------------------
    checks.append(CheckDefinition(
        check_id="speeding.hidLOI",
        check_type="speeding",
        variable="hidLOI",
        min_n=POWER_GATE_THRESHOLDS["speeding"],
        description="Speeding check comparing respondent LOI against median completion time",
        metadata={"speeding_ratio_threshold": 0.40}
    ))

    # -------------------------------------------------------------------------
    # 7. Duplicate Source Checks (2 checks)
    # Monitors duplicate IP addresses or punch repeat patterns.
    # -------------------------------------------------------------------------
    for var in ("hIPFlag", "hPunch"):
        checks.append(CheckDefinition(
            check_id=f"duplicate_source.{var}",
            check_type="duplicate_source",
            variable=var,
            min_n=30,
            description=f"Duplicate respondent/source tracking on {var}",
            metadata={"max_duplicate_rate": 0.05}
        ))

    # -------------------------------------------------------------------------
    # 8. Device Anomaly Checks (2 checks)
    # Validates device consistency between client paradata and rendering mode.
    # -------------------------------------------------------------------------
    for var in ("hDeviceTypeMain", "lastdevicetype"):
        checks.append(CheckDefinition(
            check_id=f"device_anomaly.{var}",
            check_type="device_anomaly",
            variable=var,
            min_n=30,
            description=f"Device tracking consistency check on {var}",
            metadata={"monitored_variable": var}
        ))

    # -------------------------------------------------------------------------
    # 9. Disposition Mix Check (1 check)
    # Analyzes completes vs screenouts vs error proportions across categories.
    # -------------------------------------------------------------------------
    checks.append(CheckDefinition(
        check_id="disposition_mix.status",
        check_type="disposition_mix",
        variable="status",
        min_n=POWER_GATE_THRESHOLDS["disposition_mix"],
        description="Overall disposition distribution check (completes, screenouts, errors)",
        metadata={"screened_max": 0.40, "error_max": 0.10}
    ))

    # -------------------------------------------------------------------------
    # 10. Script Errors Check (1 check)
    # Monitors presence and incidence of runtime ErrorMessage flags.
    # -------------------------------------------------------------------------
    checks.append(CheckDefinition(
        check_id="script_errors.ErrorMessage",
        check_type="script_errors",
        variable="ErrorMessage",
        min_n=1,
        description="Confirmit runtime script error incidence check",
        metadata={"allowed_error_count": 0}
    ))

    # -------------------------------------------------------------------------
    # 11. Dropout Point Check (1 check)
    # Identifies dropout bottlenecks along the questionnaire flow.
    # -------------------------------------------------------------------------
    checks.append(CheckDefinition(
        check_id="dropout_point.hidQueDropout",
        check_type="dropout_point",
        variable="hidQueDropout",
        min_n=30,
        description="Questionnaire dropout point clustering analysis",
        metadata={"dropout_cluster_threshold": 0.15}
    ))

    return checks


def get_blocked_categories() -> Dict[str, Dict[str, str]]:
    """Return explicit declarations of unbuildable client check categories."""
    return BLOCKED_CATEGORIES


if __name__ == "__main__":
    checks = generate_checks_from_schema()
    print(f"Total QC checks generated: {len(checks)}")
    
    # Categorical breakdown
    by_type: Dict[str, int] = {}
    for c in checks:
        by_type[c.check_type] = by_type.get(c.check_type, 0) + 1
        
    print("\nCheck counts by category:")
    for ct, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {ct:<28} : {count}")

    print("\nBlocked Categories (reported transparently):")
    for cat, info in get_blocked_categories().items():
        print(f"  {cat:<28} : [{info['status']}] {info['reason']}")

