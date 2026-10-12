"""Tool suite and guardrails for the autonomous Investigator Agent.

Provides 5 strictly read-only tools:
1. query_data: Aggregates responses with top-20 row context truncation.
2. get_schema: Retrieves question types, choices, and boundaries.
3. sample_respondents: Fetches a handful of records to eyeball.
4. walk_survey: Steps through the in-memory survey state machine.
5. conclude: Enforces structured conclusion and termination.

Also implements safety guardrails:
- 25 tool call hard ceiling per investigation.
- 3x loop detection stopping repeated identical queries.
- Top-20 row truncation cap with omission notices.
"""
import sys
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SCHEMA_PATH, MAX_TOOL_CALLS, MAX_QUERY_ROWS, LOOP_DETECTION_LIMIT
from src.survey_model import walk_survey as execute_walk_survey
from src.flattener import flatten_dataset


# =============================================================================
# Tool Function Implementations
# =============================================================================

def get_schema(variable: str, schema_path: Path = SCHEMA_PATH) -> Dict[str, Any]:
    """Look up metadata, type, options, limits, and loop location for a survey question.
    
    Args:
        variable: Variable name to inspect (e.g. 'Q7Sk', 'S5', 'T14', 'Q9', 'LoopT1_T4').
        schema_path: Path to the Confirmit schema JSON.

    Returns:
        Dictionary containing variableType, question titles, declared numeric limits,
        available options, sub-item fields, and loop container if applicable.
    """
    clean_var = variable.strip()
    with open(schema_path) as f:
        schema = json.load(f)

    # Search root variables and keys
    root = schema.get("root", {})
    for item in root.get("variables", []) + root.get("keys", []):
        if item.get("name") == clean_var:
            return {
                "variable": clean_var,
                "found": True,
                "container": "root",
                "variableType": item.get("variableType"),
                "isCompound": item.get("isCompound", False),
                "lowerNumericLimit": item.get("lowerNumericLimit"),
                "upperNumericLimit": item.get("upperNumericLimit"),
                "fields": [f.get("code") for f in item.get("fields", [])],
                "options": [
                    {"code": o.get("code"), "text": o.get("texts", [{}])[0].get("text", "")}
                    for o in item.get("options", [])
                ],
                "titles": [t.get("text") for t in item.get("titles", [])]
            }

    # Search child loop blocks (LoopT1, LpPSPrices)
    for child in root.get("children", []):
        child_name = child.get("name")
        for item in child.get("variables", []) + child.get("keys", []):
            item_name = item.get("name")
            if item_name == clean_var or f"{child_name}_{item_name}" == clean_var:
                return {
                    "variable": clean_var,
                    "found": True,
                    "container": child_name,
                    "variableType": item.get("variableType"),
                    "isCompound": item.get("isCompound", False),
                    "lowerNumericLimit": item.get("lowerNumericLimit"),
                    "upperNumericLimit": item.get("upperNumericLimit"),
                    "fields": [f.get("code") for f in item.get("fields", [])],
                    "options": [
                        {"code": o.get("code"), "text": o.get("texts", [{}])[0].get("text", "")}
                        for o in item.get("options", [])
                    ],
                    "titles": [t.get("text") for t in item.get("titles", [])]
                }

    return {
        "variable": clean_var,
        "found": False,
        "message": f"Variable '{clean_var}' not found in survey schema definition."
    }


def query_data(
    records: List[Dict[str, Any]],
    target_variable: Optional[str] = None,
    group_by: Optional[str] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    metric: str = "count"
) -> Dict[str, Any]:
    """Aggregate over flattened respondent records with strict context truncation.
    
    Workhorse analytical tool for the investigator agent. Groups respondents by
    dimensions (e.g. language 'l', device 'hDeviceTypeMain', sample source 'bgSource')
    and calculates statistical metrics (missing rates, means, proportions).

    Args:
        records: List of flattened respondent dictionaries.
        target_variable: Variable to measure (e.g. 'Q7Sk', 'hidLOI', 'S5').
        group_by: Categorical variable to segment by (e.g. 'l', 'lastdevicetype', 'bgSource').
        filter_dict: Equality filters (e.g. {'status': 'complete'}, {'l': '1046'}).
        metric: Metric to compute ('count', 'missing_rate', 'mean', 'median', 'distribution').

    Returns:
        Dictionary with aggregated table, row counts, and truncation notice if > 20 rows.
    """
    # 1. Apply filtering
    filtered = list(records)
    if filter_dict:
        for fk, fv in filter_dict.items():
            filtered = [
                r for r in filtered 
                if str(r.get(fk, "")).lower() == str(fv).lower() or r.get(fk) == fv
            ]

    total_filtered_n = len(filtered)
    if total_filtered_n == 0:
        return {
            "total_records": 0,
            "group_by": group_by,
            "target_variable": target_variable,
            "metric": metric,
            "rows": [],
            "message": "No respondent records matched the specified filter criteria."
        }

    # 2. Partition into groups
    groups: Dict[str, List[Dict[str, Any]]] = {}
    if group_by:
        for r in filtered:
            g_val = str(r.get(group_by, "MISSING_OR_NULL"))
            groups.setdefault(g_val, []).append(r)
    else:
        groups["ALL"] = filtered

    # 3. Compute metric per group
    rows = []
    for g_key, g_records in groups.items():
        g_size = len(g_records)
        row_entry: Dict[str, Any] = {
            "group": g_key,
            "n": g_size
        }

        # Metric: Missing rate / Item non-response
        if metric in ("missing_rate", "non_response") and target_variable:
            missing_count = 0
            for r in g_records:
                val = r.get(target_variable)
                if val is None or val == "" or val == [] or val == {}:
                    missing_count += 1
            missing_rate = missing_count / g_size if g_size > 0 else 0.0
            row_entry["missing_count"] = missing_count
            row_entry["missing_rate"] = round(missing_rate, 4)

        # Metric: Numeric mean
        elif metric == "mean" and target_variable:
            vals = []
            for r in g_records:
                v = r.get(f"{target_variable}__val", r.get(target_variable))
                if isinstance(v, (int, float)):
                    vals.append(v)
            row_entry["mean"] = round(statistics.mean(vals), 2) if vals else None
            row_entry["valid_n"] = len(vals)

        # Metric: Numeric median (e.g. for hidLOI speeding check)
        elif metric == "median" and target_variable:
            vals = []
            for r in g_records:
                v = r.get(f"{target_variable}__val", r.get(target_variable))
                if isinstance(v, (int, float)):
                    vals.append(v)
            row_entry["median"] = round(statistics.median(vals), 2) if vals else None
            row_entry["valid_n"] = len(vals)

        # Metric: Frequency distribution
        elif metric == "distribution" and target_variable:
            dist_counts: Dict[str, int] = {}
            for r in g_records:
                v = r.get(target_variable)
                if v is not None:
                    code = str(v)
                    dist_counts[code] = dist_counts.get(code, 0) + 1
            row_entry["distribution"] = dist_counts

        # Metric: Default simple count
        else:
            row_entry["count"] = g_size

        rows.append(row_entry)

    # Sort rows by group size descending
    rows.sort(key=lambda x: x.get("n", 0), reverse=True)

    # 4. Context Truncation Guardrail: Cap at MAX_QUERY_ROWS (20)
    total_rows = len(rows)
    omitted_rows = 0
    truncation_notice = None

    if total_rows > MAX_QUERY_ROWS:
        omitted_rows = total_rows - MAX_QUERY_ROWS
        rows = rows[:MAX_QUERY_ROWS]
        truncation_notice = (
            f"[Truncated: showing top {MAX_QUERY_ROWS} rows. "
            f"{omitted_rows} additional rows omitted to preserve context window.]"
        )

    result: Dict[str, Any] = {
        "total_records": total_filtered_n,
        "group_by": group_by,
        "target_variable": target_variable,
        "metric": metric,
        "total_groups": total_rows,
        "rows": rows
    }
    if truncation_notice:
        result["truncation_notice"] = truncation_notice

    return result


def sample_respondents(
    records: List[Dict[str, Any]],
    filter_dict: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    n: int = 3
) -> Dict[str, Any]:
    """Pull a handful of full respondent records to eyeball concrete data patterns.
    
    Strictly capped at 5 records to prevent context explosion.

    Args:
        records: List of flattened respondent records.
        filter_dict: Filter conditions (e.g. {'l': '1046', 'lastdevicetype': 1}).
        fields: Optional subset of field names to extract (defaults to key diagnostic fields).
        n: Number of sample records to fetch (maximum 5).

    Returns:
        Dictionary with count of sampled records and list of record summaries.
    """
    n_clamped = max(1, min(5, n))
    filtered = list(records)

    if filter_dict:
        for fk, fv in filter_dict.items():
            filtered = [
                r for r in filtered
                if str(r.get(fk, "")).lower() == str(fv).lower() or r.get(fk) == fv
            ]

    sampled_raw = filtered[:n_clamped]
    diagnostic_fields = fields or [
        "respid", "status", "l", "lastdevicetype", "hDeviceTypeMain", 
        "bgSource", "hidLOI", "S2", "S5", "Q5", "Q7Mx", "Q7Sk", "T14"
    ]

    sampled_records = []
    for r in sampled_raw:
        item: Dict[str, Any] = {}
        for f in diagnostic_fields:
            if f in r:
                item[f] = r[f]
        sampled_records.append(item)

    return {
        "sampled_count": len(sampled_records),
        "total_matching": len(filtered),
        "records": sampled_records
    }


def walk_survey(
    language: str = "9",
    device: str = "desktop",
    target_question: Optional[str] = None,
    mock_inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Step through the in-memory survey state machine as a simulated respondent.
    
    The closed loop: Enables the agent to verify whether an observed oddity
    in the response data is caused by a live questionnaire script, rendering,
    or validation defect.

    Args:
        language: Language identifier (e.g. '1046', '2058', '9', '18', '31').
        device: Device type ('mobile', 'desktop', '1', '3').
        target_question: Question to inspect (e.g. 'Q7Sk', 'S5', 'Q9', 'Q7Mx').
        mock_inputs: Optional test values to submit during the walk.

    Returns:
        Structured simulation findings with rendered controls, validation state,
        and observed bug flags.
    """
    return execute_walk_survey(
        language=language,
        device=device,
        target_question=target_question,
        mock_inputs=mock_inputs
    )


def conclude(
    cause: str,
    confidence: str,
    evidence: List[str],
    severity: str,
    affected_markets: Optional[List[str]] = None,
    reproduction_steps: Optional[str] = None
) -> Dict[str, Any]:
    """Finish the investigation and hand back a structured finding.
    
    Enforces a strict structured conclusion schema instead of free-form prose.

    Args:
        cause: Identified root cause of the anomaly.
        confidence: Confidence level ('high', 'medium', 'low').
        evidence: List of evidentiary points established during investigation.
        severity: Assessed severity ('critical', 'high', 'medium', 'low').
        affected_markets: List of affected market language codes.
        reproduction_steps: Steps to reproduce or verify in survey model.

    Returns:
        Standardized conclusion dictionary.
    """
    conf = confidence.lower() if confidence else "medium"
    if conf not in ("high", "medium", "low"):
        conf = "medium"

    sev = severity.lower() if severity else "medium"
    if sev not in ("critical", "high", "medium", "low"):
        sev = "medium"

    return {
        "is_concluded": True,
        "cause": cause.strip(),
        "confidence": conf,
        "evidence": evidence or [],
        "severity": sev,
        "affected_markets": affected_markets or [],
        "reproduction_steps": reproduction_steps or ""
    }


# =============================================================================
# Investigation Guardrails Engine
# =============================================================================

class InvestigationGuardrails:
    """Manages execution bounds, call ceilings, and loop prevention for an investigation."""

    def __init__(
        self,
        max_calls: int = MAX_TOOL_CALLS,
        loop_limit: int = LOOP_DETECTION_LIMIT
    ):
        self.max_calls = max_calls
        self.loop_limit = loop_limit
        self.call_count = 0
        self.call_history: List[Tuple[str, str]] = []  # List of (tool_name, serialized_args)
        self.is_aborted = False
        self.abort_reason: Optional[str] = None

    def record_and_validate(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate whether a tool call is permitted by guardrails.
        
        Args:
            tool_name: Name of tool to execute.
            tool_args: Tool argument dictionary.

        Returns:
            Tuple of (is_allowed, error_message_if_blocked).
        """
        # 1. Hard call count ceiling (25 tool calls)
        self.call_count += 1
        if self.call_count > self.max_calls:
            self.is_aborted = True
            self.abort_reason = (
                f"Hard cap of {self.max_calls} tool calls reached. "
                "Investigation halted to prevent runaway execution."
            )
            return False, self.abort_reason

        # 2. Loop detection (identical tool + identical args >= 3 times)
        args_repr = json.dumps(tool_args, sort_keys=True)
        call_signature = (tool_name, args_repr)

        same_call_occurrences = sum(1 for prev in self.call_history if prev == call_signature)
        if same_call_occurrences >= (self.loop_limit - 1):
            self.is_aborted = True
            self.abort_reason = (
                f"Loop detected: tool '{tool_name}' invoked {self.loop_limit} times "
                "with identical arguments without gaining new information."
            )
            return False, self.abort_reason

        self.call_history.append(call_signature)
        return True, None


# =============================================================================
# Tool Definitions for LLM Function Calling Schema
# =============================================================================

INVESTIGATOR_TOOL_DEFINITIONS = [
    {
        "name": "query_data",
        "description": "Aggregate and segment survey response data across dimensions (language, device, source). Automatically caps results at top 20 rows to prevent context window overflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_variable": {
                    "type": "string",
                    "description": "Variable name to measure (e.g. 'Q7Sk', 'hidLOI', 'S5', 'T14')."
                },
                "group_by": {
                    "type": "string",
                    "description": "Dimension to group by: 'l' (language/market), 'lastdevicetype' or 'hDeviceTypeMain' (device), 'bgSource' (sample vendor)."
                },
                "filter_dict": {
                    "type": "object",
                    "description": "Optional equality filters (e.g. {'status': 'complete'}, {'l': '1046'})."
                },
                "metric": {
                    "type": "string",
                    "enum": ["count", "missing_rate", "mean", "median", "distribution"],
                    "description": "Metric to compute: 'missing_rate' for item non-response, 'median' for LOI, 'count' for frequency."
                }
            },
            "required": ["target_variable"]
        }
    },
    {
        "name": "get_schema",
        "description": "Look up question metadata, type, declared numeric limits, choice options, and sub-items from Confirmit schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": "Question variable name to inspect (e.g. 'Q7Sk', 'S5', 'T14', 'Q9')."
                }
            },
            "required": ["variable"]
        }
    },
    {
        "name": "sample_respondents",
        "description": "Pull up to 5 full respondent records matching specific criteria to eyeball concrete responses.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_dict": {
                    "type": "object",
                    "description": "Filter criteria (e.g. {'l': '1046', 'lastdevicetype': 1})."
                },
                "n": {
                    "type": "integer",
                    "description": "Number of sample records to fetch (1 to 5, default 3)."
                }
            }
        }
    },
    {
        "name": "walk_survey",
        "description": "The closed loop: step through the questionnaire state machine as a respondent on a specific device and language to test if survey code, rendering, or skip logic is defective.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Language code (e.g. '1046' for Portuguese, '2058' for Spanish, '9' for English)."
                },
                "device": {
                    "type": "string",
                    "enum": ["mobile", "desktop"],
                    "description": "Device environment ('mobile' or 'desktop')."
                },
                "target_question": {
                    "type": "string",
                    "description": "Question variable to reach and test (e.g. 'Q7Sk', 'S5', 'Q9')."
                },
                "mock_inputs": {
                    "type": "object",
                    "description": "Optional simulated inputs to submit (e.g. {'S5': 115.0}, {'S2': '1'})."
                }
            },
            "required": ["language", "device", "target_question"]
        }
    },
    {
        "name": "conclude",
        "description": "Finish the investigation and submit structured root-cause findings. This is the only valid way to complete an investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "cause": {
                    "type": "string",
                    "description": "Concise summary of the verified root cause."
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level in findings."
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of evidentiary proof points gathered."
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                    "description": "Assessed business and data severity."
                },
                "affected_markets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of affected language/market codes."
                },
                "reproduction_steps": {
                    "type": "string",
                    "description": "Concrete steps required to reproduce the issue in the survey."
                }
            },
            "required": ["cause", "confidence", "evidence", "severity"]
        }
    }
]

