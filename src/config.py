"""Global configuration constants and paths for SurveyNav REQ-009."""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Core Data Files
SCHEMA_PATH = DATA_DIR / "synthetic_survey_responses_schema.json"
FAULTS_PATH = DATA_DIR / "faults.json"
SAMPLE_RESPONSES_PATH = DATA_DIR / "sample_responses.json"

RESPONSES_N45_PATH = DATA_DIR / "responses_n45.json"
RESPONSES_N150_PATH = DATA_DIR / "responses_n150.json"
GROUND_TRUTH_N45_PATH = DATA_DIR / "ground_truth_n45.json"
GROUND_TRUTH_N150_PATH = DATA_DIR / "ground_truth_n150.json"

# Statistical Power Gate Minimum Sample Sizes (Valid From)
POWER_GATE_THRESHOLDS = {
    "range_violation": 1,
    "allocation_sum": 1,
    "straightlining": 1,
    "speeding": 30,
    "item_non_response": 30,
    "disposition_mix": 50,
    "routing_anomaly": 50,
    "quota_pace": 100,
    "degenerate_distribution": 200
}

# Blocked Categories (Honest reporting of missing client specifications)
BLOCKED_CATEGORIES = {
    "routing_conformance": {
        "reason": "Requires questionnaire Word specification for skip conditions, not present in raw Confirmit schema.",
        "status": "BLOCKED"
    },
    "quota_conformance": {
        "reason": "Requires target quota plan and market targets not provided in dataset.",
        "status": "BLOCKED"
    }
}

# Investigator Guardrails
MAX_TOOL_CALLS = 25
MAX_QUERY_ROWS = 20
LOOP_DETECTION_LIMIT = 3
INVESTIGATION_TIMEOUT = 60

# Confirmit Language Map
LANGUAGE_MAP = {
    "1": "English (US/UK)",
    "9": "English",
    "18": "German",
    "31": "French",
    "1046": "Portuguese (Brazil)",
    "2058": "Spanish (Mexico)"
}

# Device Map
DEVICE_MAP = {
    "1": "Mobile",
    "2": "Tablet",
    "3": "Desktop",
    "4": "Laptop",
    "5": "Other"
}

