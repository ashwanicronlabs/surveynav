import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surveynav.src.config import SCHEMA_PATH, FAULTS_PATH, SAMPLE_RESPONSES_PATH, POWER_GATE_THRESHOLDS

def test_schema_exists_and_valid():
    assert SCHEMA_PATH.exists(), f"Schema file missing at {SCHEMA_PATH}"
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    assert "root" in schema
    assert "variables" in schema["root"]
    assert "children" in schema["root"]
    assert len(schema["languages"]) == 5
    
    # Check variable count (195 variables/keys total as specified)
    root_vars = len(schema["root"]["variables"])
    t1_vars = len(schema["root"]["children"][0]["variables"])
    price_vars = len(schema["root"]["children"][1]["variables"])
    total_vars = root_vars + t1_vars + price_vars + 3 # 3 keys
    assert total_vars == 195, f"Expected 195 variables, found {total_vars}"

def test_faults_catalogue_valid():
    assert FAULTS_PATH.exists(), f"Faults file missing at {FAULTS_PATH}"
    with open(FAULTS_PATH) as f:
        faults = json.load(f)
    assert len(faults) == 8, f"Expected 8 planted faults, found {len(faults)}"
    
    fault_ids = [f["fault_id"] for f in faults]
    assert fault_ids == ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"]
    
    # Check F01 has in_survey = True and min_n = 30
    f01 = faults[0]
    assert f01["in_survey"] is True
    assert f01["min_n"] == 30
    assert f01["target_variable"] == "Q7Sk"

def test_sample_responses_valid():
    assert SAMPLE_RESPONSES_PATH.exists()
    with open(SAMPLE_RESPONSES_PATH) as f:
        resps = json.load(f)
    assert len(resps["items"]) == 6

def test_power_gate_thresholds():
    assert POWER_GATE_THRESHOLDS["range_violation"] == 1
    assert POWER_GATE_THRESHOLDS["speeding"] == 30
    assert POWER_GATE_THRESHOLDS["item_non_response"] == 30
    assert POWER_GATE_THRESHOLDS["disposition_mix"] == 50
    assert POWER_GATE_THRESHOLDS["degenerate_distribution"] == 200

if __name__ == "__main__":
    test_schema_exists_and_valid()
    test_faults_catalogue_valid()
    test_sample_responses_valid()
    test_power_gate_thresholds()
    print("All Step 1 scaffold tests passed successfully!")
