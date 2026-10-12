"""Test suite for Step 2: Synthetic Data Generator and Flattener."""
import sys
import json
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RESPONSES_N45_PATH,
    RESPONSES_N150_PATH,
    GROUND_TRUTH_N45_PATH,
    GROUND_TRUTH_N150_PATH
)
from src.flattener import flatten_dataset, flatten_respondent


def test_generated_files_exist_and_counts():
    """Verify that both n=45 and n=150 response and ground truth files exist and have exact record counts."""
    # Confirm file existence on disk
    assert RESPONSES_N45_PATH.exists(), f"Missing {RESPONSES_N45_PATH}"
    assert RESPONSES_N150_PATH.exists(), f"Missing {RESPONSES_N150_PATH}"
    assert GROUND_TRUTH_N45_PATH.exists(), f"Missing {GROUND_TRUTH_N45_PATH}"
    assert GROUND_TRUTH_N150_PATH.exists(), f"Missing {GROUND_TRUTH_N150_PATH}"

    # Verify underpowered sample size: exactly 45 respondents
    with open(RESPONSES_N45_PATH) as f:
        data_45 = json.load(f)
    assert len(data_45["items"]) == 45, f"Expected 45 items, got {len(data_45['items'])}"

    # Verify main sample size: exactly 150 respondents
    with open(RESPONSES_N150_PATH) as f:
        data_150 = json.load(f)
    assert len(data_150["items"]) == 150, f"Expected 150 items, got {len(data_150['items'])}"


def test_ground_truth_fault_integrity():
    """Validate that ground truth manifests track all 8 planted faults with exact affected respondent lists."""
    with open(GROUND_TRUTH_N150_PATH) as f:
        gt_150 = json.load(f)

    # Ensure all 8 faults are declared
    assert len(gt_150) == 8, f"Expected 8 planted faults in ground truth, got {len(gt_150)}"
    fault_ids = [f["fault_id"] for f in gt_150]
    assert fault_ids == ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"]

    # Ensure consistent ground truth schema per fault
    for entry in gt_150:
        assert "affected_count" in entry, f"Missing affected_count in {entry['fault_id']}"
        assert "affected_respondents" in entry, f"Missing affected_respondents in {entry['fault_id']}"
        assert len(entry["affected_respondents"]) == entry["affected_count"], (
            f"Count mismatch in {entry['fault_id']}: {len(entry['affected_respondents'])} vs {entry['affected_count']}"
        )


def test_flattener_behavior():
    """Verify that flatten_dataset properly flattens nested Confirmit loops, multi-choice, and grids."""
    with open(RESPONSES_N150_PATH) as f:
        items = json.load(f)["items"]

    flat = flatten_dataset(items)
    assert len(flat) == 150

    sample = flat[0]
    # Check core scalar fields
    assert "respid" in sample, "Flattened record missing respid"
    assert "status" in sample, "Flattened record missing status"
    assert "l" in sample, "Flattened record missing language 'l'"
    assert "lastdevicetype" in sample, "Flattened record missing lastdevicetype"
    assert "hidLOI" in sample, "Flattened record missing hidLOI duration"

    # Check loop block flattening (e.g. LoopT1 and LpPSPrices counts)
    assert "LoopT1__count" in sample, "LoopT1 iteration count missing"
    assert "LpPSPrices__count" in sample, "LpPSPrices iteration count missing"

    # Check multi-choice flattening into list and count metrics
    assert "ErrorMessage" in sample
    assert isinstance(sample["ErrorMessage"], list), "ErrorMessage not flattened to list of codes"
    assert "ErrorMessage__selected_count" in sample

    # Check compound numeric flattening with automatic sum calculation
    if sample.get("Q7Mx") is not None:
        assert "Q7Mx__sum" in sample, "Q7Mx numeric allocation sum missing"
        assert "Q7Mx_1" in sample, "Q7Mx part 1 missing"


def test_planted_faults_presence():
    """Verify that the generated n=150 dataset exhibits the intended symptoms for all planted faults."""
    with open(RESPONSES_N150_PATH) as f:
        items = json.load(f)["items"]

    flat = flatten_dataset(items)

    # Fault F01: Q7Sk missing on mobile in languages 1046 and 2058
    f01_missings = [
        r for r in flat 
        if r["l"] in ("1046", "2058") and str(r["lastdevicetype"]) == "1" and r.get("Q7Sk") is None
    ]
    assert len(f01_missings) > 0, "F01 missing Q7Sk not observed"

    # Fault F02: T14 grid straightlining (all 7 items have identical value)
    straightliners = [
        r for r in flat
        if all(r.get(f"T14_{i}") == r.get("T14_1") for i in range(1, 8)) and r.get("T14_1") is not None
    ]
    assert len(straightliners) >= 2, f"F02 straightliners insufficient: {len(straightliners)}"

    # Fault F04: S5 accepts out-of-range ratings > 100.0
    range_violators = [r for r in flat if (r.get("S5__val") or 0) > 100.0]
    assert len(range_violators) >= 2, f"F04 range violations not found: {len(range_violators)}"

    # Fault F05: Q7Mx parts do not sum to 100
    allocation_violators = [
        r for r in flat 
        if r.get("Q7Mx__sum") is not None and abs(r["Q7Mx__sum"] - 100.0) > 0.001
    ]
    assert len(allocation_violators) >= 2, f"F05 allocation sum violations not found: {len(allocation_violators)}"

    # Fault F07: Q5 option 1 selected by disproportionate majority (>75%)
    q5_val1_count = sum(1 for r in flat if str(r.get("Q5")) == "1")
    q5_total = sum(1 for r in flat if r.get("Q5") is not None)
    ratio = q5_val1_count / q5_total if q5_total else 0
    assert ratio > 0.75, f"F07 degenerate distribution ratio too low: {ratio}"


if __name__ == "__main__":
    test_generated_files_exist_and_counts()
    test_ground_truth_fault_integrity()
    test_flattener_behavior()
    test_planted_faults_presence()
    print("All Step 2 data generator and flattener tests passed successfully!")
