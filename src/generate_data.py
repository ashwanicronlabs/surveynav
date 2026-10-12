"""Synthetic survey respondent generator with fault injection.

Generates realistic Confirmit survey datasets adhering to the client schema,
and injects the 8 planted faults declared in data/faults.json with ground truth logging.
"""
import sys
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Support running from root or surveynav subdir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    SCHEMA_PATH,
    FAULTS_PATH,
    DATA_DIR,
    RESPONSES_N45_PATH,
    RESPONSES_N150_PATH,
    GROUND_TRUTH_N45_PATH,
    GROUND_TRUTH_N150_PATH
)

LANGUAGES = ["9", "18", "31", "1046", "2058"]
DEVICES = ["1", "2", "3", "4", "5"]  # 1: Mobile, 2: Tablet, 3: Desktop, etc.
SOURCES = ["1", "2", "3", "4"]


def generate_baseline_respondent(respid: int, rng: random.Random) -> Dict[str, Any]:
    """Generate a valid, coherent baseline respondent matching the Confirmit schema.
    
    This function generates a single clean respondent before any intentional faults
    are injected. It simulates standard market research respondent profiles,
    including realistic distributions for completion statuses, survey duration (LOI),
    demographics/screeners, grid evaluations, allocation sliders, and loop iterations.

    Args:
        respid: Unique respondent identification number.
        rng: Seeded random number generator instance for reproducible generation.

    Returns:
        A dictionary containing all top-level variables, grids, multi-choice sets,
        and nested loop blocks matching the structure of Confirmit data exports.
    """
    lang = rng.choice(LANGUAGES)
    device = rng.choice(DEVICES)
    source = rng.choice(SOURCES)
    
    # 1. Respondent Disposition Mix:
    # Industry standard soft launch distribution: ~70% completes, ~15% screen-outs,
    # ~8% incompletes/dropouts, ~5% quota-full, ~2% script/server errors.
    status_roll = rng.random()
    if status_roll < 0.70:
        status = "complete"
    elif status_roll < 0.85:
        status = "screened"
    elif status_roll < 0.93:
        status = "incomplete"
    elif status_roll < 0.98:
        status = "quotafull"
    else:
        status = "error"

    # 2. Length of Interview (LOI):
    # Standard survey duration modeled as a normal distribution around 15 mins (std dev 3 mins).
    # Clamped between 7.0 and 35.0 mins to avoid negative or infinite interview lengths.
    loi_min = rng.gauss(15.0, 3.0)
    loi_min = max(7.0, min(35.0, loi_min))

    # 3. System and Paradata Variables
    record: Dict[str, Any] = {
        "respid": respid,
        "responseid": respid,
        "status": status,
        "l": lang,
        "lastdevicetype": int(device),
        "hDeviceTypeMain": device,
        "bgSource": source,
        "hidsource": source,
        "lastchannel": rng.choice([1, 2, 4, 8]),
        "lastrenderingmode": rng.choice([1, 2, 8]),
        "hidLOI": round(loi_min, 2),
        "hidLOISec": round(loi_min * 60, 1),
        "hIPFlag": rng.choice(["1", "2"]),
        "hPunch": rng.choice(["1", "2", "3", "4"]),
        "ErrorMessage": {"true": [rng.choice(["1", "2", "3"])], "false": []},
        "hidQueDropout": "2025-01-05T08:30:00Z",
        "interview_start": "2025-01-05T08:00:00Z",
        "interview_end": "2025-01-05T08:20:00Z",
        "last_touched": "2025-01-05T08:18:00Z",
        "lastcomplete": "2025-01-05T08:20:00Z" if status == "complete" else None,
        "first_question_on_last_page_displayed": "Thank you for participating."
    }

    # 4. Screener Section (S0 to S14):
    # Standard single-choice, multi-choice, and numeric range answers.
    record["S0"] = rng.choice(["1", "2", "3", "4"])
    record["S0a"] = rng.choice(["1", "2"])
    record["S0b"] = rng.choice(["1", "2"])
    record["S0c"] = rng.choice(["1", "2"])
    record["S0d"] = {"true": [rng.choice(["1", "2"])]}
    record["S1"] = rng.choice(["1", "2"])
    record["S2"] = rng.choice(["1", "2"])
    record["S3"] = rng.choice(["1", "2"])
    record["S4"] = rng.choice(["1", "2", "3"])
    record["S5"] = {"1": round(rng.uniform(10.0, 95.0), 1)}  # Valid 0-100 rating
    record["S6"] = rng.choice(["1", "2", "3", "4", "5", "6", "7", "8"])
    record["S7"] = {"1": round(rng.uniform(20.0, 90.0), 1)}
    record["S8"] = {"1": round(rng.uniform(15.0, 85.0), 1)}
    record["S9"] = {"1": round(rng.uniform(10.0, 90.0), 1)}
    record["S10"] = {str(i): round(rng.uniform(10.0, 90.0), 1) for i in [1, 2, 3, 4]}
    record["S11"] = rng.choice(["1", "2"])
    record["S12"] = rng.choice(["1", "2"])
    record["S13"] = {"1": round(rng.uniform(10.0, 90.0), 1)}
    record["S14"] = rng.choice(["1", "2"])

    # 5. Core Questionnaire (Q1 to Q21):
    record["Q1"] = {"1": round(rng.uniform(10.0, 90.0), 1)}
    record["Q2"] = {"1": round(rng.uniform(10.0, 90.0), 1)}
    record["Q3"] = rng.choice(["1", "2", "3", "4", "5", "6", "7"])
    record["Q4"] = {str(i): round(rng.uniform(10.0, 90.0), 1) for i in [1, 2, 3, 4]}
    # Q5 options: uniformly distributed across options 1, 2, and 96 by default
    record["Q5"] = rng.choice(["1", "2", "96"])
    record["Q6"] = {"1": round(rng.uniform(10.0, 90.0), 1)}

    # Q7Mx: Allocation question with 3 parts strictly totaling 100 points
    p1 = rng.randint(20, 50)
    p2 = rng.randint(20, 50)
    p3 = 100 - (p1 + p2)
    record["Q7Mx"] = {"1": p1, "2": p2, "3": p3}

    # Q7Sk: Allocation question with 4 parts strictly totaling 100 points
    s1 = rng.randint(15, 35)
    s2 = rng.randint(15, 35)
    s3 = rng.randint(15, 35)
    s4 = 100 - (s1 + s2 + s3)
    record["Q7Sk"] = {"1": s1, "2": s2, "3": s3, "4": s4}

    record["Q7Tk"] = {str(i): rng.randint(10, 40) for i in range(1, 5)}
    record["Q7Dk"] = {"true": ["99"]}

    # Q9 Expected Skip Logic:
    # Only respondents with S2 == "2" should see Q9; S2 == "1" skips directly to Q11
    if record["S2"] == "2":
        record["Q9"] = rng.choice(["1", "2", "3", "4", "5"])
    else:
        record["Q9"] = None

    record["Q10"] = {"true": [rng.choice(["1", "2", "3", "4"])]}
    record["Q11"] = rng.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    record["Q12"] = {str(i): rng.randint(1, 5) for i in range(1, 8)}
    record["Q17"] = rng.choice(["1", "2", "3"])
    record["Q18"] = {"1": rng.choice(["1", "2", "3"])}
    record["Q19"] = {str(i): rng.choice(["1", "2", "3"]) for i in range(1, 4)}
    record["Q20"] = {str(i): rng.randint(1, 5) for i in range(1, 9)}
    record["Q21"] = {"1": rng.randint(1, 5)}

    # 6. Grid and Ranking Questions (T11 to T24):
    record["T11"] = rng.choice(["1", "2", "3"])
    record["T12"] = rng.choice(["1", "2", "3", "4"])
    record["T13"] = {str(i): str(rng.randint(1, 7)) for i in range(1, 9)}

    # T14: 7-item grid; realistic varied answers across items 1 through 7
    record["T14"] = {str(i): str(rng.randint(1, 2)) for i in range(1, 8)}

    record["T15"] = {str(i): rng.randint(1, 5) for i in range(1, 4)}
    record["T16"] = {"1": rng.randint(1, 5)}
    record["T17"] = rng.choice(["1", "2", "3", "4"])
    record["T18"] = {"1": rng.randint(1, 5)}
    record["T18Dk"] = {"false": ["1"]}
    record["T19"] = {"true": [rng.choice(["1", "2", "3"])], "false": ["4", "5"]}
    record["T20"] = {str(i): rng.randint(1, 5) for i in range(1, 4)}
    record["T21"] = {str(i): rng.randint(1, 5) for i in range(1, 4)}
    record["T23"] = {"1": rng.randint(1, 5)}
    record["T24"] = rng.choice(["1", "2", "3", "4"])
    record["T24New"] = {str(i): str(rng.randint(1, 5)) for i in range(1, 9)}

    # 7. Pricing Module (P1 to P12):
    record["P1"] = {str(i): round(rng.uniform(1.0, 5.0), 2) for i in range(1, 5)}
    record["P2"] = {"1": round(rng.uniform(1.0, 5.0), 2)}
    record["P3"] = rng.choice(["1", "2"])
    record["PP1"] = rng.choice(["1", "2", "3", "4"])
    record["PP2"] = rng.choice(["1", "2", "3", "4"])
    record["PP3"] = rng.choice(["1", "2", "3", "4"])
    record["PP4"] = rng.choice(["1", "2", "3", "4"])
    record["PP5"] = rng.choice(["1", "2", "3", "4"])
    record["P9"] = {"1": round(rng.uniform(1.0, 5.0), 2)}
    record["P10"] = {"1": round(rng.uniform(1.0, 5.0), 2)}
    record["P11"] = {"1": round(rng.uniform(1.0, 5.0), 2)}
    record["P12"] = {str(i): "Valid feedback response." for i in range(1, 4)}

    # 8. Nested Loop Blocks (LoopT1 and LpPSPrices):
    record["LoopT1"] = [
        {
            "LoopT1": "1",
            "T1": {"1": "Clear feedback."},
            "T2": {"1": "Useful response."},
            "T3": {"1": rng.choice(["1", "2", "3"])},
            "T4": rng.choice(["1", "2", "3", "4"]),
            "T5": {"1": rng.randint(1, 5)},
            "T6": {"true": ["1", "2"], "false": ["3"]},
            "T7": {str(i): str(rng.randint(1, 7)) for i in range(1, 8)},
            "T7b": rng.choice(["1", "2"]),
            "T9": rng.choice(["1", "2"]),
            "T10": {"1": "Good experience."}
        }
    ]

    record["LpPSPrices"] = [
        {
            "LpPSPrices": "1",
            "dLpPSItr": "1",
            "P6": rng.choice(["1", "2", "3", "4", "5"]),
            "P7": {"1": rng.randint(1, 5)},
            "P8": {"true": ["1", "2"], "false": ["3"]},
            "dPSResponseType": "1"
        }
    ]

    return record


def inject_faults(
    records: List[Dict[str, Any]],
    rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Inject the 8 planted faults declared in faults.json and record ground truth.
    
    This function applies the specific data anomalies representing each of the 8 faults:
    - F01: Allocation slider fails to render on mobile for languages 1046 and 2058 (Blank Q7Sk).
    - F02: Fraud/bot straightlining across all 7 items of grid T14.
    - F03: Speeding completions (<40% of median LOI) originating from bgSource=3.
    - F04: Numeric boundary violation accepting values > 100 on question S5.
    - F05: Allocation sum violation permitting parts to total != 100 on Q7Mx.
    - F06: Routing skip error sending S2=1 respondents into Q9.
    - F07: UI pre-selected default causing degenerate 88% selection on Q5.
    - F08: Screener over-rejection in market 1046 (>80% screened out).

    Args:
        records: Baseline respondent records to mutate.
        rng: Seeded random generator for repeatable fault assignments.

    Returns:
        A tuple of (mutated_records, ground_truth_manifest).
    """
    n = len(records)
    ground_truth: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Fault F01: Q7Sk slider fails on mobile in languages 1046 and 2058
    # Symptom: Leaves Q7Sk blank (item non-response cluster).
    # Minimum valid n: 30
    # -------------------------------------------------------------------------
    f01_affected = []
    for r in records:
        if r["l"] in ("1046", "2058") and str(r["lastdevicetype"]) == "1":
            r["Q7Sk"] = None
            f01_affected.append(r["respid"])
    ground_truth.append({
        "fault_id": "F01",
        "target_variable": "Q7Sk",
        "check_type": "item_non_response",
        "description": "Allocation slider fails on mobile in languages 1046 and 2058",
        "min_n": 30,
        "affected_count": len(f01_affected),
        "affected_respondents": f01_affected
    })

    # -------------------------------------------------------------------------
    # Fault F02: Straightlining on 7-item grid T14
    # Symptom: All 7 items given identical option value without reading.
    # Minimum valid n: 1
    # -------------------------------------------------------------------------
    straightline_count = max(2, int(n * 0.05))
    f02_candidates = [r for r in records if r["status"] == "complete"]
    f02_selected = rng.sample(f02_candidates, min(straightline_count, len(f02_candidates)))
    f02_affected = []
    for r in f02_selected:
        val = rng.choice(["1", "2"])
        r["T14"] = {str(i): val for i in range(1, 8)}
        f02_affected.append(r["respid"])
    ground_truth.append({
        "fault_id": "F02",
        "target_variable": "T14",
        "check_type": "straightlining",
        "description": "Straightlining on 7-item grid T14",
        "min_n": 1,
        "affected_count": len(f02_affected),
        "affected_respondents": f02_affected
    })

    # -------------------------------------------------------------------------
    # Fault F03: Speeding cluster from bgSource = "3"
    # Symptom: hidLOI < 30% of median for completes from sample vendor 3.
    # Minimum valid n: 30
    # -------------------------------------------------------------------------
    f03_affected = []
    for r in records:
        if r.get("bgSource") == "3" and r["status"] == "complete":
            r["hidLOI"] = round(rng.uniform(2.5, 4.2), 2)
            r["hidLOISec"] = round(r["hidLOI"] * 60, 1)
            f03_affected.append(r["respid"])
    ground_truth.append({
        "fault_id": "F03",
        "target_variable": "hidLOI",
        "check_type": "speeding",
        "description": "Speeding cluster from sample source bgSource=3",
        "min_n": 30,
        "affected_count": len(f03_affected),
        "affected_respondents": f03_affected
    })

    # -------------------------------------------------------------------------
    # Fault F04: S5 accepts values above declared upper limit (100.0)
    # Symptom: Range violation with numeric values like 105, 115, 125.
    # Minimum valid n: 1
    # -------------------------------------------------------------------------
    f04_candidates = [r for r in records if r["status"] == "complete"]
    f04_selected = rng.sample(f04_candidates, min(3, len(f04_candidates)))
    f04_affected = []
    for r in f04_selected:
        out_of_bound_val = rng.choice([105.0, 115.0, 125.0])
        r["S5"] = {"1": out_of_bound_val}
        f04_affected.append(r["respid"])
    ground_truth.append({
        "fault_id": "F04",
        "target_variable": "S5",
        "check_type": "range_violation",
        "description": "Question S5 accepts values > 100.0",
        "min_n": 1,
        "affected_count": len(f04_affected),
        "affected_respondents": f04_affected
    })

    # -------------------------------------------------------------------------
    # Fault F05: Allocation sum violation on Q7Mx
    # Symptom: Sum of parts does not total 100 (e.g. 80, 85, 110).
    # Minimum valid n: 1
    # -------------------------------------------------------------------------
    f05_candidates = [r for r in records if r["status"] == "complete" and r["respid"] not in f04_affected]
    f05_selected = rng.sample(f05_candidates, min(3, len(f05_candidates)))
    f05_affected = []
    for r in f05_selected:
        bad_sum = rng.choice([80, 85, 110, 120])
        r["Q7Mx"] = {"1": 40, "2": 25, "3": bad_sum - 65}
        f05_affected.append(r["respid"])
    ground_truth.append({
        "fault_id": "F05",
        "target_variable": "Q7Mx",
        "check_type": "allocation_sum",
        "description": "Allocation question Q7Mx does not sum to 100",
        "min_n": 1,
        "affected_count": len(f05_affected),
        "affected_respondents": f05_affected
    })

    # -------------------------------------------------------------------------
    # Fault F06: Routing bug sends S2=1 respondents into Q9
    # Symptom: Non-null responses on Q9 for respondents who should have skipped it.
    # Minimum valid n: 50
    # -------------------------------------------------------------------------
    f06_affected = []
    for r in records:
        if r.get("S2") == "1" and r["status"] == "complete":
            r["Q9"] = rng.choice(["1", "2", "3", "4", "5"])
            f06_affected.append(r["respid"])
    ground_truth.append({
        "fault_id": "F06",
        "target_variable": "Q9",
        "check_type": "routing_anomaly",
        "description": "Routing bug sends S2=1 into Q9 instead of skipping",
        "min_n": 50,
        "affected_count": len(f06_affected),
        "affected_respondents": f06_affected
    })

    # -------------------------------------------------------------------------
    # Fault F07: Pre-selected default on Q5 leads to degenerate distribution
    # Symptom: Option 1 accepted by ~88% of respondents due to pre-selected UI default.
    # Minimum valid n: 50
    # -------------------------------------------------------------------------
    f07_affected = []
    for r in records:
        if r["status"] in ("complete", "screened"):
            if rng.random() < 0.88:
                r["Q5"] = "1"
                f07_affected.append(r["respid"])
            else:
                r["Q5"] = rng.choice(["2", "96"])
    ground_truth.append({
        "fault_id": "F07",
        "target_variable": "Q5",
        "check_type": "degenerate_distribution",
        "description": "Pre-selected default option 1 on Q5 leads to degenerate distribution",
        "min_n": 50,
        "affected_count": len(f07_affected),
        "affected_respondents": f07_affected
    })

    # -------------------------------------------------------------------------
    # Fault F08: Screener over-rejection in market language 1046
    # Symptom: Screen-out rate in market 1046 is abnormally high (>80% vs ~15% norm).
    # Minimum valid n: 50
    # -------------------------------------------------------------------------
    f08_affected = []
    for r in records:
        if r.get("l") == "1046":
            if rng.random() < 0.82:
                r["status"] = "screened"
                f08_affected.append(r["respid"])
            else:
                r["status"] = "complete"
    ground_truth.append({
        "fault_id": "F08",
        "target_variable": "status",
        "check_type": "disposition_mix",
        "description": "Screener over-rejection in market language 1046 (>80% screened out)",
        "min_n": 50,
        "affected_count": len(f08_affected),
        "affected_respondents": f08_affected
    })

    return records, ground_truth


def generate_dataset(n: int, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate a deterministic dataset of size N with ground truth fault manifests.
    
    Args:
        n: Total number of respondents to generate (e.g. 45 or 150).
        seed: Random seed for exact reproducibility across runs.
        
    Returns:
        A tuple of (respondents_list, ground_truth_faults_list).
    """
    rng = random.Random(seed + n)
    base_respid = 200000

    raw_records = []
    for i in range(1, n + 1):
        raw_records.append(generate_baseline_respondent(base_respid + i, rng))

    return inject_faults(raw_records, rng)


def save_dataset(n: int, dest_dir: Path = DATA_DIR) -> Tuple[Path, Path]:
    """Generate and write response and ground truth JSON files to disk.
    
    Args:
        n: Total respondents count (e.g. 45 or 150).
        dest_dir: Destination folder path for output files.
        
    Returns:
        Tuple of (path_to_responses_json, path_to_ground_truth_json).
    """
    records, ground_truth = generate_dataset(n)
    
    resp_path = dest_dir / f"responses_n{n}.json"
    gt_path = dest_dir / f"ground_truth_n{n}.json"

    with open(resp_path, "w") as f:
        json.dump({"itemCount": len(records), "totalCount": len(records), "items": records}, f, indent=2)

    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return resp_path, gt_path


if __name__ == "__main__":
    # Command-line entry point: accepts optional sample sizes as arguments (e.g. python src/generate_data.py 150)
    target_sizes = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
    if not target_sizes:
        target_sizes = [45, 150]

    for size in target_sizes:
        rp, gp = save_dataset(size, DATA_DIR)
        print(f"Generated n={size} dataset at: {rp} and {gp}")
        
        # Also sync to surveynav/data if running from parent directory
        alt_data = Path("/home/cron/SurveyNav/surveynav/data")
        if alt_data.exists() and alt_data != DATA_DIR:
            save_dataset(size, alt_data)
