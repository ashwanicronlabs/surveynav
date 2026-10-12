"""Test suite for Step 4: Survey Model State Machine (Closed-Loop Simulator)."""
import sys
from pathlib import Path

# Adjust path for running directly from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.survey_model import walk_survey, SurveyStateMachine


def test_f01_closed_loop_slider_failure():
    """Verify flagship Fault F01: Q7Sk slider fails to render only on mobile in languages 1046 and 2058."""
    # Test Language 1046 on Mobile -> Must trigger F01 rendering bug
    res_1046_mobile = walk_survey(language="1046", device="mobile", target_question="Q7Sk")
    target_state = res_1046_mobile["target_state"]
    assert "NONE" in target_state["rendered_controls"], "Expected slider controls to be missing on mobile 1046"
    assert "BYPASSED" in target_state["validation_status"]
    assert any("F01" in b for b in res_1046_mobile["bugs_observed"])

    # Test Language 2058 on Mobile -> Must trigger F01 rendering bug
    res_2058_mobile = walk_survey(language="2058", device="mobile", target_question="Q7Sk")
    assert any("F01" in b for b in res_2058_mobile["bugs_observed"])

    # Test Language 1046 on Desktop -> Must render normally (closed loop contrast)
    res_1046_desktop = walk_survey(language="1046", device="desktop", target_question="Q7Sk")
    assert "Active HTML5 Slider" in res_1046_desktop["target_state"]["rendered_controls"]
    assert not any("F01" in b for b in res_1046_desktop["bugs_observed"])

    # Test Language 9 (English) on Mobile -> Must render normally (localized issue only)
    res_en_mobile = walk_survey(language="9", device="mobile", target_question="Q7Sk")
    assert "Active HTML5 Slider" in res_en_mobile["target_state"]["rendered_controls"]
    assert not any("F01" in b for b in res_en_mobile["bugs_observed"])


def test_f04_range_violation_walk():
    """Verify Fault F04: S5 accepts values > 100 without raising validation errors."""
    # Out of range input 115.0
    res_invalid = walk_survey(target_question="S5", mock_inputs={"S5": 115.0})
    assert "DEFECTIVE" in res_invalid["target_state"]["validation_status"]
    assert any("F04" in b for b in res_invalid["bugs_observed"])

    # Valid input 75.0
    res_valid = walk_survey(target_question="S5", mock_inputs={"S5": 75.0})
    assert res_valid["target_state"]["validation_status"] == "OK (Normal range)"
    assert not any("F04" in b for b in res_valid["bugs_observed"])


def test_f05_allocation_sum_walk():
    """Verify Fault F05: Q7Mx allows proceeding when parts total != 100."""
    # Defective sum 85.0
    res_bad_sum = walk_survey(target_question="Q7Mx", mock_inputs={"Q7Mx": 85.0})
    assert "DISABLED" in res_bad_sum["target_state"]["validation_status"]
    assert any("F05" in b for b in res_bad_sum["bugs_observed"])

    # Valid sum 100.0
    res_good_sum = walk_survey(target_question="Q7Mx", mock_inputs={"Q7Mx": 100.0})
    assert "OK" in res_good_sum["target_state"]["validation_status"]
    assert not any("F05" in b for b in res_good_sum["bugs_observed"])


def test_f06_routing_skip_walk():
    """Verify Fault F06: Routing bug forces S2=1 respondents into Q9 instead of skipping."""
    # S2 = 1 (should skip Q9, but survey mistakenly routes them in)
    res_buggy_route = walk_survey(target_question="Q9", mock_inputs={"S2": "1"})
    assert "ROUTING FAILURE" in res_buggy_route["target_state"]["validation_status"]
    assert any("F06" in b for b in res_buggy_route["bugs_observed"])

    # S2 = 2 (legitimately qualifies for Q9)
    res_valid_route = walk_survey(target_question="Q9", mock_inputs={"S2": "2"})
    assert res_valid_route["target_state"]["validation_status"] == "OK"
    assert not any("F06" in b for b in res_valid_route["bugs_observed"])


if __name__ == "__main__":
    test_f01_closed_loop_slider_failure()
    test_f04_range_violation_walk()
    test_f05_allocation_sum_walk()
    test_f06_routing_skip_walk()
    print("All Step 4 survey model tests passed successfully!")

