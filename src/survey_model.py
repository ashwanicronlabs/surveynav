"""Survey Model State Machine (Closed-Loop Simulator).

Provides an in-memory, deterministic ~50-100 line simulation of the survey runtime.
Enables the Investigator Agent to autonomously test hypotheses by simulating
a respondent walking the survey on specific devices and languages, closing the
loop without browser or network dependencies.
"""
from typing import Any, Dict, List, Optional, Tuple


class SurveyQuestionState:
    """Represents the rendered state and behavior of a survey question."""

    def __init__(
        self,
        question_id: str,
        question_type: str,
        rendered_controls: str,
        validation_status: str,
        experience_summary: str,
        bug_flag: Optional[str] = None
    ):
        self.question_id = question_id
        self.question_type = question_type
        self.rendered_controls = rendered_controls
        self.validation_status = validation_status
        self.experience_summary = experience_summary
        self.bug_flag = bug_flag

    def to_dict(self) -> Dict[str, Any]:
        """Convert question runtime state to dictionary."""
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "rendered_controls": self.rendered_controls,
            "validation_status": self.validation_status,
            "experience_summary": self.experience_summary,
            "bug_flag": self.bug_flag
        }


class SurveyStateMachine:
    """In-memory state machine standing in for the Confirmit questionnaire runtime.
    
    Covers ~10 key questions (S1, S2, S5, Q1, Q5, Q7Mx, Q7Sk, Q9, Q11, Q12) and
    encodes the survey-side behaviors of the planted faults:
    - F01: Q7Sk slider fails to render on mobile in languages 1046 and 2058.
    - F04: S5 accepts ratings > 100 without validation error.
    - F05: Q7Mx permits submission when parts total != 100.
    - F06: Inverted skip logic routes S2=1 respondents into Q9 instead of skipping.
    """

    def __init__(self, language: str = "9", device: str = "desktop"):
        """Initialize survey state machine with respondent environment.
        
        Args:
            language: Language code ('9', '18', '31', '1046', '2058').
            device: Device type ('mobile', 'desktop', '1', '3').
        """
        self.language = str(language).strip()
        self.device = "mobile" if str(device).lower() in ("mobile", "1") else "desktop"

    def render_question(
        self,
        question_id: str,
        mock_input: Optional[Any] = None,
        prior_answers: Optional[Dict[str, Any]] = None
    ) -> SurveyQuestionState:
        """Simulate rendering and respondent interaction for a single question.
        
        Args:
            question_id: Question variable identifier (e.g. 'Q7Sk', 'S5', 'Q9').
            mock_input: Optional simulated input value provided by walker.
            prior_answers: Dictionary of previous answers influencing routing.
            
        Returns:
            SurveyQuestionState describing controls, validation, and observed bugs.
        """
        p_ans = prior_answers or {}

        # ---------------------------------------------------------------------
        # Q7Sk: Constant-sum allocation slider (4 items)
        # Encodes Fault F01: Slider fails to render on mobile for 1046 and 2058
        # ---------------------------------------------------------------------
        if question_id == "Q7Sk":
            if self.device == "mobile" and self.language in ("1046", "2058"):
                return SurveyQuestionState(
                    question_id="Q7Sk",
                    question_type="allocation_slider",
                    rendered_controls="NONE (Widget initialization exception: missing localized touch slider bundle)",
                    validation_status="BYPASSED (Form permits clicking 'Next' with empty slider values)",
                    experience_summary=(
                        f"CRITICAL RENDERING BUG: In language {self.language} on {self.device}, "
                        "the slider control fails to render completely. Respondent sees only question text "
                        "and clicking 'Next' submits empty/blank response."
                    ),
                    bug_flag="F01: Allocation slider fails to render on mobile in languages 1046/2058"
                )
            else:
                return SurveyQuestionState(
                    question_id="Q7Sk",
                    question_type="allocation_slider",
                    rendered_controls="Active HTML5 Slider Controls (4 parts, 0-100 totalizer)",
                    validation_status="ACTIVE (Requires parts to sum to 100)",
                    experience_summary=f"Slider rendered normally on {self.device} in language {self.language}."
                )

        # ---------------------------------------------------------------------
        # S5: Numeric rating (0-100)
        # Encodes Fault F04: Missing upper limit validation allows values > 100
        # ---------------------------------------------------------------------
        elif question_id == "S5":
            entered_val = float(mock_input) if mock_input is not None else 50.0
            if entered_val > 100.0:
                return SurveyQuestionState(
                    question_id="S5",
                    question_type="numeric_input",
                    rendered_controls="Standard Numeric Text Input Box",
                    validation_status="DEFECTIVE (Missing upperNumericLimit validation rule)",
                    experience_summary=(
                        f"RANGE VALIDATION BUG: Entered value {entered_val} (> 100.0) is accepted "
                        "without triggering expected 'Value must be between 0 and 100' validation error."
                    ),
                    bug_flag="F04: Numeric question S5 accepts out-of-range values > 100"
                )
            else:
                return SurveyQuestionState(
                    question_id="S5",
                    question_type="numeric_input",
                    rendered_controls="Standard Numeric Text Input Box",
                    validation_status="OK (Normal range)",
                    experience_summary="Input accepted within normal 0-100 limits."
                )

        # ---------------------------------------------------------------------
        # Q7Mx: Allocation constant-sum (3 items)
        # Encodes Fault F05: Sum validation disabled, allows parts != 100
        # ---------------------------------------------------------------------
        elif question_id == "Q7Mx":
            entered_sum = mock_input if isinstance(mock_input, (int, float)) else 100.0
            if abs(entered_sum - 100.0) > 0.001:
                return SurveyQuestionState(
                    question_id="Q7Mx",
                    question_type="allocation_sum",
                    rendered_controls="3 Numeric Input Fields with Dynamic Totalizer",
                    validation_status="DISABLED (Client-side sum check disabled in script)",
                    experience_summary=(
                        f"ALLOCATION VALIDATION BUG: Inputs totaling {entered_sum} (!= 100) allowed to proceed. "
                        "Next button remains clickable despite totalizer showing non-100 sum."
                    ),
                    bug_flag="F05: Allocation question Q7Mx allows continuation when parts do not total 100"
                )
            else:
                return SurveyQuestionState(
                    question_id="Q7Mx",
                    question_type="allocation_sum",
                    rendered_controls="3 Numeric Input Fields with Dynamic Totalizer",
                    validation_status="OK (Sum equals 100)",
                    experience_summary="Parts totaled 100; page advanced normally."
                )

        # ---------------------------------------------------------------------
        # S2: Screener Qualification
        # Influences routing to Q9
        # ---------------------------------------------------------------------
        elif question_id == "S2":
            s2_choice = str(mock_input) if mock_input is not None else "2"
            return SurveyQuestionState(
                question_id="S2",
                question_type="single_choice",
                rendered_controls="Radio buttons (1: Primary, 2: Secondary)",
                validation_status="OK",
                experience_summary=f"Selected S2={s2_choice}. Expected routing: S2=1 skips Q9; S2=2 shows Q9."
            )

        # ---------------------------------------------------------------------
        # Q9: Specialty question
        # Encodes Fault F06: Routing bug routes S2=1 into Q9 instead of skipping
        # ---------------------------------------------------------------------
        elif question_id == "Q9":
            s2_val = str(p_ans.get("S2", "2"))
            if s2_val == "1":
                return SurveyQuestionState(
                    question_id="Q9",
                    question_type="single_choice",
                    rendered_controls="Radio buttons (1 to 5)",
                    validation_status="ROUTING FAILURE (Condition S2!=1 inverted in survey script)",
                    experience_summary=(
                        "ROUTING LOGIC BUG: Respondent with S2=1 was incorrectly routed into Q9. "
                        "Specification states S2=1 must skip directly to Q11."
                    ),
                    bug_flag="F06: Routing skip bug routes S2=1 respondents into Q9"
                )
            else:
                return SurveyQuestionState(
                    question_id="Q9",
                    question_type="single_choice",
                    rendered_controls="Radio buttons (1 to 5)",
                    validation_status="OK",
                    experience_summary="Q9 displayed as expected for S2=2."
                )

        # ---------------------------------------------------------------------
        # Other standard questions (S1, Q1, Q5, Q11, Q12)
        # ---------------------------------------------------------------------
        elif question_id == "S1":
            return SurveyQuestionState("S1", "single_choice", "Radio buttons (1, 2)", "OK", "S1 rendered normally.")
        elif question_id == "Q1":
            return SurveyQuestionState("Q1", "numeric_rating", "Numeric input box", "OK", "Q1 rendered normally.")
        elif question_id == "Q5":
            return SurveyQuestionState("Q5", "single_choice", "Radio buttons (1, 2, 96)", "OK", "Q5 rendered normally.")
        elif question_id == "Q11":
            return SurveyQuestionState("Q11", "single_choice", "Radio buttons (1 to 12)", "OK", "Q11 rendered normally.")
        elif question_id == "Q12":
            return SurveyQuestionState("Q12", "matrix_grid", "7-row rating matrix", "OK", "Q12 rendered normally.")
        elif question_id == "T14":
            return SurveyQuestionState("T14", "matrix_grid", "7-row grid (options 1, 2)", "OK", "T14 survey controls normal (fault is respondent fraud).")
        else:
            return SurveyQuestionState(
                question_id=question_id,
                question_type="generic_question",
                rendered_controls="Standard Web Controls",
                validation_status="OK",
                experience_summary=f"Question {question_id} rendered normally."
            )

    def walk(
        self,
        target_question: Optional[str] = None,
        mock_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Walk the survey flow and return complete simulated respondent experience.
        
        Args:
            target_question: Target question to test or reach (e.g. 'Q7Sk', 'S5', 'Q9').
            mock_inputs: Optional test inputs to supply at specific questions.
            
        Returns:
            Dictionary with session details, sequence of visited questions,
            target question evaluation, and list of bugs observed.
        """
        inputs = mock_inputs or {}
        sequence = ["S1", "S2", "S5", "Q1", "Q5", "Q7Mx", "Q7Sk", "Q9", "Q11", "Q12"]
        
        visited_log: List[Dict[str, Any]] = []
        bugs_found: List[str] = []
        target_state: Optional[Dict[str, Any]] = None

        prior_answers = dict(inputs)

        for qid in sequence:
            # Handle skip logic check for Q9
            if qid == "Q9":
                s2_ans = str(prior_answers.get("S2", "1"))
                # In broken survey, S2=1 hits Q9 anyway!
                state = self.render_question(qid, inputs.get(qid), prior_answers)
                if state.bug_flag:
                    bugs_found.append(state.bug_flag)
                visited_log.append(state.to_dict())
                if target_question == qid:
                    target_state = state.to_dict()
                    break
                continue

            q_input = inputs.get(qid)
            state = self.render_question(qid, q_input, prior_answers)
            
            if q_input is not None:
                prior_answers[qid] = q_input

            if state.bug_flag and state.bug_flag not in bugs_found:
                bugs_found.append(state.bug_flag)

            visited_log.append(state.to_dict())

            if target_question == qid:
                target_state = state.to_dict()
                break

        # If target question was not reached in standard flow, render it directly
        if target_question and not target_state:
            direct_state = self.render_question(target_question, inputs.get(target_question), prior_answers)
            target_state = direct_state.to_dict()
            if direct_state.bug_flag and direct_state.bug_flag not in bugs_found:
                bugs_found.append(direct_state.bug_flag)

        return {
            "session": {
                "language": self.language,
                "device": self.device,
                "target_question": target_question
            },
            "questions_visited": [item["question_id"] for item in visited_log],
            "target_state": target_state,
            "bugs_observed": bugs_found,
            "walk_log": visited_log
        }


def walk_survey(
    language: str = "9",
    device: str = "desktop",
    target_question: Optional[str] = None,
    mock_inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Top-level function invoked by the investigator agent's walk_survey tool.
    
    Steps through the survey state machine as a simulated respondent and
    returns the exact UI rendering, validation, and routing behavior experienced.

    Args:
        language: Language code (e.g. '1046', '2058', '9', '18', '31').
        device: Device type ('mobile', 'desktop', '1', '3').
        target_question: Target question variable to inspect (e.g. 'Q7Sk', 'S5', 'Q9').
        mock_inputs: Optional simulated answers to provide during the walk.

    Returns:
        Structured audit report confirming or disproving survey-side bugs.
    """
    machine = SurveyStateMachine(language=language, device=device)
    return machine.walk(target_question=target_question, mock_inputs=mock_inputs)


if __name__ == "__main__":
    print("=================================================================")
    print("STEP 4: SURVEY MODEL STATE MACHINE (CLOSED-LOOP SIMULATOR)")
    print("=================================================================")

    # Test Flagship Fault F01: Q7Sk on Mobile in Language 1046 (Portuguese)
    res_f01 = walk_survey(language="1046", device="mobile", target_question="Q7Sk")
    print("\n--- TEST 1: walk_survey(language='1046', device='mobile', target='Q7Sk') ---")
    print(f"Target State Rendered: {res_f01['target_state']['rendered_controls']}")
    print(f"Validation Status:     {res_f01['target_state']['validation_status']}")
    print(f"Experience Summary:    {res_f01['target_state']['experience_summary']}")
    print(f"Bugs Observed:         {res_f01['bugs_observed']}")

    # Contrast with Desktop in Language 1046 (Works normally)
    res_desktop = walk_survey(language="1046", device="desktop", target_question="Q7Sk")
    print("\n--- TEST 2: walk_survey(language='1046', device='desktop', target='Q7Sk') ---")
    print(f"Target State Rendered: {res_desktop['target_state']['rendered_controls']}")
    print(f"Bugs Observed:         {res_desktop['bugs_observed']}")

    # Test Fault F04: S5 accepts values > 100
    res_f04 = walk_survey(language="9", device="desktop", target_question="S5", mock_inputs={"S5": 115.0})
    print("\n--- TEST 3: walk_survey(target='S5', mock_inputs={'S5': 115.0}) ---")
    print(f"Validation Status:     {res_f04['target_state']['validation_status']}")
    print(f"Bugs Observed:         {res_f04['bugs_observed']}")

    # Test Fault F06: Routing bug where S2=1 enters Q9
    res_f06 = walk_survey(language="9", device="desktop", target_question="Q9", mock_inputs={"S2": "1"})
    print("\n--- TEST 4: walk_survey(target='Q9', mock_inputs={'S2': '1'}) ---")
    print(f"Experience:            {res_f06['target_state']['experience_summary']}")
    print(f"Bugs Observed:         {res_f06['bugs_observed']}")
