"""Autonomous Investigator Agent for Soft-Launch QC Anomalies.

Launches a dedicated, isolated investigator instance per anomaly.
Executes an autonomous tool-calling loop (6–25 calls) guided by prompt directives:
- State your theory before you test it.
- If a theory does not hold, declare a dead end and pivot.
- Close the loop by walking the survey model before concluding.
- Terminate strictly through the structured conclude tool.
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tools import (
    get_schema,
    query_data,
    sample_respondents,
    walk_survey,
    conclude,
    InvestigationGuardrails,
    INVESTIGATOR_TOOL_DEFINITIONS
)
from src.llm_adapter import LLMClient, LLMResponse, MockInvestigatorPolicy


INVESTIGATOR_SYSTEM_PROMPT = """You are an expert Survey Data Quality Investigator at Inizio Ignite.
You are tasked with investigating an anomaly flagged during soft-launch quality control.

YOUR MANDATE:
1. Work out the ROOT CAUSE of why this anomaly is occurring in the data.
2. STATE YOUR HYPOTHESIS BEFORE EACH TEST. Never run queries blindly without explaining what you are looking for.
3. EMBRACE AND DECLARE DEAD ENDS: If a query disproves your theory, explicitly declare it as a dead end and pivot to a new hypothesis.
4. TEST AGAINST THE SURVEY MODEL: If you suspect a questionnaire, UI rendering, validation, or routing defect, call `walk_survey` to verify whether the live survey behaves that way.
5. ISOLATED MEMORY: You have your own independent investigation. Do not assume conclusions from other checks.
6. TERMINATION: Conclude ONLY by invoking the `conclude` tool with structured evidence, severity, and reproduction steps.

AVAILABLE TOOLS:
- `get_schema(variable)`: Look up question type, options, limits, sub-items.
- `query_data(target_variable, group_by, filter_dict, metric)`: Segment data by language ('l'), device ('lastdevicetype'), source ('bgSource'). Capped at top 20 rows.
- `sample_respondents(filter_dict, n)`: Eyeball up to 5 individual records.
- `walk_survey(language, device, target_question, mock_inputs)`: Step through the questionnaire runtime as a respondent to confirm survey-side bugs.
- `conclude(cause, confidence, evidence, severity, affected_markets, reproduction_steps)`: Finish and submit findings.
"""


class InvestigationStep:
    """Represents a single step in the agent's reasoning trace."""

    def __init__(
        self,
        step_number: int,
        thought: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any]
    ):
        self.step_number = step_number
        self.thought = thought
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result

    def to_dict(self) -> Dict[str, Any]:
        """Convert step record to dictionary."""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result
        }


class InvestigatorAgent:
    """Autonomous agent investigating a single soft-launch anomaly to identify root cause."""

    def __init__(
        self,
        anomaly: Dict[str, Any],
        flattened_records: List[Dict[str, Any]],
        llm_client: Optional[LLMClient] = None,
        max_tool_calls: int = 25
    ):
        """Initialize investigator with isolated conversation context and guardrails.
        
        Args:
            anomaly: Standardized anomaly record from check_runner.
            flattened_records: Full dataset of flattened respondent dictionaries.
            llm_client: LLM client interface (defaults to auto-detected client).
            max_tool_calls: Hard ceiling for tool calls before safety abort.
        """
        self.anomaly = anomaly
        self.records = flattened_records
        self.client = llm_client or LLMClient()
        self.guardrails = InvestigationGuardrails(max_calls=max_tool_calls)
        self.policy = MockInvestigatorPolicy(anomaly)
        
        # Isolated conversation memory (never shared across agents)
        self.messages: List[Dict[str, Any]] = []
        self.trace: List[InvestigationStep] = []
        self.conclusion: Optional[Dict[str, Any]] = None
        self.dead_ends: List[str] = []
        self.closed_loop_verified = False

    def run(self) -> Dict[str, Any]:
        """Execute the autonomous multi-step tool-use investigation loop.
        
        Returns:
            Structured investigation findings including step count, reasoning trace,
            dead-end count, closed-loop status, and final conclusion.
        """
        # Seed initial prompt with the flagged anomaly
        initial_user_msg = (
            f"Please investigate the following QC anomaly:\n"
            f"Check ID: {self.anomaly.get('check_id')}\n"
            f"Check Type: {self.anomaly.get('check_type')}\n"
            f"Variable: {self.anomaly.get('variable')}\n"
            f"Observed Statistic: {self.anomaly.get('statistic')}\n"
            f"Expected Threshold: {self.anomaly.get('expected')}\n"
            f"Description: {self.anomaly.get('description')}\n"
            f"Affected Count: {len(self.anomaly.get('affected_respondents', []))}\n"
            f"Find out why this problem is happening and verify whether it is a survey script bug."
        )
        self.messages.append({"role": "user", "content": initial_user_msg})

        step_counter = 0

        # Autonomous loop: continues until conclude() tool is called or guardrails abort
        while not self.conclusion and not self.guardrails.is_aborted:
            step_counter += 1

            # 1. Model call
            response: LLMResponse = self.client.chat_with_tools(
                system_prompt=INVESTIGATOR_SYSTEM_PROMPT,
                messages=self.messages,
                tools=INVESTIGATOR_TOOL_DEFINITIONS,
                mock_policy=self.policy
            )

            # Record thought text
            thought_text = response.content
            if "dead end" in thought_text.lower():
                self.dead_ends.append(thought_text)

            # Check if model made tool calls
            if not response.tool_calls:
                # If model ended without tool calling, synthesize fallback conclusion
                self.conclusion = conclude(
                    cause=f"Investigation concluded: {thought_text[:120]}",
                    confidence="medium",
                    evidence=["Statistical pattern observed."],
                    severity="medium"
                )
                break

            # 2. Dispatch each requested tool call
            tool_results_for_turn = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})

                # Validate against guardrails (call limit and loop detection)
                is_allowed, block_reason = self.guardrails.record_and_validate(tool_name, tool_args)
                if not is_allowed:
                    # Guardrail tripped
                    tool_result = {"error": block_reason}
                    self.conclusion = conclude(
                        cause=f"Investigation halted by guardrail: {block_reason}",
                        confidence="low",
                        evidence=["Safety limits triggered to prevent runaway execution."],
                        severity="low"
                    )
                    tool_results_for_turn.append(tool_result)
                    break

                # Execute requested tool
                tool_result = self._dispatch_tool(tool_name, tool_args)
                tool_results_for_turn.append(tool_result)

                # Record reasoning trace step
                self.trace.append(InvestigationStep(
                    step_number=step_counter,
                    thought=thought_text,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result
                ))

                # Check if walk_survey verified a closed loop
                if tool_name == "walk_survey" and tool_result.get("bugs_observed"):
                    self.closed_loop_verified = True

                # Check if conclude was called
                if tool_name == "conclude":
                    self.conclusion = tool_result
                    break

            # 3. Update conversation history
            self.messages.append({
                "role": "assistant",
                "content": thought_text,
                "tool_calls": response.tool_calls
            })
            self.messages.append({
                "role": "user",
                "content": json.dumps(tool_results_for_turn)
            })

        return {
            "check_id": self.anomaly.get("check_id"),
            "variable": self.anomaly.get("variable"),
            "check_type": self.anomaly.get("check_type"),
            "steps_count": len(self.trace),
            "dead_ends_count": len(self.dead_ends),
            "closed_loop_verified": self.closed_loop_verified,
            "conclusion": self.conclusion,
            "guardrail_aborted": self.guardrails.is_aborted,
            "abort_reason": self.guardrails.abort_reason,
            "token_usage": self.client.tracker.summary(),
            "trace": [s.to_dict() for s in self.trace]
        }

    def _dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool function with arguments in investigator context."""
        if tool_name == "get_schema":
            return get_schema(args.get("variable", ""))

        elif tool_name == "query_data":
            return query_data(
                records=self.records,
                target_variable=args.get("target_variable"),
                group_by=args.get("group_by"),
                filter_dict=args.get("filter_dict"),
                metric=args.get("metric", "count")
            )

        elif tool_name == "sample_respondents":
            return sample_respondents(
                records=self.records,
                filter_dict=args.get("filter_dict"),
                n=args.get("n", 3)
            )

        elif tool_name == "walk_survey":
            return walk_survey(
                language=args.get("language", "9"),
                device=args.get("device", "desktop"),
                target_question=args.get("target_question"),
                mock_inputs=args.get("mock_inputs")
            )

        elif tool_name == "conclude":
            return conclude(
                cause=args.get("cause", ""),
                confidence=args.get("confidence", "medium"),
                evidence=args.get("evidence", []),
                severity=args.get("severity", "medium"),
                affected_markets=args.get("affected_markets"),
                reproduction_steps=args.get("reproduction_steps")
            )

        else:
            return {"error": f"Unknown tool '{tool_name}'."}


def investigate_anomaly(
    anomaly: Dict[str, Any],
    flattened_records: List[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None
) -> Dict[str, Any]:
    """Convenience helper to launch an isolated investigator agent for an anomaly."""
    agent = InvestigatorAgent(anomaly, flattened_records, llm_client)
    return agent.run()


if __name__ == "__main__":
    from src.config import RESPONSES_N150_PATH
    from src.flattener import flatten_dataset

    print("=================================================================")
    print("STEP 6: AUTONOMOUS INVESTIGATOR AGENT (LIVE DEMO TRACE)")
    print("=================================================================")

    with open(RESPONSES_N150_PATH) as f:
        records_raw = json.load(f)["items"]
    records_flat = flatten_dataset(records_raw)

    # Launch investigation for the flagship closed-loop anomaly F01 (Q7Sk)
    anomaly_f01 = {
        "check_id": "item_non_response.Q7Sk",
        "check_type": "item_non_response",
        "variable": "Q7Sk",
        "statistic": 0.0899,
        "expected": 0.05,
        "description": "Missing rate of 9.0% exceeds expected maximum of 5.0%",
        "affected_respondents": [200003, 200009]
    }

    print(f"\nLaunching Investigator Agent for: {anomaly_f01['check_id']}...\n")
    report = investigate_anomaly(anomaly_f01, records_flat)

    print(f"Investigation Finished in {report['steps_count']} steps.")
    print(f"Dead Ends Encountered:     {report['dead_ends_count']}")
    print(f"Closed-Loop Verified:      {report['closed_loop_verified']}")
    print("\nREASONING TRACE:")
    for step in report["trace"]:
        print(f"\n[Step {step['step_number']}] Tool: {step['tool_name']}({json.dumps(step['tool_args'])})")
        print(f"  Thought: {step['thought']}")
        if step['tool_name'] == 'conclude':
            print(f"  Conclusion: {step['tool_result']['cause']}")
            print(f"  Confidence: {step['tool_result']['confidence']} | Severity: {step['tool_result']['severity']}")
            print(f"  Reproduction Steps: {step['tool_result']['reproduction_steps']}")

    print("\nToken & Cost Totals:")
    print(report["token_usage"])
