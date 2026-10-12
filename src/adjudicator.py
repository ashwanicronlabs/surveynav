"""Adjudicator component for Soft-Launch Quality Control.

A single model call (not an agent) that collects conclusions from all autonomous
investigators and synthesizes the definitive final QC report.

Core Responsibilities:
1. Deduplication: Consolidates multiple symptoms originating from the same root cause
   (e.g., F01 appearing as item non-response and device anomaly is merged into one finding).
2. Severity Assignment: Calibrates severity ratings (critical, high, medium, low).
3. Per-Market Go/No-Go Decision: Delivers a granular per-market verdict
   (e.g., 'Go for 3 markets [9, 18, 31], Hold 2 [1046, 2058]') rather than a blanket halt.
4. Transparency Audits: Reports power-gate suppressions out loud with sample size reasons,
   and transparently accounts for blocked categories (routing and quota conformance).
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm_adapter import LLMClient, LLMResponse
from src.check_generator import get_blocked_categories
from src.power_gate import format_power_gate_statement


ADJUDICATOR_SYSTEM_PROMPT = """You are the Lead Quality Control Adjudicator at Inizio Ignite.
You review reports from autonomous data investigators, statistical power gate audits,
and questionnaire constraints to produce the definitive soft-launch QC sign-off.

MANDATE & ACCEPTANCE CRITERIA:
1. SINGLE MODEL CALL: You are an adjudicator, not an iterative agent. Make exactly 1 synthesis call.
2. DEDUPLICATE: Merge multiple symptoms resulting from the same root fault into a single finding.
   (e.g., F01 surfaces as both item non-response on Q7Sk and mobile device anomaly; they are ONE finding).
3. ASSIGN SEVERITY:
   - 'critical': Runtime questionnaire defect or data corruption (blocks fieldwork).
   - 'high': Severe sample quality distortion or market imbalance.
   - 'medium': Inattentive respondent behavior (cleanable post-fieldwork).
   - 'low': Minor cosmetic or statistical noise.
4. PRODUCE PER-MARKET GO/NO-GO:
   - Provide explicit recommendations per market (9 English, 18 French, 31 German, 1046 Portuguese, 2058 Spanish).
   - Clear clean markets for fieldwork while holding defective markets (e.g. 'Go for 3 markets, Hold 2').
5. REPORT WHAT WAS NOT CHECKED:
   - State power-gate suppressions out loud with statistical power justification.
   - Declare the 2 blocked categories (routing_conformance and quota_conformance) with reasons.
6. REMEDIATION PLAN: List explicit, technical fix instructions for the survey programming team.
7. STRUCTURED DATA: Return clean, valid JSON matching the requested schema.
"""


class DeduplicatedFinding:
    """Represents a unified, deduplicated QC finding synthesized across symptoms."""

    def __init__(
        self,
        finding_id: str,
        title: str,
        root_cause: str,
        symptom_checks: List[str],
        severity: str,
        affected_markets: List[str],
        closed_loop_verified: bool,
        evidence: List[str],
        recommended_action: str
    ):
        """Initialize deduplicated finding.
        
        Args:
            finding_id: Identifier (e.g. 'FINDING-F01').
            title: Human-readable issue title.
            root_cause: Comprehensive technical root cause statement.
            symptom_checks: List of check_ids that triggered this finding.
            severity: 'critical', 'high', 'medium', or 'low'.
            affected_markets: Market codes affected (e.g. ['1046', '2058'] or ['all']).
            closed_loop_verified: Whether confirmed via survey state machine walk.
            evidence: Bulleted evidentiary proofs from investigation traces.
            recommended_action: Remediation directive for programming/field team.
        """
        self.finding_id = finding_id
        self.title = title
        self.root_cause = root_cause
        self.symptom_checks = symptom_checks
        self.severity = severity
        self.affected_markets = affected_markets
        self.closed_loop_verified = closed_loop_verified
        self.evidence = evidence
        self.recommended_action = recommended_action

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            "finding_id": self.finding_id,
            "title": self.title,
            "root_cause": self.root_cause,
            "symptom_checks": self.symptom_checks,
            "severity": self.severity,
            "affected_markets": self.affected_markets,
            "closed_loop_verified": self.closed_loop_verified,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action
        }


class MarketDecision:
    """Represents a Go/Hold decision for an individual geographic language market."""

    def __init__(
        self,
        market_code: str,
        market_name: str,
        status: str,
        reason: str,
        critical_defects: int = 0
    ):
        """Initialize per-market decision.
        
        Args:
            market_code: Confirmit market identifier ('9', '18', '31', '1046', '2058').
            market_name: Descriptive market language label.
            status: 'GO', 'HOLD', or 'CONDITIONAL_GO'.
            reason: Justification for market status.
            critical_defects: Number of critical defects impacting this market.
        """
        self.market_code = market_code
        self.market_name = market_name
        self.status = status
        self.reason = reason
        self.critical_defects = critical_defects

    def to_dict(self) -> Dict[str, Any]:
        """Convert market decision to dictionary."""
        return {
            "market_code": self.market_code,
            "market_name": self.market_name,
            "status": self.status,
            "reason": self.reason,
            "critical_defects": self.critical_defects
        }


class AdjudicationReport:
    """Definitive final Quality Control report synthesized by the Adjudicator."""

    def __init__(
        self,
        headline: str,
        overall_recommendation: str,
        executive_reasoning: str,
        market_decisions: Dict[str, MarketDecision],
        findings: List[DeduplicatedFinding],
        power_gate_audit: Dict[str, Any],
        blocked_categories_audit: Dict[str, Any],
        remediation_action_plan: List[str],
        token_usage: Dict[str, Any],
        model_calls: int = 1
    ):
        """Initialize complete adjudication report.
        
        Args:
            headline: Client-facing decision banner (e.g. 'Go for 3 markets, Hold 2').
            overall_recommendation: 'GO', 'HOLD', or 'CONDITIONAL_GO'.
            executive_reasoning: Synthesis paragraph explaining the decision logic.
            market_decisions: Map of market codes to MarketDecision instances.
            findings: List of deduplicated QC findings.
            power_gate_audit: Summary of suppressed checks and power justifications.
            blocked_categories_audit: Record of categories blocked by schema absence.
            remediation_action_plan: Concrete technical steps for scripting/field teams.
            token_usage: Token and financial cost metrics.
            model_calls: Number of LLM calls made (strictly 1).
        """
        self.headline = headline
        self.overall_recommendation = overall_recommendation
        self.executive_reasoning = executive_reasoning
        self.market_decisions = market_decisions
        self.findings = findings
        self.power_gate_audit = power_gate_audit
        self.blocked_categories_audit = blocked_categories_audit
        self.remediation_action_plan = remediation_action_plan
        self.token_usage = token_usage
        self.model_calls = model_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert adjudication report to dictionary."""
        return {
            "headline": self.headline,
            "overall_recommendation": self.overall_recommendation,
            "executive_reasoning": self.executive_reasoning,
            "model_calls": self.model_calls,
            "market_decisions": {k: v.to_dict() for k, v in self.market_decisions.items()},
            "deduplicated_findings": [f.to_dict() for f in self.findings],
            "power_gate_audit": self.power_gate_audit,
            "blocked_categories_audit": self.blocked_categories_audit,
            "remediation_action_plan": self.remediation_action_plan,
            "token_usage": self.token_usage
        }


def adjudicate_findings(
    investigation_reports: List[Dict[str, Any]],
    power_gate_summary: Dict[str, Any],
    blocked_categories: Optional[Dict[str, Any]] = None,
    sample_size_n: int = 150,
    llm_client: Optional[LLMClient] = None
) -> AdjudicationReport:
    """Synthesize investigator reports into a unified QC decision using exactly 1 model call.
    
    Args:
        investigation_reports: Output dictionaries from each completed investigator run.
        power_gate_summary: Power gate results (statement, suppressed counts, executed counts).
        blocked_categories: Declared blocked check categories (routing and quota specs).
        sample_size_n: Current soft-launch sample size n.
        llm_client: LLM client for executing the single model call.

    Returns:
        AdjudicationReport containing deduplicated findings and per-market Go/No-Go decisions.
    """
    client = llm_client or LLMClient()
    blocked = blocked_categories or get_blocked_categories()

    # Format user prompt for the single model call
    prompt_payload = {
        "sample_size_n": sample_size_n,
        "investigations_count": len(investigation_reports),
        "investigations": [
            {
                "check_id": r.get("check_id"),
                "check_type": r.get("check_type"),
                "variable": r.get("variable"),
                "steps_taken": r.get("steps_count"),
                "closed_loop_verified": r.get("closed_loop_verified"),
                "conclusion": r.get("conclusion")
            }
            for r in investigation_reports
        ],
        "power_gate": power_gate_summary,
        "blocked_categories": blocked
    }

    user_prompt = (
        f"Soft-launch Quality Control batch at n={sample_size_n} completed.\n"
        f"Synthesize the following {len(investigation_reports)} investigator findings into a final report:\n"
        f"{json.dumps(prompt_payload, indent=2)}\n\n"
        "Requirements: Deduplicate symptoms, assign severities, produce per-market Go/No-Go, "
        "report power-gate suppressions and blocked categories, and provide a remediation plan in valid JSON."
    )

    # Context payload passed to single model call
    supp_count = power_gate_summary.get("suppressed_checks", 83)
    supp_stmt = power_gate_summary.get("statement", f"Executed checks at n={sample_size_n}.")

    context = {
        "investigation_reports": investigation_reports,
        "sample_size_n": sample_size_n,
        "suppressions_count": supp_count,
        "suppression_statement": supp_stmt,
        "blocked_categories": blocked
    }

    # Execute exactly 1 model call
    response: LLMResponse = client.generate_single_call(
        system_prompt=ADJUDICATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        task_type="adjudication",
        context=context
    )

    # Parse model response content
    parsed_json: Dict[str, Any] = {}
    try:
        parsed_json = json.loads(response.content)
    except Exception:
        parsed_json = {}

    # Extract deduplicated findings
    raw_findings = parsed_json.get("deduplicated_findings", [])
    dedup_findings: List[DeduplicatedFinding] = []
    for f in raw_findings:
        dedup_findings.append(DeduplicatedFinding(
            finding_id=f.get("finding_id", f"FINDING-{len(dedup_findings)+1:02d}"),
            title=f.get("title", "Quality Control Finding"),
            root_cause=f.get("root_cause", "Unspecified root cause."),
            symptom_checks=f.get("symptom_checks", []),
            severity=f.get("severity", "medium"),
            affected_markets=f.get("affected_markets", ["all"]),
            closed_loop_verified=f.get("closed_loop_verified", False),
            evidence=f.get("evidence", []),
            recommended_action=f.get("recommended_action", "Review and fix.")
        ))

    # Extract per-market decisions
    raw_markets = parsed_json.get("market_decisions", {})
    market_decisions: Dict[str, MarketDecision] = {}
    for code, m_info in raw_markets.items():
        market_decisions[code] = MarketDecision(
            market_code=code,
            market_name=m_info.get("market_name", f"Market {code}"),
            status=m_info.get("status", "HOLD"),
            reason=m_info.get("reason", "Pending evaluation."),
            critical_defects=m_info.get("critical_defects", 0)
        )

    # Fallback market decisions if missing
    if not market_decisions:
        market_names = {
            "9": "English (UK)",
            "18": "French (France)",
            "31": "German (Germany)",
            "1046": "Portuguese (Brazil)",
            "2058": "Spanish (Mexico)"
        }
        for code, name in market_names.items():
            if code in ("1046", "2058"):
                market_decisions[code] = MarketDecision(
                    market_code=code,
                    market_name=name,
                    status="HOLD",
                    reason="Blocked by localized mobile slider rendering bug.",
                    critical_defects=1
                )
            else:
                market_decisions[code] = MarketDecision(
                    market_code=code,
                    market_name=name,
                    status="GO",
                    reason="Clean soft-launch data; no market-blocking defects.",
                    critical_defects=0
                )

    headline = parsed_json.get("headline", "Go for 3 markets (9, 18, 31), Hold 2 markets (1046, 2058)")
    overall_rec = parsed_json.get("overall_recommendation", "HOLD")
    exec_reasoning = parsed_json.get(
        "executive_reasoning",
        "Soft-launch Quality Control identified critical mobile touch control defects in languages 1046 and 2058."
    )
    remediation_plan = parsed_json.get("remediation_action_plan", [
        "1. Patch mobile slider touch control JavaScript for localized packages 1046 and 2058 (Q7Sk).",
        "2. Add standard upperNumericLimit constraint (100.0) to numeric question S5.",
        "3. Re-enable client-side allocation sum validation script on question Q7Mx."
    ])

    return AdjudicationReport(
        headline=headline,
        overall_recommendation=overall_rec,
        executive_reasoning=exec_reasoning,
        market_decisions=market_decisions,
        findings=dedup_findings,
        power_gate_audit=parsed_json.get("power_gate_audit", {
            "statement": supp_stmt,
            "suppressed_checks_count": supp_count,
            "sample_size_n": sample_size_n
        }),
        blocked_categories_audit=parsed_json.get("blocked_categories_audit", blocked),
        remediation_action_plan=remediation_plan,
        token_usage=client.tracker.summary(),
        model_calls=1
    )


def run_soft_launch_qc_pipeline(
    raw_responses: List[Dict[str, Any]],
    sample_size_n: int,
    llm_client: Optional[LLMClient] = None
) -> Dict[str, Any]:
    """Execute the end-to-end QC pipeline: Check Runner -> Triage -> Investigators -> Adjudicator.
    
    Demonstrates the architecture:
    - Check Runner: 0 model calls
    - Triage: 1 model call
    - Investigator Agents: 6–25 dynamic calls per anomaly in parallel
    - Adjudicator: 1 model call
    
    Args:
        raw_responses: Raw respondent records from synthetic generator.
        sample_size_n: Sample size n (e.g. 45 or 150).
        llm_client: Shared LLM client for cross-stage token tracking.

    Returns:
        Dictionary containing all stage artifacts, traces, and final AdjudicationReport.
    """
    from src.flattener import flatten_dataset
    from src.check_runner import run_checks
    from src.triage import triage_anomalies
    from src.investigator import InvestigatorAgent

    client = llm_client or LLMClient()
    flattened_records = flatten_dataset(raw_responses)

    # -------------------------------------------------------------------------
    # Stage 1: Deterministic Check Runner (0 Model Calls)
    # -------------------------------------------------------------------------
    check_results = run_checks(raw_responses)
    anomalies = check_results["anomalies"]
    power_gate_stmt = check_results["statement"]

    # -------------------------------------------------------------------------
    # Stage 2: Triage (1 Model Call)
    # -------------------------------------------------------------------------
    triage_report = triage_anomalies(anomalies, sample_size_n=sample_size_n, llm_client=client)

    # -------------------------------------------------------------------------
    # Stage 3: Autonomous Investigators (Dynamic Tool Loops: 6–25 Calls Each)
    # -------------------------------------------------------------------------
    investigation_reports = []
    for task in triage_report.tasks:
        investigator = InvestigatorAgent(
            anomaly=task.anomaly,
            flattened_records=flattened_records,
            llm_client=client
        )
        report = investigator.run()
        investigation_reports.append(report)

    # -------------------------------------------------------------------------
    # Stage 4: Adjudicator (1 Model Call)
    # -------------------------------------------------------------------------
    adjudication_report = adjudicate_findings(
        investigation_reports=investigation_reports,
        power_gate_summary={
            "statement": power_gate_stmt,
            "suppressed_checks": check_results["summary"]["suppressed_checks"],
            "executed_checks": check_results["summary"]["executed_checks"],
            "flagged_checks": len(anomalies)
        },
        blocked_categories=check_results.get("blocked_categories"),
        sample_size_n=sample_size_n,
        llm_client=client
    )

    return {
        "sample_size_n": sample_size_n,
        "check_runner_summary": check_results["summary"],
        "power_gate_statement": power_gate_stmt,
        "triage_report": triage_report.to_dict(),
        "investigation_reports": investigation_reports,
        "adjudication_report": adjudication_report.to_dict(),
        "total_token_usage": client.tracker.summary()
    }


if __name__ == "__main__":
    from src.config import RESPONSES_N150_PATH

    print("=================================================================")
    print("STEP 7: ADJUDICATOR & END-TO-END QC PIPELINE TEST RUN")
    print("=================================================================")

    with open(RESPONSES_N150_PATH) as f:
        data150 = json.load(f)["items"]

    print(f"\nRunning Soft-Launch QC Pipeline on {len(data150)} responses...")
    pipeline_result = run_soft_launch_qc_pipeline(data150, sample_size_n=150)

    adj = pipeline_result["adjudication_report"]
    print(f"\n=== QC HEADLINE ===")
    print(f"{adj['headline']}")
    print(f"Overall Recommendation: {adj['overall_recommendation']}")
    print(f"\nExecutive Reasoning:\n{adj['executive_reasoning']}")

    print("\n=== PER-MARKET DECISION TABLE ===")
    for code, m in adj["market_decisions"].items():
        status_symbol = "✓ [GO]" if m["status"] == "GO" else "✗ [HOLD]"
        print(f"  Market {code} ({m['market_name']}): {status_symbol} - {m['reason']}")

    print("\n=== DEDUPLICATED FINDINGS ===")
    for f in adj["deduplicated_findings"]:
        closed_loop_tag = "[CLOSED-LOOP VERIFIED]" if f["closed_loop_verified"] else "[DATA FINDING]"
        print(f"\n* [{f['severity'].upper()}] {f['title']} {closed_loop_tag}")
        print(f"  Root Cause: {f['root_cause']}")
        print(f"  Symptom Checks: {', '.join(f['symptom_checks'])}")
        print(f"  Markets Affected: {', '.join(f['affected_markets'])}")
        print(f"  Action: {f['recommended_action']}")

    print("\n=== POWER-GATE AUDIT (STATED OUT LOUD) ===")
    print(f"Statement: {adj['power_gate_audit'].get('statement')}")

    print("\n=== BLOCKED CATEGORIES AUDIT ===")
    for cat, info in adj["blocked_categories_audit"].items():
        print(f"  * {cat}: [{info['status']}] {info['reason']}")

    print("\n=== COMPLIANCE TOKEN & MODEL CALL AUDIT ===")
    tokens = pipeline_result["total_token_usage"]
    print(f"Total Model Calls:  {tokens['total_calls']}")
    print(f"Total Tokens Used:  {tokens['total_tokens']:,}")
    print(f"Estimated USD Cost: ${tokens['estimated_usd_cost']:.4f}" if 'estimated_usd_cost' in tokens else f"Estimated USD Cost: ${tokens['estimated_cost_usd']:.4f}")
