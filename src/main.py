"""SurveyNav REQ-009: Automated Soft-Launch QC Autonomous Agent Platform.

Main command-line entrypoint delivering the executive terminal demonstration.
Executes deterministic rules engine, statistical power gate, triage, autonomous
investigators, adjudicator, and ground truth benchmark scoring.

Supports side-by-side evaluation of n=45 vs n=150 to demonstrate:
1. Power gate suppression out loud at n=45 vs full execution at n=150.
2. Single-agent architecture (Check Runner: 0, Triage: 1, Investigator: 6–25, Adjudicator: 1).
3. Closed-loop survey walker reproducing mobile UI faults in milliseconds.
4. Per-market Go/No-Go recommendations.
5. Ground truth scoring against the 8 planted faults with honest miss explanations.
"""
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import box

from src.config import RESPONSES_N45_PATH, RESPONSES_N150_PATH, RUNS_DIR
from src.llm_adapter import LLMClient
from src.adjudicator import run_soft_launch_qc_pipeline
from src.scorer import score_adjudication_report, ScoringReport


console = Console()


def display_banner() -> None:
    """Print executive system banner."""
    title_text = Text("INIZIO IGNITE · SURVEYNAV REQ-009", style="bold cyan")
    sub_text = Text(
        "Automated Soft-Launch QC Platform · 339 Rules · Statistical Power Gate · Autonomous Investigator",
        style="dim white"
    )
    banner_content = Text.assemble(title_text, "\n", sub_text)
    console.print(Panel(banner_content, border_style="cyan", box=box.DOUBLE))


def run_single_evaluation(
    sample_size_n: int,
    llm_client: Optional[LLMClient] = None,
    save_trace: bool = True
) -> Dict[str, Any]:
    """Execute complete end-to-end QC and benchmark scoring for a specific sample size.
    
    Args:
        sample_size_n: Sample size n (45 or 150).
        llm_client: Unified LLM client adapter.
        save_trace: Whether to write audit logs to runs/.

    Returns:
        Consolidated run results dictionary including pipeline and scoring metrics.
    """
    client = llm_client or LLMClient()
    path = RESPONSES_N45_PATH if sample_size_n < 100 else RESPONSES_N150_PATH

    with open(path) as f:
        data = json.load(f)["items"]

    # 1. Execute end-to-end QC pipeline
    pipeline_result = run_soft_launch_qc_pipeline(
        raw_responses=data,
        sample_size_n=sample_size_n,
        llm_client=client
    )

    # 2. Score against Ground Truth benchmark
    score_report: ScoringReport = score_adjudication_report(
        adjudication_report=pipeline_result["adjudication_report"],
        sample_size_n=sample_size_n,
        token_usage=pipeline_result["total_token_usage"]
    )

    combined_run = {
        "timestamp": datetime.now().isoformat(),
        "sample_size_n": sample_size_n,
        "pipeline": pipeline_result,
        "score_report": score_report.to_dict()
    }

    # 3. Write compliance audit trace
    if save_trace:
        os.makedirs(RUNS_DIR, exist_ok=True)
        trace_file = RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_n{sample_size_n}.json"
        with open(trace_file, "w") as f:
            json.dump(combined_run, f, indent=2)
        combined_run["trace_file"] = str(trace_file)

    return combined_run


def display_side_by_side_comparison(run45: Dict[str, Any], run150: Dict[str, Any]) -> None:
    """Render side-by-side terminal comparison table contrasting n=45 and n=150."""
    p45 = run45["pipeline"]
    p150 = run150["pipeline"]
    s45 = run45["score_report"]
    s150 = run150["score_report"]
    adj45 = p45["adjudication_report"]
    adj150 = p150["adjudication_report"]

    table = Table(
        title="[bold yellow]SIDE-BY-SIDE SOFT-LAUNCH QC EVALUATION[/bold yellow]",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    table.add_column("Pipeline Metric / Decision", style="bold white", width=34)
    table.add_column("Batch 1: n = 45 (Underpowered)", style="yellow", width=36)
    table.add_column("Batch 2: n = 150 (Full Scale)", style="green", width=36)

    # 1. Power Gate Statement
    table.add_row(
        "Power Gate Statement",
        p45["power_gate_statement"],
        p150["power_gate_statement"]
    )

    # 2. Executed vs Suppressed Checks
    table.add_row(
        "Executed Checks / Suppressed",
        f"{p45['check_runner_summary']['executed_checks']} executed / [bold yellow]{p45['check_runner_summary']['suppressed_checks']} suppressed[/bold yellow]",
        f"{p150['check_runner_summary']['executed_checks']} executed / [bold green]{p150['check_runner_summary']['suppressed_checks']} suppressed[/bold green]"
    )

    # 3. Flagged Anomalies
    table.add_row(
        "Flagged Anomalies for Triage",
        f"[bold red]{len(p45['triage_report']['tasks'])} flagged[/bold red] (5 faults active)",
        f"[bold red]{len(p150['triage_report']['tasks'])} flagged[/bold red] (disposition mix now active)"
    )

    # 4. Model Architecture & Calls
    table.add_row(
        "Model Calls Architecture",
        f"Runner: 0 | Triage: 1 | Agent: {len(p45['investigation_reports'])} | Adj: 1",
        f"Runner: 0 | Triage: 1 | Agent: {len(p150['investigation_reports'])} | Adj: 1"
    )

    # 5. Recommendation
    table.add_row(
        "Adjudicator Recommendation",
        f"[bold yellow]{adj45['headline']}[/bold yellow]",
        f"[bold red]{adj150['headline']}[/bold red]"
    )

    # 6. Benchmark Detection Rate
    table.add_row(
        "Ground Truth Detection Rate",
        f"{s45['detected_count']}/{s45['total_planted_faults']} ([bold yellow]{s45['detection_rate']:.1%}[/bold yellow])",
        f"{s150['detected_count']}/{s150['total_planted_faults']} ([bold green]{s150['detection_rate']:.1%}[/bold green])"
    )

    # 7. Diagnosis Accuracy
    table.add_row(
        "Root Cause Diagnosis Accuracy",
        f"{s45['diagnosed_count']}/{s45['detected_count']} ([bold green]{s45['diagnosis_accuracy']:.1%}[/bold green])",
        f"{s150['diagnosed_count']}/{s150['detected_count']} ([bold green]{s150['diagnosis_accuracy']:.1%}[/bold green])"
    )

    # 8. Token Economics
    tok45 = p45["total_token_usage"]
    tok150 = p150["total_token_usage"]
    table.add_row(
        "Token Economics & Cost",
        f"{tok45['total_tokens']:,} tokens (${tok45['estimated_cost_usd']:.4f})",
        f"{tok150['total_tokens']:,} tokens (${tok150['estimated_cost_usd']:.4f})"
    )

    console.print(table)


def display_flagship_trace(run150: Dict[str, Any]) -> None:
    """Render detailed step-by-step trace of the flagship closed-loop investigation (F01)."""
    investigations = run150["pipeline"]["investigation_reports"]
    f01_report = next((r for r in investigations if "Q7Sk" in r.get("check_id", "")), investigations[0])

    console.print()
    console.print(Panel(
        f"[bold cyan]FLAGSHIP REASONING TRACE: {f01_report['check_id']}[/bold cyan]\n"
        f"[dim white]Autonomous multi-step tool loop · Dead end exploration · Closed-loop survey walker confirmation[/dim white]",
        border_style="cyan"
    ))

    tree = Tree("[bold green]Investigator Agent Workflow[/bold green]")
    for step in f01_report.get("trace", []):
        num = step["step_number"]
        tool = step["tool_name"]
        thought = step["thought"]
        args = json.dumps(step["tool_args"])
        
        step_branch = tree.add(f"[bold yellow]Step {num}:[/bold yellow] [bold white]{tool}[/bold white]({args})")
        step_branch.add(f"[italic dim white]{thought}[/italic dim white]")

    console.print(tree)

    conc = f01_report.get("conclusion", {})
    if conc:
        console.print(Panel(
            f"[bold green]Final Conclusion Formulated by Agent:[/bold green]\n"
            f"[bold white]Root Cause:[/bold white] {conc.get('cause')}\n"
            f"[bold white]Severity:[/bold white] [bold red]{conc.get('severity').upper()}[/bold red] | "
            f"[bold white]Confidence:[/bold white] [bold cyan]{conc.get('confidence').upper()}[/bold cyan] | "
            f"[bold white]Affected Markets:[/bold white] {', '.join(conc.get('affected_markets', []))}\n"
            f"[bold white]Reproduction Steps:[/bold white] {conc.get('reproduction_steps')}",
            border_style="green"
        ))


def display_adjudication_details(run150: Dict[str, Any]) -> None:
    """Render per-market Go/No-Go decisions and deduplicated findings."""
    adj = run150["pipeline"]["adjudication_report"]

    console.print()
    # Market decisions table
    market_table = Table(
        title="[bold cyan]PER-MARKET GO / NO-GO SIGN-OFF TABLE[/bold cyan]",
        box=box.SIMPLE_HEAVY
    )
    market_table.add_column("Market Code", style="bold white", width=12)
    market_table.add_column("Language Market", style="white", width=22)
    market_table.add_column("Status", width=12)
    market_table.add_column("Sign-Off Justification", style="dim white")

    for code, m in adj["market_decisions"].items():
        if m["status"] == "GO":
            status_cell = "[bold green]✓ GO[/bold green]"
        else:
            status_cell = "[bold red]✗ HOLD[/bold red]"
        market_table.add_row(code, m["market_name"], status_cell, m["reason"])

    console.print(market_table)

    # Blocked categories panel
    blocked_txt = []
    for cat, info in adj.get("blocked_categories_audit", {}).items():
        blocked_txt.append(f"• [bold yellow]{cat}[/bold yellow] [[bold red]{info['status']}[/bold red]]: {info['reason']}")

    console.print(Panel(
        "\n".join(blocked_txt),
        title="[bold yellow]Declared Blocked QC Categories (Confirmit Schema Constraints)[/bold yellow]",
        border_style="yellow"
    ))


def display_ground_truth_scorecard(score_report: Dict[str, Any]) -> None:
    """Render ground truth scorecard with transparent miss reporting."""
    console.print()
    score_table = Table(
        title="[bold green]GROUND TRUTH BENCHMARK SCORECARD (8 PLANTED FAULTS)[/bold green]",
        box=box.ROUNDED
    )
    score_table.add_column("Fault ID", style="bold cyan", width=10)
    score_table.add_column("Target Variable", style="white", width=16)
    score_table.add_column("Planted Fault Description", width=38)
    score_table.add_column("Benchmark Result", width=26)

    for ev in score_report["fault_evaluations"]:
        if ev["detected"] and ev["diagnosed_correctly"]:
            res = "[bold green]✓ DETECTED & DIAGNOSED[/bold green]"
        elif ev["detected"]:
            res = "[bold yellow]⚠ DETECTED (PARTIAL)[/bold yellow]"
        else:
            res = "[bold red]✗ MISSED (EXPLAINED)[/bold red]"
        score_table.add_row(ev["fault_id"], ev["target_variable"], ev["description"], res)

    console.print(score_table)

    # Miss reporting
    misses = score_report.get("misses_breakdown", [])
    if misses:
        miss_lines = []
        for m in misses:
            miss_lines.append(f"• [bold cyan]{m['fault_id']} ({m['target_variable']}):[/bold cyan] {m['reason']}")
        console.print(Panel(
            "\n".join(miss_lines),
            title="[bold white]Transparent Miss Reporting (Client Credibility Audit)[/bold white]",
            border_style="cyan"
        ))


def main() -> None:
    """CLI entrypoint parsing options and running the evaluation demo."""
    parser = argparse.ArgumentParser(description="SurveyNav REQ-009 Soft-Launch QC Platform")
    parser.add_argument("--n", type=int, choices=[45, 150], default=None, help="Sample size to evaluate (45 or 150)")
    parser.add_argument("--both", action="store_true", default=True, help="Run side-by-side comparison of n=45 and n=150")
    parser.add_argument("--provider", type=str, default="auto", help="LLM Provider: auto, mock, gemini, anthropic, openai")
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON audit traces to runs/")

    args = parser.parse_args()
    display_banner()

    save_traces = not args.no_save
    client = LLMClient(provider=args.provider)

    if args.n == 45:
        console.print("\n[bold yellow]Executing Single Evaluation at n=45...[/bold yellow]\n")
        run45 = run_single_evaluation(45, llm_client=client, save_trace=save_traces)
        display_ground_truth_scorecard(run45["score_report"])
        if "trace_file" in run45:
            console.print(f"\n[dim green]Audit trace saved to: {run45['trace_file']}[/dim green]")

    elif args.n == 150:
        console.print("\n[bold green]Executing Single Evaluation at n=150...[/bold green]\n")
        run150 = run_single_evaluation(150, llm_client=client, save_trace=save_traces)
        display_flagship_trace(run150)
        display_adjudication_details(run150)
        display_ground_truth_scorecard(run150["score_report"])
        if "trace_file" in run150:
            console.print(f"\n[dim green]Audit trace saved to: {run150['trace_file']}[/dim green]")

    else:
        # Default flagship mode: Side-by-side run of n=45 and n=150
        console.print("\n[bold cyan]Executing Flagship Side-by-Side Soft-Launch Evaluation (n=45 vs n=150)...[/bold cyan]\n")
        run45 = run_single_evaluation(45, llm_client=client, save_trace=save_traces)
        run150 = run_single_evaluation(150, llm_client=client, save_trace=save_traces)

        display_side_by_side_comparison(run45, run150)
        display_flagship_trace(run150)
        display_adjudication_details(run150)
        display_ground_truth_scorecard(run150["score_report"])

        console.print("\n[bold cyan]Compliance Audit Traces Generated:[/bold cyan]")
        if "trace_file" in run45:
            console.print(f"  • n=45 Trace:  [dim white]{run45['trace_file']}[/dim white]")
        if "trace_file" in run150:
            console.print(f"  • n=150 Trace: [dim white]{run150['trace_file']}[/dim white]")


if __name__ == "__main__":
    main()
