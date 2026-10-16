"""Runner CLI with Rich benchmark reporting (Day 5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agent_trace import __version__
from agent_trace.runner.suite import load_suite, run_suite, summarize

console = Console()


def cmd_list(_args: argparse.Namespace) -> int:
    from agent_trace.scenarios.builtin import builtin_scenarios
    table = Table(title="Agent Trace scenarios")
    table.add_column("id", style="cyan")
    table.add_column("title")
    table.add_column("adversarial", justify="center")
    table.add_column("tools required")
    for s in builtin_scenarios():
        table.add_row(s.id, s.title, "yes" if s.adversarial else "no",
                      ",".join(s.expected.required_tools) or "-")
    console.print(table)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ids = args.scenario or None  # --scenario can repeat; None = all
    try:
        scenarios = load_suite(ids)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    use_llm = bool(args.llm)
    if args.llm and not __import__("os").getenv("OPENAI_API_KEY"):
        console.print("[yellow]--llm requested but OPENAI_API_KEY is not set; "
                      "falling back to offline heuristics.[/yellow]")
        use_llm = False

    console.print(Panel(f"Running {len(scenarios)} scenario(s) "
                        f"({'real LLM' if use_llm else 'offline heuristics'})",
                        title=f"Agent Trace v{__version__}"))
    results = run_suite(scenarios, use_llm=use_llm, verbose=args.verbose)

    table = Table(title="Benchmark scorecard")
    table.add_column("scenario")
    table.add_column("result", justify="center")
    table.add_column("det grades", justify="center")
    table.add_column("judge", justify="center")
    table.add_column("tools", justify="right")
    table.add_column("latency ms", justify="right")
    for r in results:
        det_ok = sum(1 for g in r.deterministic if g.passed)
        det_tot = len(r.deterministic)
        judge = f"{r.judge.score:.2f}" if r.judge else "-"
        table.add_row(r.scenario_id,
                      "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
                      f"{det_ok}/{det_tot}", judge,
                      str(r.tool_call_count), f"{r.total_latency_ms:.0f}")
    console.print(table)

    summary = summarize(results)
    console.print(Panel(
        f"pass rate {summary['passed']}/{summary['total']} "
        f"({summary['pass_rate']:.0%})  |  "
        f"red-team {summary['redteam_passed']}/{summary['redteam_total']}  |  "
        f"avg tools {summary['avg_tool_calls']:.1f}  |  "
        f"avg latency {summary['avg_latency_ms']:.0f} ms",
        title="Summary"))

    for r in results:
        if not r.passed and r.failure_trace:
            console.print(Panel(r.failure_trace, title=f"FAIL: {r.scenario_id}",
                                style="red"))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary,
                   "results": [r.model_dump() for r in results]}
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Wrote JSON report to {out}[/green]")

    if args.traces_dir:
        tdir = Path(args.traces_dir)
        tdir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.trajectory:
                (tdir / f"{r.scenario_id}.json").write_text(
                    json.dumps(r.trajectory.model_dump(), indent=2),
                    encoding="utf-8")
        console.print(f"[green]Wrote trajectories to {tdir}/[/green]")

    return 0 if summary["failed"] == 0 else 1


def cmd_serve(_args: argparse.Namespace) -> int:
    from agent_trace.mock_crm.server import mcp
    console.print("[green]Starting mock CRM FastMCP server...[/green]")
    mcp.run()
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import uvicorn
    console.print(f"[green]Agent Trace UI on http://127.0.0.1:{args.port}[/green] "
                  "(Ctrl+C to stop)")
    uvicorn.run("agent_trace.ui.server:app", host="127.0.0.1", port=args.port,
                reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-trace",
                                description="Evaluate tool-calling agents (Agent Trace)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List builtin scenarios").set_defaults(func=cmd_list)
    pr = sub.add_parser("run", help="Run evaluation suite")
    pr.add_argument("--scenario", action="append", default=[],
                    help="Scenario id to run (repeatable; default: all)")
    pr.add_argument("--llm", action="store_true",
                    help="Use real OpenAI for simulator/agent/judge (needs OPENAI_API_KEY)")
    pr.add_argument("--verbose", action="store_true")
    pr.add_argument("--output", default="",
                    help="Write JSON report to this path")
    pr.add_argument("--traces-dir", default="",
                    help="Write per-scenario trajectory JSON files here")
    pr.set_defaults(func=cmd_run)
    sub.add_parser("serve", help="Run the mock CRM FastMCP server").set_defaults(func=cmd_serve)
    pu = sub.add_parser("ui", help="Launch the minimal test UI in a browser")
    pu.add_argument("--port", type=int, default=8000)
    pu.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
