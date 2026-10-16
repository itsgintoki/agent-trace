"""Test-suite orchestration over all scenarios (Day 5)."""

from __future__ import annotations

from agent_trace.core.types import Scenario
from agent_trace.core.types import ScenarioResult
from agent_trace.mock_crm.store import CRMStore
from agent_trace.runner.loop import run_scenario
from agent_trace.scenarios.builtin import builtin_scenarios


def load_suite(ids: list[str] | None = None) -> list[Scenario]:
    all_s = builtin_scenarios()
    if not ids:
        return all_s
    wanted = set(ids)
    out = [s for s in all_s if s.id in wanted]
    unknown = wanted - {s.id for s in out}
    if unknown:
        raise ValueError(f"unknown scenario(s): {sorted(unknown)}")
    return out


def run_suite(scenarios: list[Scenario], use_llm: bool = False,
              verbose: bool = False) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for sc in scenarios:
        results.append(run_scenario(sc, store=CRMStore(),
                                    use_llm=use_llm, verbose=verbose))
    return results


def summarize(results: list[ScenarioResult]) -> dict:
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    adv = [r for r in results if r.trajectory and "RED TEAM" in (r.title or "").upper()]
    return {
        "total": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": (passed / n) if n else 0.0,
        "avg_tool_calls": sum(r.tool_call_count for r in results) / n if n else 0.0,
        "avg_latency_ms": sum(r.total_latency_ms for r in results) / n if n else 0.0,
        "redteam_passed": sum(1 for r in adv if r.passed),
        "redteam_total": len(adv),
    }
