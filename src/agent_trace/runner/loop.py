"""Multi-turn execution loop (Day 2): user simulator <-> target agent + tools."""

from __future__ import annotations

import time
import uuid

from agent_trace.agent.target_agent import TargetAgent
from agent_trace.agent.tool_adapter import CRMToolAdapter
from agent_trace.core.trajectory import TrajectoryLogger
from agent_trace.core.types import Scenario, ToolCallRecord
from agent_trace.grading.deterministic import run_deterministic_grades
from agent_trace.grading.judge import judge_trajectory
from agent_trace.core.types import ScenarioResult
from agent_trace.mock_crm.store import CRMStore
from agent_trace.sim.user_simulator import UserSimulator


def run_scenario(scenario: Scenario, store: CRMStore | None = None,
                 use_llm: bool = False, verbose: bool = False) -> ScenarioResult:
    store = store or CRMStore()
    store.reset(seed=scenario.seed_contacts)
    adapter = CRMToolAdapter(store)
    agent = TargetAgent(adapter, use_llm=use_llm)
    sim = UserSimulator(scenario, use_llm=use_llm)
    logger = TrajectoryLogger(scenario.id)

    history: list[dict] = []  # OpenAI-style history for the agent
    transcript_tail = ""

    for turn in range(scenario.max_turns):
        # --- user turn ---
        sim_turn = sim.next_message(transcript_tail)
        if sim_turn.done:
            break
        t0 = time.perf_counter()
        logger.log_message(turn, "user", sim_turn.message,
                           latency_ms=(time.perf_counter() - t0) * 1000.0)
        history.append({"role": "user", "content": sim_turn.message})
        transcript_tail += f"\nUser: {sim_turn.message}"
        if verbose:
            print(f"[user] {sim_turn.message}")

        # --- agent turn ---
        t0 = time.perf_counter()
        action = agent.act(sim_turn.message, history)
        agent_latency = (time.perf_counter() - t0) * 1000.0

        # --- tool execution ---
        records: list[ToolCallRecord] = []
        tool_summaries: list[str] = []
        for name, args in action.tool_calls:
            tc0 = time.perf_counter()
            try:
                result = adapter.call(name, args)
                err = None
            except Exception as e:  # noqa: BLE001 — tool errors are part of the trace
                result, err = None, str(e)
            rec = ToolCallRecord(
                id=uuid.uuid4().hex[:8], name=name, arguments=args,
                result=result, error=err,
                latency_ms=(time.perf_counter() - tc0) * 1000.0)
            records.append(rec)
            tool_summaries.append(f"{name}({args}) -> {'ERROR '+err if err else 'ok'}")
            history.append({"role": "tool", "content": f"{name} -> {result if not err else 'ERROR: '+err}"})

        logger.log_message(turn, "agent", action.reply, tool_calls=records,
                           latency_ms=agent_latency)
        history.append({"role": "assistant", "content": action.reply})
        transcript_tail += f"\nAgent: {action.reply}"
        if verbose:
            print(f"[agent] {action.reply}")
            for s in tool_summaries:
                print(f"  [tool] {s}")

        # Early stop: simulator exhausted AND agent made no tool calls this turn.
        # (The loop also ends naturally when sim returns [DONE].)
        if not records and turn >= len(scenario.followups):
            # Give the agent one more user turn to finish before closing.
            pass

    final_state = store.snapshot()
    logger.set_final_state(final_state)
    traj = logger.trajectory
    tool_names = [tc.name for tc in traj.tool_calls]

    det = run_deterministic_grades(final_state, tool_names, scenario.expected)
    det_passed = all(g.passed for g in det)
    verdict = judge_trajectory(scenario, traj, det_passed, use_llm=use_llm)
    passed = det_passed and verdict.passed

    failure_trace = ""
    if not passed:
        lines = [f"Scenario {scenario.id} FAILED"]
        for g in det:
            if not g.passed:
                lines.append(f"  [deterministic:{g.name}] {g.details}")
        if not verdict.passed:
            lines.append(f"  [judge] score={verdict.score} reasons={verdict.reasons} "
                         f"violations={verdict.policy_violations}")
        lines.append("  --- transcript ---")
        lines.append(traj.transcript())
        failure_trace = "\n".join(lines)

    return ScenarioResult(
        scenario_id=scenario.id, title=scenario.title,
        deterministic=det, judge=verdict, passed=passed,
        tool_call_count=len(tool_names),
        total_latency_ms=traj.total_latency_ms,
        trajectory=traj, failure_trace=failure_trace)
