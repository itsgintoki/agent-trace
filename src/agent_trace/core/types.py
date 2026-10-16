"""Core Pydantic types: messages, trajectories, scenarios, results."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class TrajectoryEvent(BaseModel):
    turn: int
    speaker: Literal["user", "agent", "tool"]
    content: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    latency_ms: float = 0.0


class Trajectory(BaseModel):
    scenario_id: str
    events: list[TrajectoryEvent] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    total_latency_ms: float = 0.0

    def transcript(self) -> str:
        lines: list[str] = []
        for e in self.events:
            lines.append(f"[t{e.turn} {e.speaker}] {e.content}")
            for tc in e.tool_calls:
                status = f"ERROR {tc.error}" if tc.error else "ok"
                lines.append(f"  - tool {tc.name} {tc.arguments} -> {status}")
        return "\n".join(lines)


class ExpectedState(BaseModel):
    """Deterministic assertions for Day 3 graders."""
    must_have_contacts: list[dict[str, Any]] = Field(default_factory=list)
    must_not_have_emails: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_deal_stages: dict[str, str] = Field(default_factory=dict)  # deal key -> stage
    min_tool_calls: int = 0


class Scenario(BaseModel):
    id: str
    title: str
    description: str = ""
    persona: str = ""
    goal: str = ""
    initial_message: str
    followups: list[str] = Field(default_factory=list)
    max_turns: int = 6
    seed_contacts: list[dict[str, Any]] = Field(default_factory=list)
    adversarial: bool = False
    injection_payload: str = ""
    expected: ExpectedState = Field(default_factory=ExpectedState)
    judge_rubric: list[str] = Field(default_factory=list)


class GradeResult(BaseModel):
    name: str
    passed: bool
    details: str = ""
    score: float = 1.0  # 0..1


class JudgeVerdict(BaseModel):
    """Pydantic structured output for the LLM-as-a-judge (Day 4)."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    tone_ok: bool = True
    injection_resisted: bool = True


class ScenarioResult(BaseModel):
    scenario_id: str
    title: str
    deterministic: list[GradeResult] = Field(default_factory=list)
    judge: JudgeVerdict | None = None
    passed: bool = False
    tool_call_count: int = 0
    total_latency_ms: float = 0.0
    trajectory: Trajectory | None = None
    failure_trace: str = ""
