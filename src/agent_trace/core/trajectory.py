"""Trajectory logger: records every message, tool call, and state mutation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_trace.core.types import ToolCallRecord, Trajectory, TrajectoryEvent


class TrajectoryLogger:
    def __init__(self, scenario_id: str) -> None:
        self._traj = Trajectory(scenario_id=scenario_id)
        self._t0 = time.perf_counter()

    def log_message(self, turn: int, speaker: str, content: str,
                    tool_calls: list[ToolCallRecord] | None = None,
                    latency_ms: float = 0.0) -> None:
        assert speaker in ("user", "agent", "tool")
        self._traj.events.append(TrajectoryEvent(
            turn=turn, speaker=speaker, content=content,  # type: ignore[arg-type]
            tool_calls=tool_calls or [], latency_ms=latency_ms))
        for tc in tool_calls or []:
            self._traj.tool_calls.append(tc)

    def set_final_state(self, state: dict[str, Any]) -> None:
        self._traj.final_state = state
        self._traj.total_latency_ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def trajectory(self) -> Trajectory:
        return self._traj

    def save_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(self._traj.model_dump_json() + "\n")
        return p

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._traj.model_dump(), indent=2), encoding="utf-8")
        return p
