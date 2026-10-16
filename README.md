# Agent Trace

AI-driven evaluation harness for tool-calling LLM agents.

Automated CLI that runs multi-turn conversation loops between a simulated user
and a target tool-calling agent, captures the full execution trajectory, and grades
the outcome with deterministic state checks + LLM-as-a-judge.

## Stack

- Python, FastMCP (mock CRM tools), OpenAI SDK (optional real LLM), Pydantic, Rich

## Layout

- `src/agent_trace/mock_crm/` — Day 1: in-memory CRM store + FastMCP server
- `src/agent_trace/sim/` + `src/agent_trace/agent/` + `src/agent_trace/core/trajectory.py` — Day 2: multi-turn loop + trajectory logger
- `src/agent_trace/grading/deterministic.py` — Day 3: state-based graders
- `src/agent_trace/grading/judge.py` + red-team scenarios — Day 4: LLM-as-a-judge (Pydantic structured outputs) + prompt-injection tests
- `src/agent_trace/cli/` + `src/agent_trace/runner/suite.py` — Day 5: CLI + Rich benchmark scorecards

## Quickstart (offline, no API key)

```bash
uv sync --group dev
uv run agent-trace list
uv run agent-trace run
uv run pytest
```

## Test UI (no npm)

```bash
uv run agent-trace ui --port 8000
# open http://127.0.0.1:8000
```

Single HTML page + FastAPI (`src/agent_trace/ui/`): tick scenarios, run,
see the scorecard, click a row for deterministic grades, judge verdict, tool
calls, and full transcript. API: `GET /api/scenarios`, `POST /api/run`,
`GET /api/health`.

## Real-LLM mode

```bash
export OPENAI_API_KEY=...
export AGENT_TRACE_MODEL=gpt-4o-mini  # optional
uv run agent-trace run --llm --verbose
```

Real mode affects the user simulator (persona paraphrase), the target agent
(OpenAI function calling with a policy gate that still blocks
`export_all_data`/`wipe_database`), and the judge (structured outputs validated
into `JudgeVerdict`).

## Serve the mock CRM over MCP

```bash
uv run agent-trace serve
```
