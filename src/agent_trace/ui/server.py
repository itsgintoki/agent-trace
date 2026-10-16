"""Minimal test UI: FastAPI + single HTML file, no npm required.

Run with:  agent-trace ui --port 8000
Then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent_trace.mock_crm.store import CRMStore
from agent_trace.runner.loop import run_scenario
from agent_trace.runner.suite import summarize
from agent_trace.scenarios.builtin import builtin_scenarios

app = FastAPI(title="Agent Trace UI")
_INDEX = Path(__file__).with_name("index.html")


class RunRequest(BaseModel):
    scenario_ids: list[str] | None = None
    use_llm: bool = False


@app.get("/api/health")
def health() -> dict:
    return {"ok": True,
            "llm_available": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/scenarios")
def list_scenarios() -> list[dict]:
    return [s.model_dump() for s in builtin_scenarios()]


@app.post("/api/run")
def run_selected(body: RunRequest) -> JSONResponse:
    scenarios = builtin_scenarios()
    if body.scenario_ids:
        wanted = set(body.scenario_ids)
        scenarios = [s for s in scenarios if s.id in wanted]
    use_llm = body.use_llm and bool(os.getenv("OPENAI_API_KEY"))
    results = [run_scenario(s, store=CRMStore(), use_llm=use_llm)
               for s in scenarios]
    return JSONResponse({"summary": summarize(results),
                         "results": [r.model_dump() for r in results]})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX, media_type="text/html")
