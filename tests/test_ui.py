from fastapi.testclient import TestClient

from agent_trace.ui.server import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["ok"] is True


def test_scenarios_lists_six():
    data = client.get("/api/scenarios").json()
    assert len(data) == 6
    assert {s["id"] for s in data} >= {"lookup-query", "injection-export"}


def test_run_single_scenario():
    out = client.post("/api/run", json={"scenario_ids": ["lookup-query"]}).json()
    assert out["summary"]["passed"] == 1
    assert out["results"][0]["passed"] is True


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200 and "<title>Agent Trace" in r.text
