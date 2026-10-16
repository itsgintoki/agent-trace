from agent_trace.runner.loop import run_scenario
from agent_trace.runner.suite import run_suite
from agent_trace.scenarios.builtin import builtin_scenarios


def test_happy_paths_pass_offline():
    for sid in ("create-contact-basic", "update-contact-status",
                "deal-lifecycle", "lookup-query"):
        sc = next(s for s in builtin_scenarios() if s.id == sid)
        res = run_scenario(sc, use_llm=False)
        assert res.passed, f"{sid} failed:\n{res.failure_trace}"
        assert res.trajectory is not None and res.trajectory.events


def test_redteam_refusals_pass_offline():
    for sid in ("injection-export", "injection-wipe"):
        sc = next(s for s in builtin_scenarios() if s.id == sid)
        res = run_scenario(sc, use_llm=False)
        assert res.passed, f"{sid} failed:\n{res.failure_trace}"
        assert res.judge is not None and res.judge.injection_resisted


def test_full_suite_passes():
    results = run_suite(builtin_scenarios(), use_llm=False)
    assert len(results) == 6
    assert all(r.passed for r in results), "\n".join(
        r.failure_trace for r in results if not r.passed)
