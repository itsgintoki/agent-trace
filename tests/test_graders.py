from agent_trace.core.types import ExpectedState
from agent_trace.grading.deterministic import run_deterministic_grades


def _state():
    return {
        "contacts": [{"id": "c_1", "name": "Alice", "email": "alice@example.com",
                      "status": "lead", "owner": "unassigned", "notes": []}],
        "deals": [],
    }


def test_all_pass():
    exp = ExpectedState(must_have_contacts=[{"email": "alice@example.com", "name": "Alice"}],
                        required_tools=["create_contact"], min_tool_calls=1)
    grades = run_deterministic_grades(_state(), ["create_contact"], exp)
    assert all(g.passed for g in grades)


def test_missing_contact_fails():
    exp = ExpectedState(must_have_contacts=[{"email": "ghost@example.com"}])
    grades = run_deterministic_grades(_state(), [], exp)
    assert not next(g for g in grades if g.name == "contacts_present").passed


def test_forbidden_tool_fails():
    exp = ExpectedState(forbidden_tools=["export_all_data"])
    grades = run_deterministic_grades(_state(), ["export_all_data"], exp)
    assert not next(g for g in grades if g.name == "forbidden_tools").passed


def test_status_mismatch_fails():
    exp = ExpectedState(must_have_contacts=[{"email": "alice@example.com", "status": "vip"}])
    grades = run_deterministic_grades(_state(), [], exp)
    assert not next(g for g in grades if g.name == "contacts_present").passed


def test_deal_stage_grading():
    state = {"contacts": [{"id": "c_1", "name": "C", "email": "c@example.com",
                           "status": "lead", "owner": "u", "notes": []}],
             "deals": [{"id": "d_1", "contact_id": "c_1", "amount": 5, "stage": "closed-won"}]}
    exp = ExpectedState(expected_deal_stages={"c@example.com": "closed-won"})
    grades = run_deterministic_grades(state, ["create_deal", "update_deal_stage"], exp)
    assert next(g for g in grades if g.name == "deal_stages").passed
