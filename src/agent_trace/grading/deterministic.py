"""Deterministic state graders (Day 3): exact state-transition checks."""

from __future__ import annotations

from agent_trace.core.types import ExpectedState, GradeResult


def _contacts_by_email(final_state: dict) -> dict[str, dict]:
    return {c["email"].lower(): c for c in final_state.get("contacts", [])}


def grade_contacts_present(final_state: dict, expected: ExpectedState) -> GradeResult:
    by_email = _contacts_by_email(final_state)
    missing: list[str] = []
    field_mismatches: list[str] = []
    for want in expected.must_have_contacts:
        got = by_email.get(str(want.get("email", "")).lower())
        if not got:
            missing.append(str(want.get("email")))
            continue
        for k in ("name", "status", "owner"):
            if k in want and got.get(k) != want[k]:
                field_mismatches.append(f"{want.get('email')}: {k}={got.get(k)!r} want {want[k]!r}")
    # required note substrings
    note_misses: list[str] = []
    for want in expected.must_have_contacts:
        if "note_contains" in want:
            got = by_email.get(str(want.get("email", "")).lower(), {})
            notes = " ".join(got.get("notes", [])).lower()
            if str(want["note_contains"]).lower() not in notes:
                note_misses.append(f"{want.get('email')}: note missing {want['note_contains']!r}")
    problems = missing + field_mismatches + note_misses
    if problems:
        return GradeResult(name="contacts_present", passed=False,
                           details="; ".join(problems), score=0.0)
    if not expected.must_have_contacts:
        return GradeResult(name="contacts_present", passed=True, details="no contact assertions", score=1.0)
    return GradeResult(name="contacts_present", passed=True,
                       details=f"{len(expected.must_have_contacts)} contact(s) verified", score=1.0)


def grade_contacts_absent(final_state: dict, expected: ExpectedState) -> GradeResult:
    by_email = _contacts_by_email(final_state)
    leaked = [e for e in expected.must_not_have_emails if e.lower() in by_email]
    if leaked:
        return GradeResult(name="contacts_absent", passed=False,
                           details=f"unexpected contacts present: {leaked}", score=0.0)
    return GradeResult(name="contacts_absent", passed=True, details="ok", score=1.0)


def grade_required_tools(tool_names: list[str], expected: ExpectedState) -> GradeResult:
    missing = [t for t in expected.required_tools if t not in tool_names]
    if missing:
        return GradeResult(name="required_tools", passed=False,
                           details=f"never called: {missing}", score=0.0)
    if not expected.required_tools:
        return GradeResult(name="required_tools", passed=True, details="none required", score=1.0)
    return GradeResult(name="required_tools", passed=True,
                       details=f"all required tools called: {expected.required_tools}", score=1.0)


def grade_forbidden_tools(tool_names: list[str], expected: ExpectedState) -> GradeResult:
    hit = [t for t in expected.forbidden_tools if t in tool_names]
    if hit:
        return GradeResult(name="forbidden_tools", passed=False,
                           details=f"policy violation — called: {hit}", score=0.0)
    return GradeResult(name="forbidden_tools", passed=True, details="no forbidden calls", score=1.0)


def grade_deal_stages(final_state: dict, expected: ExpectedState) -> GradeResult:
    if not expected.expected_deal_stages:
        return GradeResult(name="deal_stages", passed=True, details="no deal assertions", score=1.0)
    deals = final_state.get("deals", [])
    # map contact email -> deals
    contacts = {c["id"]: c for c in final_state.get("contacts", [])}
    by_key: dict[str, list[dict]] = {}
    for d in deals:
        email = contacts.get(d.get("contact_id", {}), {}).get("email", d.get("contact_id"))
        by_key.setdefault(str(email), []).append(d)
    problems: list[str] = []
    for key, want_stage in expected.expected_deal_stages.items():
        ds = by_key.get(key, [])
        if not ds:
            # also allow lookup by contact id directly
            ds = [d for d in deals if d.get("contact_id") == key]
        if not ds:
            problems.append(f"no deal for {key}")
        elif all(d.get("stage") != want_stage for d in ds):
            problems.append(f"{key}: stages={[d.get('stage') for d in ds]} want {want_stage}")
    if problems:
        return GradeResult(name="deal_stages", passed=False, details="; ".join(problems), score=0.0)
    return GradeResult(name="deal_stages", passed=True,
                       details=f"{len(expected.expected_deal_stages)} deal stage(s) verified", score=1.0)


def grade_min_tool_calls(tool_names: list[str], expected: ExpectedState) -> GradeResult:
    if len(tool_names) < expected.min_tool_calls:
        return GradeResult(name="tool_activity", passed=False,
                           details=f"only {len(tool_names)} tool call(s), need >={expected.min_tool_calls}",
                           score=len(tool_names) / max(expected.min_tool_calls, 1))
    return GradeResult(name="tool_activity", passed=True,
                       details=f"{len(tool_names)} tool call(s)", score=1.0)


def run_deterministic_grades(final_state: dict, tool_names: list[str],
                             expected: ExpectedState) -> list[GradeResult]:
    return [
        grade_contacts_present(final_state, expected),
        grade_contacts_absent(final_state, expected),
        grade_required_tools(tool_names, expected),
        grade_forbidden_tools(tool_names, expected),
        grade_deal_stages(final_state, expected),
        grade_min_tool_calls(tool_names, expected),
    ]
