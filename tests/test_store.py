import pytest

from agent_trace.mock_crm.store import CRMError, CRMStore


def test_create_and_get_contact():
    s = CRMStore()
    c = s.create_contact(name="Alice", email="alice@example.com")
    assert c["id"].startswith("c_")
    assert s.get_contact(email="alice@example.com")["name"] == "Alice"
    assert s.get_contact(contact_id=c["id"])["email"] == "alice@example.com"


def test_duplicate_email_rejected():
    s = CRMStore()
    s.create_contact(name="Alice", email="alice@example.com")
    with pytest.raises(CRMError):
        s.create_contact(name="Alice2", email="alice@example.com")


def test_invalid_email_and_status():
    s = CRMStore()
    with pytest.raises(CRMError):
        s.create_contact(name="X", email="not-an-email")
    with pytest.raises(CRMError):
        s.create_contact(name="X", email="x@example.com", status="bogus")


def test_update_and_note():
    s = CRMStore()
    c = s.create_contact(name="Bob", email="bob@example.com")
    s.update_contact(c["id"], status="vip")
    assert s.get_contact(contact_id=c["id"])["status"] == "vip"
    s.add_note(c["id"], "renewed for another year")
    assert "renewed" in " ".join(s.get_contact(contact_id=c["id"])["notes"])


def test_deal_lifecycle():
    s = CRMStore()
    c = s.create_contact(name="Carol", email="carol@example.com")
    d = s.create_deal(contact_id=c["id"], amount=5000)
    assert d["stage"] == "prospecting"
    s.update_deal_stage(d["id"], "closed-won")
    assert s.list_deals(contact_id=c["id"])[0]["stage"] == "closed-won"


def test_delete_cascades_and_reset():
    s = CRMStore()
    c = s.create_contact(name="Zed", email="zed@example.com")
    s.create_deal(contact_id=c["id"], amount=10)
    s.delete_contact(c["id"])
    assert s.list_contacts() == [] and s.list_deals() == []
    s.reset(seed=[{"name": "Amy", "email": "amy@example.com"}])
    assert len(s.list_contacts()) == 1


def test_dangerous_tools_logged():
    s = CRMStore()
    s.create_contact(name="Dave", email="dave@example.com")
    snap = s.export_all_data()
    assert len(snap["contacts"]) == 1
    out = s.wipe_database()
    assert out["wiped_contacts"] == 1 and s.list_contacts() == []
