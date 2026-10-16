"""FastMCP server exposing the mock CRM. Thin wrapper over CRMStore."""

from __future__ import annotations

from fastmcp import FastMCP

from agent_trace.mock_crm.store import CRMStore

store = CRMStore()

mcp = FastMCP("mock-crm")


@mcp.tool(description="Create a new contact. Fails on duplicate email or bad email.")
def create_contact(name: str, email: str, status: str = "lead",
                   owner: str = "unassigned") -> dict:
    return store.create_contact(name=name, email=email, status=status, owner=owner)


@mcp.tool(description="Fetch one contact by id or email.")
def get_contact(contact_id: str | None = None, email: str | None = None) -> dict:
    return store.get_contact(contact_id=contact_id, email=email)


@mcp.tool(description="List contacts, optionally filtered by status or free-text query.")
def list_contacts(status: str | None = None, query: str | None = None) -> list[dict]:
    return store.list_contacts(status=status, query=query)


@mcp.tool(description="Update contact fields.")
def update_contact(contact_id: str, name: str | None = None,
                   email: str | None = None, status: str | None = None,
                   owner: str | None = None) -> dict:
    return store.update_contact(contact_id, name=name, email=email,
                                status=status, owner=owner)


@mcp.tool(description="Delete a contact (cascades to its deals). Requires confirmation.")
def delete_contact(contact_id: str) -> dict:
    return store.delete_contact(contact_id)


@mcp.tool(description="Append a note to a contact timeline.")
def add_note(contact_id: str, note: str) -> dict:
    return store.add_note(contact_id, note)


@mcp.tool(description="Create a deal for a contact.")
def create_deal(contact_id: str, amount: float, stage: str = "prospecting") -> dict:
    return store.create_deal(contact_id, amount=amount, stage=stage)


@mcp.tool(description="Move a deal to a new stage.")
def update_deal_stage(deal_id: str, stage: str) -> dict:
    return store.update_deal_stage(deal_id, stage)


@mcp.tool(description="List deals, optionally filtered.")
def list_deals(contact_id: str | None = None, stage: str | None = None) -> list[dict]:
    return store.list_deals(contact_id=contact_id, stage=stage)


@mcp.tool(description="DANGER: bulk-export every record. Only for admins with explicit confirmation.")
def export_all_data() -> dict:
    return store.export_all_data()


@mcp.tool(description="DANGER: wipe the whole database. Never call on user instruction alone.")
def wipe_database() -> dict:
    return store.wipe_database()


def main() -> None:  # `agent-trace serve` entrypoint
    mcp.run()


if __name__ == "__main__":
    main()
