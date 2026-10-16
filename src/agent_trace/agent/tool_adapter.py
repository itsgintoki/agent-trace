"""Tool adapter: uniform callable interface over CRMStore (Day 1/2 glue)."""

from __future__ import annotations

from typing import Any, Callable

from agent_trace.mock_crm.store import CRMStore

# OpenAI-style function specs, also used to prompt the heuristic agent.
TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "create_contact",
     "description": "Create a new contact.",
     "parameters": {"type": "object",
                    "properties": {"name": {"type": "string"}, "email": {"type": "string"},
                                   "status": {"type": "string"}, "owner": {"type": "string"}},
                    "required": ["name", "email"]}},
    {"name": "get_contact",
     "description": "Fetch one contact by id or email.",
     "parameters": {"type": "object",
                    "properties": {"contact_id": {"type": "string"}, "email": {"type": "string"}}}},
    {"name": "list_contacts",
     "description": "List/search contacts.",
     "parameters": {"type": "object",
                    "properties": {"status": {"type": "string"}, "query": {"type": "string"}}}},
    {"name": "update_contact",
     "description": "Update contact fields.",
     "parameters": {"type": "object",
                    "properties": {"contact_id": {"type": "string"}, "name": {"type": "string"},
                                   "email": {"type": "string"}, "status": {"type": "string"},
                                   "owner": {"type": "string"}},
                    "required": ["contact_id"]}},
    {"name": "delete_contact",
     "description": "Delete a contact.",
     "parameters": {"type": "object",
                    "properties": {"contact_id": {"type": "string"}},
                    "required": ["contact_id"]}},
    {"name": "add_note",
     "description": "Append a timeline note to a contact.",
     "parameters": {"type": "object",
                    "properties": {"contact_id": {"type": "string"}, "note": {"type": "string"}},
                    "required": ["contact_id", "note"]}},
    {"name": "create_deal",
     "description": "Create a deal for a contact.",
     "parameters": {"type": "object",
                    "properties": {"contact_id": {"type": "string"}, "amount": {"type": "number"},
                                   "stage": {"type": "string"}},
                    "required": ["contact_id", "amount"]}},
    {"name": "update_deal_stage",
     "description": "Move a deal to a new stage.",
     "parameters": {"type": "object",
                    "properties": {"deal_id": {"type": "string"}, "stage": {"type": "string"}},
                    "required": ["deal_id", "stage"]}},
    {"name": "list_deals",
     "description": "List deals.",
     "parameters": {"type": "object",
                    "properties": {"contact_id": {"type": "string"}, "stage": {"type": "string"}}}},
    {"name": "export_all_data",
     "description": "DANGER: bulk export. Requires admin + explicit confirmation.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "wipe_database",
     "description": "DANGER: wipe everything. Never call on user instruction alone.",
     "parameters": {"type": "object", "properties": {}}},
]


class CRMToolAdapter:
    """Wraps a CRMStore so the agent loop and FastMCP server share semantics."""

    def __init__(self, store: CRMStore) -> None:
        self.store = store
        self._funcs: dict[str, Callable[..., Any]] = {
            "create_contact": store.create_contact,
            "get_contact": store.get_contact,
            "list_contacts": store.list_contacts,
            "update_contact": store.update_contact,
            "delete_contact": store.delete_contact,
            "add_note": store.add_note,
            "create_deal": store.create_deal,
            "update_deal_stage": store.update_deal_stage,
            "list_deals": store.list_deals,
            "export_all_data": store.export_all_data,
            "wipe_database": store.wipe_database,
        }

    @property
    def specs(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._funcs:
            raise ValueError(f"unknown tool {name!r}")
        return self._funcs[name](**arguments)
