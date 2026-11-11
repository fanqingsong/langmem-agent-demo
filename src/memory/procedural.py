from typing import Any

from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore, Item
from langmem import create_memory_store_manager

from src.memory.common import payload, render_value

INSTRUCTIONS = """Extract durable rules for how the assistant should behave.

Keep explicit instructions about response style, teaching method, format, and
task procedure. Update a rule when the user revises it, and delete a rule when
the user retracts it.
Skip the user's identity, preferences about their own environment, the current
task, and one-off episodes. Those belong in other memories.
Prefer one general rule over several overlapping ones.
"""


class Procedure(BaseModel):
    """A durable rule for how the assistant should respond."""

    instruction: str = Field(
        ...,
        description="The behavior to follow, written as a direct instruction.",
    )
    applies_when: str = Field(
        ...,
        description="When this rule applies. Use 'always' for a global rule.",
    )


def namespace(user_id: str) -> tuple[str, ...]:
    return ("procedural", user_id)


def manager(model, store: BaseStore):
    return create_memory_store_manager(
        model,
        schemas=[Procedure],
        instructions=INSTRUCTIONS,
        namespace=("procedural", "{user_id}"),
        enable_inserts=True,
        enable_deletes=True,
        store=store,
    )


def _meta_rule(value: Any) -> bool:
    if isinstance(value, dict):
        text = value.get("instruction")
    else:
        text = value
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return "do not extract" in lowered or "no durable rule" in lowered


def list_items(store: BaseStore, user_id: str) -> list[Item]:
    scope = namespace(user_id)
    # Earlier builds stored one rewritten prompt under this key.
    if store.get(scope, "instructions") is not None:
        store.delete(scope, "instructions")
    items = store.search(scope, limit=20)
    kept = []
    for item in items:
        if _meta_rule(payload(item)):
            store.delete(scope, item.key)
            continue
        kept.append(item)
    return kept


def render(store: BaseStore, user_id: str) -> str:
    rendered = []
    for index, item in enumerate(list_items(store, user_id), start=1):
        procedure: Any = payload(item)
        if not isinstance(procedure, dict):
            rendered.append(f"Rule {index}:\n{render_value(procedure)}")
            continue
        rendered.append(
            f"Rule {index}: {procedure.get('instruction', '')} "
            f"(when: {procedure.get('applies_when', 'always')})"
        )
    return "\n".join(rendered) or "(empty)"
