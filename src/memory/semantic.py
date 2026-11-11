from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langmem import create_memory_store_manager

from src.memory.common import render_items

INSTRUCTIONS = """Extract durable facts about the user.

Keep identity, preferences, and stable background knowledge.
Skip the current task, transient questions, and instructions about how the
assistant should behave. Update a fact when the user corrects it, and delete
facts that are contradicted.
"""


class SemanticFact(BaseModel):
    """A durable fact about the user or their world."""

    content: str = Field(
        ...,
        description="The fact itself, stated so it can be recalled later.",
    )
    category: str = Field(
        ...,
        description="One of: identity, preference, knowledge, other.",
    )


def namespace(user_id: str) -> tuple[str, ...]:
    return ("semantic", user_id)


def manager(model, store: BaseStore):
    return create_memory_store_manager(
        model,
        schemas=[SemanticFact],
        instructions=INSTRUCTIONS,
        namespace=("semantic", "{user_id}"),
        enable_inserts=True,
        enable_deletes=True,
        store=store,
    )


def render(store: BaseStore, user_id: str, query: str | None = None, limit: int = 5) -> str:
    if query:
        items = store.search(namespace(user_id), query=query, limit=limit)
    else:
        items = store.search(namespace(user_id), limit=limit)
    return render_items(items)
