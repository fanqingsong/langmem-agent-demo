from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langmem import create_memory_store_manager

from src.memory.common import render_items

INSTRUCTIONS = """Maintain the single WorkingContext for this thread.

Update it to match the active task: the current goal, constraints, open questions,
and decisions made in this conversation. Drop goals from earlier threads.
Do not copy durable user facts (name, preferences, background knowledge) into this
profile; those belong in semantic memory.
Revise the existing profile instead of creating another one.
"""


class WorkingContext(BaseModel):
    """Active context for the current conversation thread."""

    goal: str = Field(
        ...,
        description="What the user is trying to accomplish in this thread.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints that apply only to the current task.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions still unresolved in this thread.",
    )
    recent_decisions: list[str] = Field(
        default_factory=list,
        description="Choices made so far in this thread.",
    )


def namespace(user_id: str, thread_id: str) -> tuple[str, ...]:
    if not thread_id:
        raise ValueError("working memory is scoped to a thread")
    return ("working", user_id, thread_id)


def manager(model, store: BaseStore):
    return create_memory_store_manager(
        model,
        schemas=[WorkingContext],
        instructions=INSTRUCTIONS,
        namespace=("working", "{user_id}", "{thread_id}"),
        default=WorkingContext(
            goal="(none yet)",
            constraints=[],
            open_questions=[],
            recent_decisions=[],
        ),
        enable_inserts=False,
        enable_deletes=False,
        store=store,
    )


def render(store: BaseStore, user_id: str, thread_id: str) -> str:
    items = store.search(namespace(user_id, thread_id), limit=5)
    return render_items(items)
