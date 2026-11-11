from typing import Any

from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langmem import create_memory_store_manager

from src.memory.common import payload, render_items, render_value

INSTRUCTIONS = """Extract episodes only when an interaction shows a reusable way
of handling a situation.

Each episode needs the situation, the reasoning, the action, and why the outcome
was good or bad. Skip greetings, bare preference statements, and unfinished
exchanges that do not yet show whether the approach worked.
Prefer one dense episode over several overlapping ones.
"""


class Episode(BaseModel):
    """Write the episode from the perspective of the agent within it.

    Use the benefit of hindsight to record the memory, saving the agent's key
    internal thought process so it can learn over time.
    """

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description=(
            "Internal reasoning process and observations of the agent in the episode "
            'that let it arrive at the correct action and result. "I ..."'
        ),
    )
    action: str = Field(
        ...,
        description=(
            "What was done, how, and in what format. Include whatever is salient "
            "to the success of the action. I ..."
        ),
    )
    result: str = Field(
        ...,
        description=(
            "Outcome and retrospective. What did you do well? What could you do "
            "better next time? I ..."
        ),
    )


def namespace(user_id: str) -> tuple[str, ...]:
    return ("episodic", user_id)


def manager(model, store: BaseStore):
    return create_memory_store_manager(
        model,
        schemas=[Episode],
        instructions=INSTRUCTIONS,
        namespace=("episodic", "{user_id}"),
        enable_inserts=True,
        enable_deletes=False,
        store=store,
    )


def _one(index: int, episode: Any) -> str:
    if not isinstance(episode, dict):
        return f"Episode {index}:\n{render_value(episode)}"
    return "\n".join(
        [
            f"Episode {index}:",
            f"When: {episode.get('observation', '')}",
            f"Thought: {episode.get('thoughts', '')}",
            f"Did: {episode.get('action', '')}",
            f"Result: {episode.get('result', '')}",
        ]
    )


def render(store: BaseStore, user_id: str, query: str | None = None, limit: int = 3) -> str:
    if query:
        items = store.search(namespace(user_id), query=query, limit=limit)
        episodes = [_one(index, payload(item)) for index, item in enumerate(items, start=1)]
        return "\n\n".join(episodes) or "(empty)"
    return render_items(store.search(namespace(user_id), limit=limit))
