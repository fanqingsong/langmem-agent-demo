"""Memory modules. Store-backed types own a schema and a store manager.

Short-term memory is the thread transcript in the checkpointer.
"""

from langgraph.store.base import BaseStore

from src.memory import episodic, procedural, semantic
from src.memory.common import USER_ID, chat_model

__all__ = ["USER_ID", "build_managers"]


def build_managers(store: BaseStore):
    model = chat_model()
    return (
        semantic.manager(model, store),
        episodic.manager(model, store),
        procedural.manager(model, store),
    )
