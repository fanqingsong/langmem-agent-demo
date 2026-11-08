import os
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Checkpointer

REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_DIMS = 768


def build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)


def _index(embeddings: OllamaEmbeddings, *, redis: bool) -> dict:
    index: dict = {
        "dims": EMBEDDING_DIMS,
        "embed": embeddings,
    }
    if redis:
        index["distance_type"] = "cosine"
    return index


@contextmanager
def open_runtime(backend: str) -> Iterator[tuple[BaseStore, Checkpointer]]:
    """Open a store and checkpointer and keep them alive for the demo process."""
    embeddings = build_embeddings()
    if backend == "memory":
        yield InMemoryStore(index=_index(embeddings, redis=False)), MemorySaver()
        return
    if backend != "redis":
        raise ValueError(f"Unknown memory backend: {backend}")

    from langgraph.checkpoint.redis import RedisSaver
    from langgraph.store.redis import RedisStore

    with (
        RedisStore.from_conn_string(
            REDIS_URI, index=_index(embeddings, redis=True)
        ) as store,
        RedisSaver.from_conn_string(REDIS_URI) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        yield store, checkpointer
