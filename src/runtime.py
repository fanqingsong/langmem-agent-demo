import os
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore
from langgraph.types import Checkpointer

EMBEDDING_DIMS = 768


def redis_uri() -> str:
    return os.getenv("REDIS_URI", "redis://localhost:6379")


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url())


def _index(embeddings: OllamaEmbeddings) -> dict:
    return {
        "dims": EMBEDDING_DIMS,
        "embed": embeddings,
        "distance_type": "cosine",
    }


@contextmanager
def open_runtime() -> Iterator[tuple[BaseStore, Checkpointer]]:
    """Open Redis store and checkpointer and keep them alive for the demo."""
    with (
        RedisStore.from_conn_string(redis_uri(), index=_index(build_embeddings())) as store,
        RedisSaver.from_conn_string(redis_uri()) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        yield store, checkpointer
