"""Read-only snapshot of the memories stored in Redis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.load import load
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore

from src.memory.common import payload

KINDS = ("short_term", "semantic", "episodic", "procedural")
STORE_KINDS = ("semantic", "episodic", "procedural")
SEARCH_LIMIT = 50


@contextmanager
def open_board(redis_uri: str) -> Iterator[tuple[RedisStore, RedisSaver]]:
    """Open the store and checkpointer for read-only snapshots."""
    with (
        RedisStore.from_conn_string(redis_uri) as store,
        RedisSaver.from_conn_string(redis_uri) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        yield store, checkpointer


def snapshot(store: BaseStore, checkpointer: RedisSaver) -> dict[str, Any]:
    items = _store_items(store)
    items.extend(_thread_items(checkpointer))
    grouped = {kind: [] for kind in KINDS}
    for item in items:
        grouped.setdefault(item["kind"], []).append(item)
    return {
        "polled_at": _now(),
        "counts": {kind: len(grouped.get(kind, [])) for kind in KINDS},
        "items": grouped,
    }


def _store_items(store: BaseStore) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind in STORE_KINDS:
        for entry in store.search((kind,), limit=SEARCH_LIMIT):
            namespace = tuple(entry.namespace)
            item_id = f"{kind}|{'.'.join(namespace)}|{entry.key}"
            if item_id in seen:
                continue
            seen.add(item_id)
            value = payload(entry)
            items.append(
                {
                    "id": item_id,
                    "kind": kind,
                    "namespace": list(namespace),
                    "key": entry.key,
                    "title": _title(kind, value),
                    "body": _body(value),
                    "created_at": _iso(entry.created_at),
                    "updated_at": _iso(entry.updated_at),
                    "fingerprint": _fingerprint(value),
                }
            )
    return items


def _thread_items(checkpointer: RedisSaver) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for saved in checkpointer.list(None):
        configurable = saved.config.get("configurable", {})
        if configurable.get("checkpoint_ns"):
            continue
        thread_id = str(configurable.get("thread_id") or "thread")
        checkpoint = saved.checkpoint or {}
        ts = str(checkpoint.get("ts") or "")
        current = latest.get(thread_id)
        if current is not None and ts < current["ts"]:
            continue
        messages = _messages(checkpoint.get("channel_values", {}).get("messages") or [])
        latest[thread_id] = {"ts": ts, "messages": messages}

    items = []
    for thread_id, row in sorted(latest.items()):
        messages = row["messages"]
        items.append(
            {
                "id": f"short_term|{thread_id}",
                "kind": "short_term",
                "namespace": ["short_term", thread_id],
                "key": thread_id,
                "title": thread_id,
                "body": messages,
                "created_at": row["ts"],
                "updated_at": row["ts"],
                "fingerprint": _fingerprint(messages),
            }
        )
    return items


def _messages(messages: list[Any]) -> list[dict[str, str]]:
    rendered = []
    for message in messages:
        message = _coerce_message(message)
        rendered.append({"role": _role(message), "content": _text(message)})
    return rendered


def _coerce_message(message: Any) -> Any:
    """Checkpoints may still hold LangChain's serialized constructor form."""
    if isinstance(message, dict) and message.get("type") == "constructor":
        try:
            return load(message)
        except Exception:
            return message.get("kwargs") or message
    return message


def _role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "message")
    if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
        return "user"
    if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
        return "assistant"
    return str(getattr(message, "type", "message"))


def _text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _title(kind: str, value: Any) -> str:
    if isinstance(value, dict):
        for key in ("content", "instruction", "observation"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    if kind == "short_term":
        return "thread"
    rendered = _body(value)
    if isinstance(rendered, str):
        return rendered.strip().splitlines()[0][:160] if rendered.strip() else "(empty)"
    return kind


def _body(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
