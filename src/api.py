"""HTTP API for the LangGraph memory agent."""

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_graph
from src.memory import semantic
from src.memory.common import USER_ID
from src.memory_board import KINDS, snapshot
from src.runtime import open_runtime

load_dotenv()

THREAD_SCOPED = ("short_term",)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "thread"
    user_id: str = USER_ID


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open_runtime() as (store, checkpointer):
        app.state.store = store
        app.state.checkpointer = checkpointer
        app.state.agent = build_graph(store, checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, str]:
    config = {"configurable": {"thread_id": body.thread_id, "user_id": body.user_id}}
    try:
        response = app.state.agent.invoke(
            {"messages": [{"role": "user", "content": body.message}]},
            config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    messages = response.get("messages") or []
    if not messages:
        raise HTTPException(status_code=500, detail="agent returned no messages")
    return {
        "reply": _text(messages[-1]),
        "thread_id": body.thread_id,
        "user_id": body.user_id,
    }


@app.get("/api/memories")
def memories(thread_id: str, user_id: str = USER_ID) -> dict[str, Any]:
    semantic.drop_meta(app.state.store, user_id)
    board = snapshot(app.state.store, app.state.checkpointer)
    items = {
        kind: [
            _public_item(item)
            for item in board["items"].get(kind, [])
            if _visible(kind, item, thread_id, user_id)
        ]
        for kind in KINDS
    }
    return {
        "polled_at": board["polled_at"],
        "thread_id": thread_id,
        "user_id": user_id,
        "counts": {kind: len(items[kind]) for kind in KINDS},
        "items": items,
    }


def _visible(kind: str, item: dict[str, Any], thread_id: str, user_id: str) -> bool:
    namespace = item.get("namespace") or []
    if kind == "short_term":
        return item.get("key") == thread_id
    if kind in KINDS and kind not in THREAD_SCOPED:
        return len(namespace) >= 2 and namespace[1] == user_id
    return False


def _public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "kind": item["kind"],
        "namespace": item["namespace"],
        "key": item["key"],
        "title": item["title"],
        "body": item["body"],
        "updated_at": item["updated_at"],
    }


def _text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)
