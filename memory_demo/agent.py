import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from memory_demo.memories import (
    USER_ID,
    build_managers,
    is_feedback,
    load_instructions,
    namespace,
    save_instructions,
)


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            return _text(message)
    return ""


def _payload(item: Any) -> Any:
    value = item.value
    if isinstance(value, dict) and "content" in value:
        return value["content"]
    return value


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_system_prompt(store: BaseStore, config: RunnableConfig, user_text: str) -> str:
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", USER_ID)
    thread_id = configurable.get("thread_id", "thread")
    instructions = load_instructions(store, user_id)

    working_items = store.search(namespace("working", user_id, thread_id), limit=5)
    semantic_items = store.search(
        namespace("semantic", user_id, thread_id),
        query=user_text,
        limit=5,
    )
    episode_items = store.search(
        namespace("episodic", user_id, thread_id),
        query=user_text,
        limit=3,
    )

    working = "\n\n".join(_render(_payload(item)) for item in working_items) or "(empty)"
    semantic = "\n\n".join(_render(_payload(item)) for item in semantic_items) or "(empty)"

    episodes = []
    for index, item in enumerate(episode_items, start=1):
        episode = _payload(item)
        if not isinstance(episode, dict):
            episodes.append(f"Episode {index}:\n{_render(episode)}")
            continue
        episodes.append(
            "\n".join(
                [
                    f"Episode {index}:",
                    f"When: {episode.get('observation', '')}",
                    f"Thought: {episode.get('thoughts', '')}",
                    f"Did: {episode.get('action', '')}",
                    f"Result: {episode.get('result', '')}",
                ]
            )
        )
    episodic = "\n\n".join(episodes) or "(empty)"

    return f"""{instructions}

Use each memory for its own job:
- Procedural instructions above decide how you respond.
- Semantic facts are durable truths about the user.
- Episodic memories are past experiences to imitate when the situation is similar.
- Working context is only the active task in this thread.
Do not claim to remember anything that is not written in these sections.

## Working memory
<working>
{working}
</working>

## Semantic memory
<semantic>
{semantic}
</semantic>

## Episodic memory
<episodic>
{episodic}
</episodic>
"""


def build_graph(store: BaseStore, checkpointer: Checkpointer):
    working_manager, semantic_manager, episode_manager, optimizer = build_managers(store)
    model = working_manager.model

    def chat(state: MessagesState, config: RunnableConfig) -> dict:
        user_text = latest_user_text(state["messages"])
        system = build_system_prompt(store, config, user_text)
        response = model.invoke(
            [{"role": "system", "content": system}, *state["messages"]],
            config,
        )
        return {"messages": [response]}

    def reflect(state: MessagesState, config: RunnableConfig) -> dict:
        payload = {"messages": state["messages"]}
        working_manager.invoke(payload, config)
        semantic_manager.invoke(payload, config)
        episode_manager.invoke(payload, config)

        feedback = latest_user_text(state["messages"])
        if is_feedback(feedback):
            configurable = config.get("configurable", {})
            user_id = configurable.get("user_id", USER_ID)
            current = load_instructions(store, user_id)
            updated = optimizer.invoke(
                {
                    "trajectories": [(state["messages"], {"comment": feedback})],
                    "prompt": current,
                }
            )
            save_instructions(store, user_id, updated)
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_node("reflect", reflect)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", "reflect")
    graph.add_edge("reflect", END)
    return graph.compile(store=store, checkpointer=checkpointer)


def format_memory_board(
    store: BaseStore,
    user_id: str,
    thread_id: str,
    messages: list[Any],
) -> str:
    sections = ["## Working memory"]
    transcript = []
    for message in messages[-6:]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = getattr(message, "type", "message")
        transcript.append(f"{role}: {_text(message)}")
    sections.append("Thread transcript (checkpointer):")
    sections.append("\n".join(transcript) or "(empty)")
    sections.append("Active profile:")
    working_items = store.search(namespace("working", user_id, thread_id), limit=5)
    sections.append(
        "\n\n".join(_render(_payload(item)) for item in working_items) or "(empty)"
    )

    sections.append("\n## Semantic memory")
    semantic_items = store.search(namespace("semantic", user_id, thread_id), limit=20)
    sections.append(
        "\n\n".join(_render(_payload(item)) for item in semantic_items) or "(empty)"
    )

    sections.append("\n## Episodic memory")
    episode_items = store.search(namespace("episodic", user_id, thread_id), limit=20)
    sections.append(
        "\n\n".join(_render(_payload(item)) for item in episode_items) or "(empty)"
    )

    sections.append("\n## Procedural memory")
    item = store.get(namespace("procedural", user_id, thread_id), "instructions")
    if item is None:
        sections.append("(empty)")
    else:
        sections.append(item.value.get("instructions", _render(item.value)))
    return "\n".join(sections)
