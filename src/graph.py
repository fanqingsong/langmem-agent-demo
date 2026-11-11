from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from src.memory import build_managers
from src.memory import episodic, procedural, semantic, short_term
from src.memory.common import scope_from

INITIAL_INSTRUCTIONS = """You are a helpful assistant.

Explain concepts clearly. Use what you know about the user when it helps.
Answer the question first, then add only the detail that is needed.
"""


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


def build_system_prompt(store: BaseStore, config: RunnableConfig, user_text: str) -> str:
    user_id = scope_from(config)[0]
    procedural_text = procedural.render(store, user_id)
    semantic_text = semantic.render(store, user_id, query=user_text, limit=5)
    episodic_text = episodic.render(store, user_id, query=user_text, limit=3)

    return f"""{INITIAL_INSTRUCTIONS}

Use each memory for its own job:
- The conversation messages are short-term memory for this thread.
- Procedural rules decide how you respond.
- Semantic facts are durable truths about the user.
- Episodic memories are past experiences to imitate when the situation is similar.
Do not claim to remember anything that is not in the messages or these sections.

## Procedural memory
<procedural>
{procedural_text}
</procedural>

## Semantic memory
<semantic>
{semantic_text}
</semantic>

## Episodic memory
<episodic>
{episodic_text}
</episodic>
"""


def build_graph(store: BaseStore, checkpointer: Checkpointer):
    semantic_manager, episode_manager, procedural_manager = build_managers(store)
    model = semantic_manager.model

    def chat(state: MessagesState, config: RunnableConfig) -> dict:
        user_text = latest_user_text(state["messages"])
        system = build_system_prompt(store, config, user_text)
        history = short_term.messages_for_model(state["messages"])
        response = model.invoke(
            [{"role": "system", "content": system}, *history],
            config,
        )
        return {"messages": [response]}

    def reflect(state: MessagesState, config: RunnableConfig) -> dict:
        payload = {"messages": state["messages"]}
        semantic.drop_meta(store, scope_from(config)[0])
        semantic_manager.invoke(payload, config)
        episode_manager.invoke(payload, config)
        procedural_manager.invoke(payload, config)
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
    messages: list[Any],
) -> str:
    sections = ["## Short-term memory"]
    sections.append(short_term.render(messages))

    sections.append("\n## Semantic memory")
    sections.append(semantic.render(store, user_id, limit=20))

    sections.append("\n## Episodic memory")
    sections.append(episodic.render(store, user_id, limit=20))

    sections.append("\n## Procedural memory")
    sections.append(procedural.render(store, user_id))
    return "\n".join(sections)
