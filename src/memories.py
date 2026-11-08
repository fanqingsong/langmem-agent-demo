from langchain_deepseek import ChatDeepSeek
from langgraph.store.base import BaseStore
from langmem import create_memory_store_manager, create_prompt_optimizer

from src.schemas import (
    INITIAL_INSTRUCTIONS,
    Episode,
    SemanticFact,
    WorkingContext,
)

USER_ID = "demo-user"

WORKING_INSTRUCTIONS = """Maintain the single WorkingContext for this thread.

Update it to match the active task: the current goal, constraints, open questions,
and decisions made in this conversation. Drop goals from earlier threads.
Do not copy durable user facts (name, preferences, background knowledge) into this
profile; those belong in semantic memory.
Revise the existing profile instead of creating another one.
"""

SEMANTIC_INSTRUCTIONS = """Extract durable facts about the user.

Keep identity, preferences, and stable background knowledge.
Skip the current task, transient questions, and instructions about how the
assistant should behave. Update a fact when the user corrects it, and delete
facts that are contradicted.
"""

EPISODIC_INSTRUCTIONS = """Extract episodes only when an interaction shows a reusable way
of handling a situation.

Each episode needs the situation, the reasoning, the action, and why the outcome
was good or bad. Skip greetings, bare preference statements, and unfinished
exchanges that do not yet show whether the approach worked.
Prefer one dense episode over several overlapping ones.
"""

FEEDBACK_MARKERS = (
    "以后",
    "从现在起",
    "接下来请",
    "记住要",
    "简短",
    "类比",
    "from now on",
    "next time",
    "be concise",
    "keep answers short",
    "keep it short",
    "use analogies",
    "teach with analogies",
    "analogy worked",
)


def chat_model() -> ChatDeepSeek:
    return ChatDeepSeek(model="deepseek-chat")


def build_managers(store: BaseStore):
    model = chat_model()
    working = create_memory_store_manager(
        model,
        schemas=[WorkingContext],
        instructions=WORKING_INSTRUCTIONS,
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
    semantic = create_memory_store_manager(
        model,
        schemas=[SemanticFact],
        instructions=SEMANTIC_INSTRUCTIONS,
        namespace=("semantic", "{user_id}"),
        enable_inserts=True,
        enable_deletes=True,
        store=store,
    )
    episodic = create_memory_store_manager(
        model,
        schemas=[Episode],
        instructions=EPISODIC_INSTRUCTIONS,
        namespace=("episodic", "{user_id}"),
        enable_inserts=True,
        enable_deletes=False,
        store=store,
    )
    procedural = create_prompt_optimizer(model, kind="prompt_memory")
    return working, semantic, episodic, procedural


def namespace(kind: str, user_id: str, thread_id: str | None = None) -> tuple[str, ...]:
    if kind == "working":
        if not thread_id:
            raise ValueError("working memory is scoped to a thread")
        return ("working", user_id, thread_id)
    if kind in {"semantic", "episodic", "procedural"}:
        return (kind, user_id)
    raise ValueError(f"Unknown memory kind: {kind}")


def is_feedback(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in FEEDBACK_MARKERS)


def load_instructions(store: BaseStore, user_id: str) -> str:
    item = store.get(namespace("procedural", user_id), "instructions")
    if item is None:
        store.put(
            namespace("procedural", user_id),
            "instructions",
            {"instructions": INITIAL_INSTRUCTIONS},
        )
        return INITIAL_INSTRUCTIONS
    return item.value["instructions"]


def save_instructions(store: BaseStore, user_id: str, instructions: str) -> None:
    store.put(
        namespace("procedural", user_id),
        "instructions",
        {"instructions": instructions},
    )
