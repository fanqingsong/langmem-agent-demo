from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

# The checkpointer keeps the whole thread. This is how much of it the model sees.
MAX_MODEL_TOKENS = 4000


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _role(message: Any) -> str:
    if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
        return "user"
    if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
        return "assistant"
    return getattr(message, "type", "message")


def messages_for_model(messages: list[Any]) -> list[Any]:
    """Keep the latest part of this thread that fits the model window."""
    if not messages:
        return []
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_MODEL_TOKENS,
        start_on="human",
        end_on=("human", "ai", "tool"),
    )


def render(messages: list[Any]) -> str:
    lines = [f"{_role(message)}: {_text(message)}" for message in messages]
    return "\n".join(lines) or "(empty)"
