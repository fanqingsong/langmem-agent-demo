import json
import os
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

USER_ID = "demo-user"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"


def siliconflow_base_url() -> str:
    return os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL)


def siliconflow_model() -> str:
    return os.getenv("SILICONFLOW_MODEL", DEFAULT_MODEL)


def textual_tool_calls(content: str) -> list[dict[str, Any]]:
    """Some models write function calls into the message text instead of tool_calls."""
    stripped = content.strip()
    if not stripped or stripped[0] not in "[{":
        return []
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    index = 0
    while True:
        start = content.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = end
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        arguments = value.get("parameters", value.get("arguments"))
        if not isinstance(name, str) or not name or arguments is None:
            continue
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


def repair_message_tool_calls(response: dict[str, Any]) -> bool:
    """Promote tool calls written into content so memory extraction can see them."""
    repaired = False
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        if message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        calls = textual_tool_calls(content)
        if not calls:
            continue
        message["tool_calls"] = calls
        message["content"] = ""
        repaired = True
    return repaired


class ChatModel(ChatOpenAI):
    def _create_chat_result(self, response: dict | Any, generation_info: dict | None = None):
        response_dict = response if isinstance(response, dict) else response.model_dump()
        if repair_message_tool_calls(response_dict):
            return super()._create_chat_result(response_dict, generation_info)
        return super()._create_chat_result(response, generation_info)


def chat_model() -> ChatOpenAI:
    return ChatModel(
        model=siliconflow_model(),
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url=siliconflow_base_url(),
    )


def scope_from(config: RunnableConfig) -> tuple[str, str]:
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", USER_ID)
    thread_id = configurable.get("thread_id", "thread")
    return user_id, thread_id


def payload(item: Any) -> Any:
    value = item.value
    if isinstance(value, dict) and "content" in value:
        return value["content"]
    return value


def render_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def render_items(items: list[Any]) -> str:
    rendered = "\n\n".join(render_value(payload(item)) for item in items)
    return rendered or "(empty)"
