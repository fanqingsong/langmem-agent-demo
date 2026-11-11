import json
import os
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


def chat_model() -> ChatOpenAI:
    return ChatOpenAI(
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
