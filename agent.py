import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.graph import build_graph, format_memory_board
from src.memory import USER_ID
from src.memory.common import siliconflow_base_url, siliconflow_model
from src.runtime import ollama_base_url, open_runtime, redis_uri

SCENES = (
    (
        "thread-a",
        "Introduce yourself and ask for an explanation",
        "Hi, I'm Alex. I prefer dark mode. What's a binary tree? "
        "I work with family trees if that helps.",
    ),
    (
        "thread-a",
        "Confirm the analogy and give behavioral feedback",
        "That analogy worked. From now on, teach with analogies and keep answers short.",
    ),
    (
        "thread-b",
        "New thread that should reuse long-term memory only",
        "What's a hash map? I already know Python dictionaries. "
        "Which display mode do I prefer?",
    ),
)


def main() -> None:
    load_dotenv()
    if not os.getenv("SILICONFLOW_API_KEY"):
        raise SystemExit(
            "SILICONFLOW_API_KEY is required. Put it in a .env file or export it before starting."
        )

    print(f"\nRedis: {redis_uri()}")
    print(f"Chat model: {siliconflow_model()} at {siliconflow_base_url()}")
    print(f"Embeddings: Ollama nomic-embed-text at {ollama_base_url()}")
    print(f"User namespace: {USER_ID}\n")

    try:
        with open_runtime() as (store, checkpointer):
            agent = build_graph(store, checkpointer)
            for thread_id, title, content in SCENES:
                print("=" * 60)
                print(f"{title} ({thread_id})")
                print("=" * 60)
                print(f"User: {content}\n")
                config: RunnableConfig = {
                    "configurable": {"thread_id": thread_id, "user_id": USER_ID}
                }
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": content}]},
                    config,
                )
                print(f"Agent: {response['messages'][-1].content}\n")
                print("-" * 60)
                print(format_memory_board(store, USER_ID, thread_id, response["messages"]))
                print()
    except Exception as exc:
        print(f"\nError: {exc}")
        print("\nTroubleshooting:")
        print("  1. Is Redis running? Try: docker compose ps")
        print("  2. Can you connect? Try: redis-cli ping")
        print("  3. Redis 8 includes RedisJSON and RediSearch, which this store needs.")
        raise

    print("=" * 60)
    print("Semantic, episodic, and procedural memories stay in Redis.")
    print("Run this script again and those memories are still there.")
    print("Short-term memory and the working profile stay on their own thread.")
    print("=" * 60)


if __name__ == "__main__":
    main()
