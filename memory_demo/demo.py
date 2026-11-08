import os
import sys

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from memory_demo.agent import build_graph, format_memory_board
from memory_demo.memories import USER_ID
from memory_demo.runtime import OLLAMA_BASE_URL, REDIS_URI, open_runtime

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


def main(backend: str) -> None:
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required. Put it in a .env file or export it before starting."
        )

    print(f"\nMemory backend: {backend}")
    if backend == "redis":
        print(f"Redis: {REDIS_URI}")
    print(f"Embeddings: Ollama nomic-embed-text at {OLLAMA_BASE_URL}")
    print(f"User namespace: {USER_ID}\n")

    try:
        with open_runtime(backend) as (store, checkpointer):
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
        if backend == "redis":
            print("\nTroubleshooting:")
            print("  1. Is Redis running? Try: docker compose ps")
            print("  2. Can you connect? Try: redis-cli ping")
            print("  3. Redis 8 includes RedisJSON and RediSearch, which this store needs.")
        raise

    print("=" * 60)
    if backend == "redis":
        print("Semantic, episodic, and procedural memories stay in Redis.")
        print("Run this script again and those memories are still there.")
        print("Working memory stays on its own thread and does not carry the other thread's goal.")
    else:
        print("In-memory store dropped when this process exits.")
        print("Run agent_redis.py to keep semantic, episodic, and procedural memories.")
    print("=" * 60)


if __name__ == "__main__":
    selected = sys.argv[1] if len(sys.argv) > 1 else "memory"
    main(selected)
