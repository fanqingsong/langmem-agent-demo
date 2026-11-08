# LangMem Agent Demo

A demonstration of four agent memory types with LangMem, LangGraph, DeepSeek for chat, Ollama for embeddings, and Redis for persistence.

| Memory | What it keeps | Where it lives | Scope |
| --- | --- | --- | --- |
| Working | Current thread transcript, plus one active-task profile | Checkpointer, and `("working", user_id, thread_id)` | This thread only |
| Semantic | Durable facts such as name and display preference | `("semantic", user_id)` | Across threads |
| Episodic | A past situation, the reasoning, the action, and the result | `("episodic", user_id)` | Across threads |
| Procedural | The system instructions the agent follows | `("procedural", user_id)` key `instructions` | Across threads |

LangMem has no separate working-memory API. The checkpointer is the thread transcript. A `create_memory_store_manager` with `enable_inserts=False` keeps the single working profile. Semantic facts and episodes use their own store managers. Procedural memory uses `create_prompt_optimizer` (`kind="prompt_memory"`) and is rewritten only when the user gives behavioral feedback.

Each turn reads all four memories into the system prompt, answers, then writes working, semantic, and episodic memories. A feedback turn also rewrites the procedural instructions. The reply on that turn still uses the previous instructions; the next thread uses the updated ones.

## Prerequisites

- **uv**: Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Ollama**: For local embeddings ([install guide](https://ollama.ai/download))
- **Redis 8**: For persistent memory (RedisJSON and RediSearch)
- **DeepSeek API Key**: Get one from [DeepSeek Platform](https://platform.deepseek.com/api_keys)

## Setup

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Start Redis and pull the embedding model:**

   ```bash
   docker compose up -d redis ollama
   docker compose up ollama-pull
   ```

   `ollama-pull` downloads `nomic-embed-text` (768 dimensions) and exits.

3. **Set the DeepSeek API key** in `.env`:

   ```bash
   DEEPSEEK_API_KEY=your-api-key-here
   ```

   `.env.example` lists the optional `REDIS_URI` and `OLLAMA_BASE_URL` overrides.

## Run

```bash
uv run agent.py
```

Or run the same script inside Compose:

```bash
docker compose run --rm agent
```

The script plays three scenes for user `demo-user`:

1. **thread-a:** Alex prefers dark mode and asks what a binary tree is, mentioning family trees.
2. **thread-a:** The analogy worked. From now on, teach with analogies and keep answers short. This turn updates procedural memory.
3. **thread-b:** Ask what a hash map is, given that Python dictionaries are familiar, and ask which display mode they prefer.

After each scene the script prints a memory board:

- Working memory shows this thread's transcript and active profile. thread-b does not keep thread-a's goal.
- Semantic memory keeps facts such as the name and dark-mode preference.
- Episodic memory keeps a reusable teaching episode when one was worth saving.
- Procedural memory prints the current instructions. Scene 3 uses the instructions rewritten in scene 2.

Run `uv run agent.py` a second time while Redis is still up. Semantic facts, episodes, and the rewritten instructions are still in the store. Working profiles stay on the thread id they were written under.

## Architecture

`agent.py` is the entrypoint. It plays the scripted conversations. `src.runtime` opens `RedisStore` and `RedisSaver` and keeps both clients open until the run finishes.

`src.graph` compiles a two-node graph:

1. `chat` searches the store and calls DeepSeek.
2. `reflect` runs the LangMem store managers, and the prompt optimizer when the latest user message is behavioral feedback.

Chat model: DeepSeek `deepseek-chat`. Embeddings: Ollama `nomic-embed-text`.

## Project goals

- [x] Persist working, semantic, episodic, and procedural memory in Redis with LangMem
- [ ] Investigate Valkey compatibility as a Redis alternative
