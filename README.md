# LangMem Agent Demo

A demonstration of agent memory with LangMem, LangGraph, SiliconFlow for chat, Ollama for embeddings, and Redis for persistence.

| Memory | What it keeps | Where it lives | Scope |
| --- | --- | --- | --- |
| Short-term | The messages in the current thread | Checkpointer, keyed by `thread_id` | This thread only |
| Working | One active-task profile | `("working", user_id, thread_id)` | This thread only |
| Semantic | Durable facts such as name and display preference | `("semantic", user_id)` | Across threads |
| Episodic | A past situation, the reasoning, the action, and the result | `("episodic", user_id)` | Across threads |
| Procedural | Durable response rules such as teaching style and length | `("procedural", user_id)` | Across threads |

Short-term memory is LangGraph thread state. `RedisSaver` appends each turn to that thread, and the next turn on the same `thread_id` resumes from the saved messages. The chat node keeps the latest part of the transcript that fits 4,000 tokens. LangMem has no separate working-memory API. A `create_memory_store_manager` with `enable_inserts=False` keeps the single working profile. Semantic facts, episodes, and procedural rules each use their own store manager. The base system prompt stays in code. Procedural rules are inserted, updated, and deleted when the user gives or retracts behavioral feedback.

Each turn reads the thread transcript and the other memories, answers, then writes working, semantic, episodic, and procedural memories. A rule written on one turn is followed on the next turn.

## Prerequisites

- **uv**: Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Ollama**: For local embeddings ([install guide](https://ollama.ai/download))
- **Redis 8**: For persistent memory (RedisJSON and RediSearch)
- **SiliconFlow API Key**: Get one from [SiliconFlow](https://cloud.siliconflow.cn/account/ak)

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

3. **Set the SiliconFlow API key** in `.env`:

   ```bash
   SILICONFLOW_API_KEY=your-api-key-here
   ```

   `.env.example` also lists `SILICONFLOW_BASE_URL`, `SILICONFLOW_MODEL`, and the optional `REDIS_URI` and `OLLAMA_BASE_URL` overrides. The default chat model is `deepseek-ai/DeepSeek-V3.2` on `https://api.siliconflow.cn/v1`.

## Run

Start the API and the chat page. Compose also starts Redis, Ollama, and the embedding pull when they are not already up:

```bash
docker compose up frontend
```

Open http://127.0.0.1:8080. The left side is the conversation. The right side shows this thread's short-term and working memory, plus the user's semantic, episodic, and procedural memory. The API is on http://127.0.0.1:8000 (`POST /api/chat`, `GET /api/memories`).

`agent.py` is still a local scripted run of the same graph:

```bash
uv run agent.py
```

The script plays three scenes for user `demo-user`:

1. **thread-a:** Alex prefers dark mode and asks what a binary tree is, mentioning family trees.
2. **thread-a:** The analogy worked. From now on, teach with analogies and keep answers short. This turn writes a procedural rule.
3. **thread-b:** Ask what a hash map is, given that Python dictionaries are familiar, and ask which display mode they prefer.

After each scene the script prints a memory board:

- Short-term memory shows this thread's messages. thread-b does not keep thread-a's transcript.
- Working memory shows this thread's active task profile. thread-b does not keep thread-a's goal.
- Semantic memory keeps facts such as the name and dark-mode preference.
- Episodic memory keeps a reusable teaching episode when one was worth saving.
- Procedural memory lists response rules. Scene 3 follows the rule written in scene 2.

Run `uv run agent.py` a second time while Redis is still up. Semantic facts, episodes, and procedural rules are still in the store. Short-term transcripts and working profiles stay on the thread id they were written under.

## Verify each memory

Extraction is done by the chat model, so the stored wording varies. A memory is working when the right panel (or `GET /api/memories`) gains the expected record, and a later turn uses it. Use a fresh `user_id` so leftovers from `demo-user` do not confuse the check. The page always uses `demo-user`; the curls below use `verify-user`.

```bash
curl -s http://127.0.0.1:8000/api/memories?thread_id=THREAD&user_id=verify-user
```

`short_term` and `working` are filtered to that thread. `semantic`, `episodic`, and `procedural` are filtered to that user.

### Short-term

Same thread resumes the transcript. A new thread does not.

1. On `thread-a`, send: `What is 17 + 25?`
2. On `thread-a`, send: `What two numbers did I just ask you to add?`
3. On `thread-b`, send the same follow-up.

`thread-a` answers 17 and 25. `thread-b` does not, because that transcript is a different checkpoint. `GET /api/memories?thread_id=thread-a` lists both turns under `short_term`; the same call for `thread-b` does not.

```bash
curl -s http://127.0.0.1:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"verify-user","thread_id":"thread-a","message":"What is 17 + 25?"}'
```

### Working

One profile per thread, rewritten in place. It does not follow the user to another thread.

1. On `thread-a`, send: `Help me compare merge sort and quick sort for homework due Friday.`
2. On `thread-a`, send: `Add heap sort to that comparison.`
3. Read `working` for `thread-a`. There is still one profile. Its goal covers the three sorts, and Friday is a constraint.
4. On `thread-b`, send: `What homework am I doing in the other thread?`

`thread-b` has an empty working profile, and the reply does not use `thread-a`'s goal. Name, display preference, and other durable facts stay out of this profile.

### Semantic

Durable facts survive a new thread, and a correction replaces the old fact.

1. On `thread-a`, send: `I'm Sam. I prefer dark mode. I already know Python dictionaries.`
2. Read `semantic`. Expect an identity fact for Sam, a dark-mode preference, and knowledge of Python dictionaries.
3. On `thread-b`, send: `What's my name, and which display mode do I prefer?`
4. The reply says Sam and dark mode, even though `thread-b` has no earlier messages.
5. On `thread-b`, send: `I prefer light mode now.`
6. Read `semantic` again. The display preference is light mode. The dark-mode fact is updated or gone.

### Episodic

An episode is written after an approach is shown to work, then retrieved for a similar situation. Confirming one success is not a standing rule.

1. On `thread-a`, send: `Explain a stack by comparing it to a stack of plates.`
2. On `thread-a`, send: `That plate analogy worked. I understood it.`
3. Read `episodic`. Expect one episode whose observation, action, and result are about the plate analogy. A greeting such as `Hi` adds no episode.
4. On `thread-b`, send: `Explain a queue.`

The reply imitates the saved approach, usually another physical analogy, because the queue question retrieves that episode. The procedural list stays empty unless you also gave a standing instruction.

### Procedural

A standing instruction is stored as a rule, followed on the next turn and on other threads, and removed when retracted.

1. On `thread-a`, send: `From now on, teach with analogies and keep each answer to three sentences.`
2. Read `procedural`. Expect a rule whose instruction mentions analogies and short answers.
3. On `thread-b`, send: `What's a hash map?`
4. The reply is short and uses an analogy.
5. On `thread-b`, send: `Stop using analogies. Give plain definitions.`
6. Read `procedural`. The analogy rule is updated or deleted.
7. On `thread-b`, send: `What's a set?` The reply is a plain definition.

Dark mode and the user's name stay in semantic memory. They are not procedural rules.

### Persistence

Restart the agent process and leave Redis running (`docker compose restart agent`). Call `GET /api/memories` for the same `user_id` and `thread_id`:

- Semantic facts, episodes, and procedural rules are still there.
- That thread's short-term transcript and working profile are still there.
- A thread id you have not used is still empty for short-term and working memory.

## Architecture

`frontend` is nginx. It serves the chat page and proxies `/api/` to the `agent` service. `src.api` keeps `RedisStore` and `RedisSaver` open and serves `POST /api/chat` and `GET /api/memories`. `agent.py` plays the scripted conversations against the same graph.

Each memory type is its own module under `src.memory`. `short_term` reads the checkpointer transcript and trims it before the model call. `working`, `semantic`, `episodic`, and `procedural` each own a schema, extraction instructions, namespace, store manager, and rendering. `src.graph` compiles a two-node graph:

1. `chat` reads the thread transcript and asks each store-backed module for its memory, then calls the SiliconFlow chat model.
2. `reflect` asks each store manager to write. The checkpointer saves the new messages with the thread. The procedural manager inserts, updates, and deletes response rules.

Chat model: SiliconFlow `deepseek-ai/DeepSeek-V3.2` (override with `SILICONFLOW_MODEL`). Embeddings: Ollama `nomic-embed-text`.

## Project goals

- [x] Persist short-term memory in Redis with a LangGraph checkpointer, and working, semantic, episodic, and procedural memory with LangMem
- [ ] Investigate Valkey compatibility as a Redis alternative
