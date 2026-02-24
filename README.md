# Nota Code

Discord-based personal knowledge workspace with GitHub-backed storage, Chroma vector indexing, and RAG answers via LM Studio (default) or Groq (`--groq`).

## What It Does
- Creates a private per-user workspace thread in Discord.
- Connects each user to a dedicated GitHub repository.
- Ingests content from YouTube, web pages, and repository files.
- Builds and updates a local Chroma vector index.
- Answers questions using RAG with LM Studio.
- Runs long tasks asynchronously with a job queue.

## Core Components
- `nota_bot.py`: Discord command handlers, RBAC, thread checks, rate limit, job orchestration.
- `github_client.py`: Workspace provisioning, clone/push, file operations, history.
- `knowledge_base.py`: Ingestion, chunking, embedding/indexing, retrieval, LM Studio calls.
- `job_queue.py`: Async job queue + SQLite state tracking.
- `security_utils.py`: Path sanitization and safety checks.
- `config.py`: `.env` loading and runtime settings.

## Requirements
- Python 3.10+
- Discord bot token
- GitHub token (repo access)
- LM Studio running locally (OpenAI-compatible API)
- Optional: Groq API key when using `--groq`

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install discord.py PyGithub GitPython chromadb sentence-transformers requests beautifulsoup4 youtube-transcript-api python-dotenv pypdf
```

## Environment Variables
Create `.env` in project root:

```dotenv
DISCORD_TOKEN=...
GITHUB_TOKEN=...
NOTA_THREAD_PARENT_CHANNEL_ID=123456789012345678

# Optional
GITHUB_ORG=
NOTA_REPO_PREFIX=nota-kb
DEFAULT_BRANCH=main
NOTA_DATA_DIR=./data

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=local-model
LM_STUDIO_MAX_PROMPT_CHARS=3500
LM_STUDIO_CONTEXT_WINDOW=2048
LM_STUDIO_RESERVED_TOKENS=512
LM_STUDIO_CHARS_PER_TOKEN=3.0

# Optional when running with --groq
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile

NOTA_ADMIN_USER_IDS=
NOTA_ALLOWED_LOCAL_FILE_EXTENSIONS=txt,json,pdf
NOTA_LOCAL_FILE_MAX_BYTES=3145728
NOTA_WEB_MAX_BYTES=2097152
NOTA_WEB_REQUEST_TIMEOUT_SECONDS=15
NOTA_ALLOWED_WEB_DOMAINS=
NOTA_COMMAND_COOLDOWN_SECONDS=5
NOTA_COMMAND_WINDOW_SECONDS=60
NOTA_COMMAND_MAX_PER_WINDOW=12
```

## Run
```bash
python nota_bot.py

# Use Groq API instead of LM Studio
python nota_bot.py --groq
```

## Discord Commands
- `/open_thread [collaborators]`: Create private command thread.
- `/setup [github_username]`: Create/connect personal GitHub workspace.
- `/invite <github_username>`: Invite collaborator to workspace repo.
- `/add <source_type> <source>`: Ingest source (`youtube`, `web`, `repo`).
- `/chat <question>`: Ask a question (RAG + LM Studio response generation).
- `/summarize_all`: Generate repository docs from indexed knowledge.
- `/analyze_update <path> <instruction>`: Analyze file and update docs.
- `/list`: List workspace files.
- `/history <path> [limit]`: Show git history for a file.
- `/edit <path> <new_content>`: Edit file.
- `/delete <path>`: Delete file.
- `/reset`: Reset repository files and local index.
- `/job_status <job_id>`: Check async job status/result.

## RAG Response Flow
1. User asks `/chat`.
2. Top-k relevant chunks are retrieved from Chroma.
3. Retrieved context is injected into the selected LLM chat prompt.
4. LM Studio (default) or Groq (`--groq`) generates final answer based on question + context.

## Security Model
- Commands run only inside managed private threads (except thread creation).
- Role-based access control (`owner`, `collaborator`, `admin`).
- Per-user rate limit and cooldown.
- Repository-relative path sanitization.
- External URL validation and internal IP blocking.
- Audit logging for command outcomes.

## Data Storage
- `data/nota.sqlite3`: user mappings, roles, jobs.
- `data/chroma/`: persistent vector index.
- `data/index_hash_state.json`: incremental indexing state.

## Notes
- If the selected LLM endpoint is unavailable, chat/doc generation returns an availability error.
- For best results, run `/add` first to build enough context before `/chat`.
