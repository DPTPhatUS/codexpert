# Codexpert

Codexpert is a Python library for AI-powered QA and deep analysis over entire codebases. It ingests local paths or public GitHub repos, builds semantic and lexical indexes, and runs a LangGraph ReAct loop to answer questions with citations.

## Core capabilities

- Dual-target analysis for local folders and public GitHub repos.
- Multi-faceted understanding via Tree-sitter parsing, ChromaDB vectors, BM25 lexical search, and NetworkX dependency graphs.
- Stateless, `repo_id`-scoped queries for multi-tenant and concurrent usage.
- Pluggable LLM provider layer (Gemini, OpenAI, and Mistral).

## Architecture

1. API interface: public classes that validate inputs and dispatch queries.
2. Orchestration layer: LangGraph ReAct loop for reasoning and tool calls.
3. Action layer: tools for file listing/reading, exact and semantic search, symbol/reference lookup, docs search, git history, and graph queries.
4. Knowledge layer: ingestion, chunking, and index building.

## Package layout

```
src/codexpert/
  core/     # configs, agent API, orchestrator
  llm/      # pluggable LLM clients
  memory/   # ingestion, chunking, indexing, storage
  tools/    # search and graph tools
  utils/    # helpers
```

## Quickstart

Install dependencies with uv:

```bash
uv pip install -e .
```

Create a local `.env` file for credentials and settings:

```bash
cp .env.example .env
```

### Local repository

```python
from codexpert import AgentConfig, CodexpertAgent, QueryRequest, RepoSpec

config = AgentConfig()
agent = CodexpertAgent(config)

repo = RepoSpec.from_source("/path/to/repo", repo_id="local-repo")
await agent.async_ingest(repo)

response = await agent.async_query(
	QueryRequest(repo_id="local-repo", question="Where is the DB connection created?")
)
print(response.answer)
```

### Public GitHub repository

```python
from codexpert import AgentConfig, CodexpertAgent, QueryRequest, RepoSpec

config = AgentConfig()
agent = CodexpertAgent(config)

repo = RepoSpec.from_source("https://github.com/owner/repo", repo_id="owner-repo")
await agent.async_ingest(repo)

response = await agent.async_query(
	QueryRequest(repo_id="owner-repo", question="How is authentication handled?")
)
print(response.answer)
```

## LLM configuration

By default, Codexpert expects Gemini. Set your API key:

```bash
export GEMINI_API_KEY="..."
```

To use OpenAI:

```python
from codexpert.core.config import AgentConfig, LLMConfig, LLMProvider

config = AgentConfig(
	llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
)
```

Set `OPENAI_API_KEY` in your environment.

To use Mistral:

```python
from codexpert.core.config import AgentConfig, LLMConfig, LLMProvider

config = AgentConfig(
	llm=LLMConfig(provider=LLMProvider.MISTRAL, model="mistral-large-latest")
)
```

Set `MISTRAL_API_KEY` in your environment.

`.env` example:

```bash
CODEXPERT_LLM_PROVIDER=mistral
CODEXPERT_LLM_MODEL=mistral-large-latest
MISTRAL_API_KEY="..."
```

The demo scripts load variables from `.env` automatically (see `.env.example`).

## LangSmith tracing

Codexpert uses LangChain and LangGraph runnables, so LangSmith tracing works through
standard LangSmith environment variables. No code changes are required in callers:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT=codexpert
```

The demo scripts load these values from `.env`. Query runs are named
`codexpert.query` and include `repo_id`, `top_k`, and `max_iterations` metadata.
You can add default tags with `AgentConfig(trace_tags=["codexpert", "dev"])`, and
pass per-request metadata with `QueryRequest(..., metadata={"user_id": "..."})`.

## Demo scripts

Local repository demo:

```bash
uv run python scripts/demo_local.py /path/to/repo \
  --question "Where is the main entry point?"
```

Public GitHub demo:

```bash
uv run python scripts/demo_github.py https://github.com/owner/repo \
  --question "How is authentication handled?"
```

## Citations

Responses include citations in the format `[path:start-end]` based on the indexed source chunks.
