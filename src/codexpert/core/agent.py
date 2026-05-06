from __future__ import annotations

import asyncio

from codexpert.core.config import AgentConfig, RepoSpec
from codexpert.core.orchestrator import ReActOrchestrator
from codexpert.core.types import QueryRequest, QueryResponse
from codexpert.llm.router import build_llm_client
from codexpert.memory.ingest import ensure_repo_index, load_existing_index


class CodexpertAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self._llm = build_llm_client(self.config.llm)

    async def async_ingest(self, repo: RepoSpec) -> None:
        await asyncio.to_thread(ensure_repo_index, repo, self.config)

    async def async_query(self, request: QueryRequest) -> QueryResponse:
        if request.repo:
            repo_index = await asyncio.to_thread(ensure_repo_index, request.repo, self.config)
        else:
            repo_index = load_existing_index(request.repo_id, self.config)

        orchestrator = ReActOrchestrator(self._llm, repo_index, self.config)
        return await orchestrator.run(request)

    def query(self, request: QueryRequest) -> QueryResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_query(request))
        raise RuntimeError("Use async_query() when an event loop is running")
