from __future__ import annotations

import asyncio

from codexpert.core.types import ToolObservation
from codexpert.tools.graph_query import graph_query
from codexpert.tools.lexical_search import lexical_search
from codexpert.tools.ripgrep_search import ripgrep_search
from codexpert.tools.semantic_search import semantic_search


class ToolRegistry:
    def __init__(self, repo_index, config) -> None:
        self.repo_index = repo_index
        self.config = config

    async def run(self, action: str, query: str, top_k: int) -> ToolObservation:
        if action == "semantic_search":
            snippets = await asyncio.to_thread(
                semantic_search, self.repo_index, query, top_k, self.config
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "lexical_search":
            snippets = await asyncio.to_thread(
                lexical_search, self.repo_index, query, top_k
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "ripgrep_search":
            snippets = await asyncio.to_thread(
                ripgrep_search, self.repo_index, query, top_k, self.config
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "graph_query":
            notes = await asyncio.to_thread(graph_query, self.repo_index, query)
            return ToolObservation(tool=action, query=query, notes=notes)

        return ToolObservation(tool=action, query=query, notes="unknown tool")
