from __future__ import annotations

import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from codexpert.core.types import ToolObservation
from codexpert.core.types import ExecutionTraceStep
from codexpert.tools.graph_query import graph_query
from codexpert.tools.lexical_search import lexical_search
from codexpert.tools.ripgrep_search import ripgrep_search
from codexpert.tools.semantic_search import semantic_search


class ToolRegistry:
    def __init__(self, repo_index, config) -> None:
        self.repo_index = repo_index
        self.config = config

    async def run(self, action: str, query: str, top_k: int) -> ToolObservation:
        return await self._run_action(action, query, top_k)

    def as_langchain_tools(self) -> list:
        @tool("semantic_search")
        async def semantic_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Semantic code search."""
            return await self._run_with_runtime("semantic_search", query, runtime)

        @tool("lexical_search")
        async def lexical_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Lexical BM25 code search."""
            return await self._run_with_runtime("lexical_search", query, runtime)

        @tool("ripgrep_search")
        async def ripgrep_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Ripgrep-based code search."""
            return await self._run_with_runtime("ripgrep_search", query, runtime)

        @tool("graph_query")
        async def graph_query_tool(query: str, runtime: ToolRuntime) -> Command:
            """Query repository graph relationships."""
            return await self._run_with_runtime("graph_query", query, runtime)

        return [
            semantic_search_tool,
            lexical_search_tool,
            ripgrep_search_tool,
            graph_query_tool,
        ]

    async def _run_action(self, action: str, query: str, top_k: int) -> ToolObservation:
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

    async def _run_with_runtime(
        self, action: str, query: str, runtime: ToolRuntime
    ) -> Command:
        top_k = runtime.state.get("top_k", self.config.top_k)
        iteration = runtime.state.get("iterations", 0)
        observation = await self._run_action(action, query, top_k)
        trace_step = ExecutionTraceStep(
            iteration=iteration,
            action=action,
            query=query,
            result_count=len(observation.snippets),
            notes=observation.notes,
        )
        tool_message = ToolMessage(
            tool_call_id=runtime.tool_call_id,
            content=self._render_tool_message(observation),
        )
        return Command(
            update={
                "observations": [observation],
                "trace": [trace_step],
                "iterations": 1,
                "messages": [tool_message],
            }
        )

    def _render_tool_message(self, observation: ToolObservation) -> str:
        if observation.snippets:
            preview = ", ".join(
                f"{snippet.path}:{snippet.start_line}-{snippet.end_line}"
                for snippet in observation.snippets[:3]
            )
            return f"{observation.tool} hits={len(observation.snippets)} {preview}"
        if observation.notes:
            return f"{observation.tool} notes={observation.notes}"
        return f"{observation.tool} no_results"
