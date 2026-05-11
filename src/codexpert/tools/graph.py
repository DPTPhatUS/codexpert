from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from codexpert.core.types import ToolObservation
from codexpert.memory.storage import RepoStorage
from codexpert.tools.base import ToolContext
from codexpert.tools.base import ToolHandler
from codexpert.tools.base import derive_query
from codexpert.tools.base import run_tool_with_runtime


def dependency_graph_query(repo_index, query: str, limit: int = 10) -> str:
    storage: RepoStorage = repo_index.storage
    graph = storage.load_graph()
    matches = [node for node in graph.nodes if query in str(node)]
    if not matches:
        return "no matching nodes"

    lines = []
    for node in matches[:limit]:
        neighbors = list(graph.successors(node)) + list(graph.predecessors(node))
        preview = ", ".join(str(n) for n in neighbors[:limit])
        lines.append(f"{node} -> {preview}")
    return "\n".join(lines)


async def _dependency_graph_query_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    notes = await asyncio.to_thread(dependency_graph_query, context.repo_index, query)
    return ToolObservation(tool="dependency_graph_query", query=query, notes=notes)


ACTION_HANDLERS: dict[str, ToolHandler] = {
    "dependency_graph_query": _dependency_graph_query_handler,
}


def make_tools(context: ToolContext) -> list:
    @tool("dependency_graph_query")
    async def dependency_graph_query_tool(query: str, runtime: ToolRuntime) -> Command:
        """Query repository graph relationships."""
        return await run_tool_with_runtime(
            "dependency_graph_query",
            runtime,
            _dependency_graph_query_handler,
            context,
            query=query,
        )

    return [dependency_graph_query_tool]
