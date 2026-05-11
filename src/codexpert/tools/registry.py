from __future__ import annotations

from codexpert.core.types import ToolObservation
from codexpert.tools import graph
from codexpert.tools import navigation
from codexpert.tools import search
from codexpert.tools.base import ToolContext
from codexpert.tools.base import ToolHandler
from codexpert.tools.base import run_action


class ToolRegistry:
    def __init__(self, repo_index, config) -> None:
        self.repo_index = repo_index
        self.config = config
        self.context = ToolContext(repo_index=repo_index, config=config)
        self.handlers: dict[str, ToolHandler] = {
            **navigation.ACTION_HANDLERS,
            **search.ACTION_HANDLERS,
            **graph.ACTION_HANDLERS,
        }

    async def run(self, action: str, query: str, top_k: int) -> ToolObservation:
        return await run_action(self.context, self.handlers, action, query=query, top_k=top_k)

    def as_langchain_tools(self) -> list:
        return [
            *navigation.make_tools(self.context),
            *search.make_tools(self.context),
            *graph.make_tools(self.context),
        ]
