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
from codexpert.tools.source_navigation import (
    definition_lookup,
    doc_search,
    git_history_search,
    grep_search,
    list_files,
    read_file,
    references_search,
    symbol_search,
)


class ToolRegistry:
    def __init__(self, repo_index, config) -> None:
        self.repo_index = repo_index
        self.config = config

    async def run(self, action: str, query: str, top_k: int) -> ToolObservation:
        return await self._run_action(action, query=query, top_k=top_k)

    def as_langchain_tools(self) -> list:
        @tool("list_files")
        async def list_files_tool(
            runtime: ToolRuntime,
            path: str = "",
            pattern: str = "*",
            max_results: int = 100,
        ) -> Command:
            """List repository files by path and glob pattern."""
            return await self._run_with_runtime(
                "list_files",
                runtime,
                path=path,
                pattern=pattern,
                max_results=max_results,
            )

        @tool("read_file")
        async def read_file_tool(
            path: str,
            runtime: ToolRuntime,
            start_line: int | None = None,
            end_line: int | None = None,
        ) -> Command:
            """Read a repository file or line range."""
            return await self._run_with_runtime(
                "read_file",
                runtime,
                path=path,
                start_line=start_line,
                end_line=end_line,
            )

        @tool("grep_search")
        async def grep_search_tool(
            pattern: str,
            runtime: ToolRuntime,
            path_glob: str | None = None,
            regex: bool = True,
            case_sensitive: bool = False,
        ) -> Command:
            """Search source files by exact text or regex."""
            return await self._run_with_runtime(
                "grep_search",
                runtime,
                pattern=pattern,
                path_glob=path_glob,
                regex=regex,
                case_sensitive=case_sensitive,
            )

        @tool("semantic_search")
        async def semantic_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Semantic code search."""
            return await self._run_with_runtime("semantic_search", runtime, query=query)

        @tool("lexical_search")
        async def lexical_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Lexical BM25 code search."""
            return await self._run_with_runtime("lexical_search", runtime, query=query)

        @tool("ripgrep_search")
        async def ripgrep_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Ripgrep-based code search."""
            return await self._run_with_runtime("ripgrep_search", runtime, query=query)

        @tool("symbol_search")
        async def symbol_search_tool(
            query: str,
            runtime: ToolRuntime,
            kind: str | None = None,
            path_glob: str | None = None,
        ) -> Command:
            """Search indexed code symbols such as functions and classes."""
            return await self._run_with_runtime(
                "symbol_search",
                runtime,
                query=query,
                kind=kind,
                path_glob=path_glob,
            )

        @tool("definition_lookup")
        async def definition_lookup_tool(
            symbol: str,
            runtime: ToolRuntime,
            path_hint: str | None = None,
        ) -> Command:
            """Find definitions for a symbol."""
            return await self._run_with_runtime(
                "definition_lookup",
                runtime,
                symbol=symbol,
                path_hint=path_hint,
            )

        @tool("references_search")
        async def references_search_tool(
            symbol: str,
            runtime: ToolRuntime,
            path_hint: str | None = None,
        ) -> Command:
            """Find references to a symbol in source files."""
            return await self._run_with_runtime(
                "references_search",
                runtime,
                symbol=symbol,
                path_hint=path_hint,
            )

        @tool("dependency_graph_query")
        async def dependency_graph_query_tool(query: str, runtime: ToolRuntime) -> Command:
            """Query repository graph relationships."""
            return await self._run_with_runtime(
                "dependency_graph_query", runtime, query=query
            )

        @tool("doc_search")
        async def doc_search_tool(query: str, runtime: ToolRuntime) -> Command:
            """Search repository docs and config files as evidence."""
            return await self._run_with_runtime("doc_search", runtime, query=query)

        @tool("git_history_search")
        async def git_history_search_tool(
            runtime: ToolRuntime,
            query: str | None = None,
            path: str | None = None,
            symbol: str | None = None,
        ) -> Command:
            """Search local Git history for commits touching a path, symbol, or topic."""
            return await self._run_with_runtime(
                "git_history_search",
                runtime,
                query=query,
                path=path,
                symbol=symbol,
            )

        return [
            list_files_tool,
            read_file_tool,
            grep_search_tool,
            semantic_search_tool,
            lexical_search_tool,
            ripgrep_search_tool,
            symbol_search_tool,
            definition_lookup_tool,
            references_search_tool,
            dependency_graph_query_tool,
            doc_search_tool,
            git_history_search_tool,
        ]

    async def _run_action(self, action: str, **kwargs) -> ToolObservation:
        query = str(
            kwargs.get("query")
            or kwargs.get("pattern")
            or kwargs.get("symbol")
            or kwargs.get("path")
            or ""
        )
        top_k = int(kwargs.get("top_k", self.config.top_k))
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
        if action == "list_files":
            snippets = await asyncio.to_thread(
                list_files,
                self.repo_index,
                kwargs.get("path") or "",
                kwargs.get("pattern") or "*",
                int(kwargs.get("max_results") or 100),
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "read_file":
            snippets = await asyncio.to_thread(
                read_file,
                self.repo_index,
                kwargs.get("path") or "",
                kwargs.get("start_line"),
                kwargs.get("end_line"),
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "grep_search":
            snippets = await asyncio.to_thread(
                grep_search,
                self.repo_index,
                kwargs.get("pattern") or "",
                kwargs.get("path_glob"),
                bool(kwargs.get("regex", True)),
                bool(kwargs.get("case_sensitive", False)),
                top_k,
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "symbol_search":
            snippets = await asyncio.to_thread(
                symbol_search,
                self.repo_index,
                query,
                kwargs.get("kind"),
                kwargs.get("path_glob"),
                top_k,
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "definition_lookup":
            snippets = await asyncio.to_thread(
                definition_lookup,
                self.repo_index,
                kwargs.get("symbol") or "",
                kwargs.get("path_hint"),
                top_k,
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "references_search":
            snippets = await asyncio.to_thread(
                references_search,
                self.repo_index,
                kwargs.get("symbol") or "",
                kwargs.get("path_hint"),
                top_k,
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "graph_query":
            notes = await asyncio.to_thread(graph_query, self.repo_index, query)
            return ToolObservation(tool=action, query=query, notes=notes)
        if action == "dependency_graph_query":
            notes = await asyncio.to_thread(graph_query, self.repo_index, query)
            return ToolObservation(tool=action, query=query, notes=notes)
        if action == "doc_search":
            snippets = await asyncio.to_thread(doc_search, self.repo_index, query, top_k)
            return ToolObservation(tool=action, query=query, snippets=snippets)
        if action == "git_history_search":
            snippets = await asyncio.to_thread(
                git_history_search,
                self.repo_index,
                kwargs.get("query"),
                kwargs.get("path"),
                kwargs.get("symbol"),
                top_k,
            )
            return ToolObservation(tool=action, query=query, snippets=snippets)

        return ToolObservation(tool=action, query=query, notes="unknown tool")

    async def _run_with_runtime(
        self, action: str, runtime: ToolRuntime | None, **kwargs
    ) -> Command:
        state = runtime.state if runtime else {}
        top_k = state.get("top_k", self.config.top_k)
        iteration = state.get("iterations", 0)
        query = str(
            kwargs.get("query")
            or kwargs.get("pattern")
            or kwargs.get("symbol")
            or kwargs.get("path")
            or ""
        )
        try:
            observation = await self._run_action(action, **kwargs, top_k=top_k)
        except Exception as exc:
            observation = ToolObservation(
                tool=action,
                query=query,
                notes=f"{type(exc).__name__}: {exc}",
            )
        trace_step = ExecutionTraceStep(
            iteration=iteration,
            action=action,
            query=observation.query,
            result_count=len(observation.snippets),
            notes=observation.notes,
        )
        tool_message = ToolMessage(
            tool_call_id=runtime.tool_call_id if runtime else "",
            content=self._render_tool_message(observation),
        )
        return Command(
            update={
                "observations": [observation],
                "evidence_set": observation.snippets,
                "trace": [trace_step],
                "iterations": 1,
                "messages": [tool_message],
            }
        )

    def _render_tool_message(self, observation: ToolObservation) -> str:
        max_chars = int(getattr(self.config, "max_tool_message_chars", 4000))
        if observation.snippets:
            lines = [
                f"{observation.tool} "
                f"hits={len(observation.snippets)} "
                f"query={observation.query}"
            ]
            for index, snippet in enumerate(observation.snippets[:5], start=1):
                label = f"{index}. {snippet.path}:{snippet.start_line}-{snippet.end_line}"
                metadata = []
                if snippet.symbol:
                    metadata.append(f"symbol={snippet.symbol}")
                if snippet.language:
                    metadata.append(f"language={snippet.language}")
                if metadata:
                    label = f"{label} ({', '.join(metadata)})"
                lines.append(label)
                if snippet.text:
                    lines.append(_indent(_truncate(snippet.text, 900)))
            return _truncate("\n".join(lines), max_chars)
        if observation.notes:
            return _truncate(
                f"{observation.tool} query={observation.query} notes={observation.notes}",
                max_chars,
            )
        return f"{observation.tool} no_results"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...<truncated>"


def _indent(text: str) -> str:
    return "\n".join(f"   {line}" for line in text.splitlines())
