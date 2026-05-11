from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from codexpert.core.types import CodeSnippet
from codexpert.core.types import ToolObservation
from codexpert.memory.chunking import EXCLUDED_DIRS
from codexpert.memory.chunking import LANGUAGE_BY_EXTENSION
from codexpert.tools.base import ToolContext
from codexpert.tools.base import ToolHandler
from codexpert.tools.base import derive_query
from codexpert.tools.base import run_tool_with_runtime
from codexpert.tools.search import grep_search
from codexpert.utils.text import read_text
from codexpert.utils.text import simple_tokenize

DOC_EXTENSIONS = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
}


def list_files(
    repo_index,
    path: str = "",
    pattern: str = "*",
    max_results: int = 100,
) -> list[CodeSnippet]:
    root = _safe_repo_path(repo_index.repo_root, path or ".")
    if not root.exists():
        return []

    files: list[Path] = []
    candidates = root.rglob("*") if root.is_dir() else [root]
    for candidate in candidates:
        if len(files) >= max_results:
            break
        if candidate.is_dir() or _is_excluded(candidate):
            continue
        rel = candidate.relative_to(repo_index.repo_root).as_posix()
        if fnmatch.fnmatch(candidate.name, pattern) or fnmatch.fnmatch(rel, pattern):
            files.append(candidate)

    return [
        CodeSnippet(
            path=file.relative_to(repo_index.repo_root).as_posix(),
            start_line=1,
            end_line=1,
            text="<file>",
            language=LANGUAGE_BY_EXTENSION.get(file.suffix.lower()),
        )
        for file in sorted(files)
    ]


def read_file(
    repo_index,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[CodeSnippet]:
    file_path = _safe_repo_path(repo_index.repo_root, path)
    if not file_path.is_file():
        return []

    text = read_text(file_path, max_chars=500000)
    lines = text.splitlines()
    if not lines:
        return []

    start = max(start_line or 1, 1)
    end = min(end_line or len(lines), len(lines))
    if end < start:
        return []

    snippet_text = "\n".join(lines[start - 1 : end])
    return [
        CodeSnippet(
            path=file_path.relative_to(repo_index.repo_root).as_posix(),
            start_line=start,
            end_line=end,
            text=snippet_text,
            language=LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower()),
        )
    ]


def doc_search(repo_index, query: str, max_results: int = 20) -> list[CodeSnippet]:
    terms = simple_tokenize(query)
    if not terms:
        return []

    snippets: list[CodeSnippet] = []
    for path in sorted(repo_index.repo_root.rglob("*")):
        if len(snippets) >= max_results:
            break
        if path.is_dir() or _is_excluded(path) or path.suffix.lower() not in DOC_EXTENSIONS:
            continue
        text = read_text(path, max_chars=250000)
        lower = text.lower()
        if not all(term in lower for term in terms[:3]):
            continue
        snippets.extend(_query_windows(path, repo_index.repo_root, terms))
    return snippets[:max_results]


def symbol_search(
    repo_index,
    query: str,
    kind: str | None = None,
    path_glob: str | None = None,
    max_results: int = 20,
) -> list[CodeSnippet]:
    del kind
    query_lower = query.lower()
    matches: list[CodeSnippet] = []
    for chunk in repo_index.storage.load_chunks():
        if path_glob and not fnmatch.fnmatch(chunk.path, path_glob):
            continue
        symbol = chunk.symbol or ""
        if query_lower in symbol.lower():
            matches.append(
                CodeSnippet(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    symbol=chunk.symbol,
                    language=chunk.language,
                )
            )
        if len(matches) >= max_results:
            break
    return matches


def definition_lookup(
    repo_index,
    symbol: str,
    path_hint: str | None = None,
    max_results: int = 10,
) -> list[CodeSnippet]:
    matches = symbol_search(
        repo_index,
        symbol,
        path_glob=path_hint,
        max_results=max_results,
    )
    exact = [match for match in matches if match.symbol == symbol]
    return exact or matches


def references_search(
    repo_index,
    symbol: str,
    path_hint: str | None = None,
    max_results: int = 20,
) -> list[CodeSnippet]:
    escaped = re.escape(symbol)
    pattern = rf"\b{escaped}\b"
    return grep_search(
        repo_index,
        pattern=pattern,
        path_glob=path_hint,
        regex=True,
        case_sensitive=True,
        max_results=max_results,
    )


async def _list_files_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    snippets = await asyncio.to_thread(
        list_files,
        context.repo_index,
        kwargs.get("path") or "",
        kwargs.get("pattern") or "*",
        int(kwargs.get("max_results") or 100),
    )
    return ToolObservation(tool="list_files", query=query, snippets=snippets)


async def _read_file_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    snippets = await asyncio.to_thread(
        read_file,
        context.repo_index,
        kwargs.get("path") or "",
        kwargs.get("start_line"),
        kwargs.get("end_line"),
    )
    return ToolObservation(tool="read_file", query=query, snippets=snippets)


async def _symbol_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(
        symbol_search,
        context.repo_index,
        query,
        kwargs.get("kind"),
        kwargs.get("path_glob"),
        top_k,
    )
    return ToolObservation(tool="symbol_search", query=query, snippets=snippets)


async def _definition_lookup_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(
        definition_lookup,
        context.repo_index,
        kwargs.get("symbol") or "",
        kwargs.get("path_hint"),
        top_k,
    )
    return ToolObservation(tool="definition_lookup", query=query, snippets=snippets)


async def _references_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(
        references_search,
        context.repo_index,
        kwargs.get("symbol") or "",
        kwargs.get("path_hint"),
        top_k,
    )
    return ToolObservation(tool="references_search", query=query, snippets=snippets)


async def _doc_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(doc_search, context.repo_index, query, top_k)
    return ToolObservation(tool="doc_search", query=query, snippets=snippets)


ACTION_HANDLERS: dict[str, ToolHandler] = {
    "list_files": _list_files_handler,
    "read_file": _read_file_handler,
    "symbol_search": _symbol_search_handler,
    "definition_lookup": _definition_lookup_handler,
    "references_search": _references_search_handler,
    "doc_search": _doc_search_handler,
}


def make_tools(context: ToolContext) -> list:
    @tool("list_files")
    async def list_files_tool(
        runtime: ToolRuntime,
        path: str = "",
        pattern: str = "*",
        max_results: int = 100,
    ) -> Command:
        """List repository files by path and glob pattern."""
        return await run_tool_with_runtime(
            "list_files",
            runtime,
            _list_files_handler,
            context,
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
        return await run_tool_with_runtime(
            "read_file",
            runtime,
            _read_file_handler,
            context,
            path=path,
            start_line=start_line,
            end_line=end_line,
        )

    @tool("symbol_search")
    async def symbol_search_tool(
        query: str,
        runtime: ToolRuntime,
        kind: str | None = None,
        path_glob: str | None = None,
    ) -> Command:
        """Search indexed code symbols such as functions and classes."""
        return await run_tool_with_runtime(
            "symbol_search",
            runtime,
            _symbol_search_handler,
            context,
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
        return await run_tool_with_runtime(
            "definition_lookup",
            runtime,
            _definition_lookup_handler,
            context,
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
        return await run_tool_with_runtime(
            "references_search",
            runtime,
            _references_search_handler,
            context,
            symbol=symbol,
            path_hint=path_hint,
        )

    @tool("doc_search")
    async def doc_search_tool(query: str, runtime: ToolRuntime) -> Command:
        """Search repository docs and config files as evidence."""
        return await run_tool_with_runtime(
            "doc_search", runtime, _doc_search_handler, context, query=query
        )

    return [
        list_files_tool,
        read_file_tool,
        symbol_search_tool,
        definition_lookup_tool,
        references_search_tool,
        doc_search_tool,
    ]


def _safe_repo_path(repo_root: Path, path: str) -> Path:
    candidate = (repo_root / path).resolve()
    root = repo_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {path}")
    return candidate


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _query_windows(path: Path, repo_root: Path, terms: list[str]) -> list[CodeSnippet]:
    lines = read_text(path, max_chars=250000).splitlines()
    rel = path.relative_to(repo_root).as_posix()
    snippets: list[CodeSnippet] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        if not any(term in lower for term in terms):
            continue
        start = max(index - 1, 0)
        end = min(index + 2, len(lines))
        snippets.append(
            CodeSnippet(
                path=rel,
                start_line=start + 1,
                end_line=end,
                text="\n".join(lines[start:end]),
            )
        )
    return snippets
