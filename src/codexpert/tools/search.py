from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from codexpert.core.types import CodeSnippet
from codexpert.core.types import ToolObservation
from codexpert.memory.chunking import LANGUAGE_BY_EXTENSION
from codexpert.memory.chunking import iter_source_files
from codexpert.memory.embeddings import build_embedding_function
from codexpert.memory.indexing import BM25Index
from codexpert.tools.base import ToolContext
from codexpert.tools.base import ToolHandler
from codexpert.tools.base import derive_query
from codexpert.tools.base import run_tool_with_runtime
from codexpert.utils.text import read_text
from codexpert.utils.text import simple_tokenize


def semantic_search(repo_index, query: str, top_k: int, config) -> list[CodeSnippet]:
    if not config.enable_vector_index:
        return []
    embedding_fn = build_embedding_function(config.embedding)
    try:
        collection = repo_index.storage.get_chroma_collection(embedding_fn, create=False)
    except Exception:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    snippets: list[CodeSnippet] = []
    for doc, metadata in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
        snippets.append(
            CodeSnippet(
                path=metadata.get("path", ""),
                start_line=metadata.get("start_line", 1),
                end_line=metadata.get("end_line", 1),
                text=doc,
                symbol=metadata.get("symbol"),
                language=metadata.get("language"),
            )
        )
    return snippets


def lexical_search(repo_index, query: str, top_k: int) -> list[CodeSnippet]:
    bm25_index: BM25Index = repo_index.storage.load_bm25()
    tokens = simple_tokenize(query)
    scores = bm25_index.bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    chunks = repo_index.storage.load_chunks()
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    snippets: list[CodeSnippet] = []
    for idx, _score in ranked:
        chunk_id = bm25_index.chunk_ids[idx]
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        snippets.append(
            CodeSnippet(
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
                symbol=chunk.symbol,
                language=chunk.language,
            )
        )
    return snippets


def ripgrep_search(repo_index, query: str, top_k: int, config) -> list[CodeSnippet]:
    if not config.ripgrep_enabled:
        return []
    if not shutil.which("rg"):
        raise RuntimeError("ripgrep (rg) is required but not installed")

    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--max-count",
        str(top_k),
        query,
        ".",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_index.repo_root),
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"rg failed: {result.stderr}")

    snippets: list[CodeSnippet] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, line_number, text = parts
        if path.startswith("./"):
            path = path[2:]
        snippets.append(
            CodeSnippet(
                path=path,
                start_line=int(line_number),
                end_line=int(line_number),
                text=text,
            )
        )
    return snippets


def grep_search(
    repo_index,
    pattern: str,
    path_glob: str | None = None,
    regex: bool = True,
    case_sensitive: bool = False,
    max_results: int = 20,
) -> list[CodeSnippet]:
    flags = 0 if case_sensitive else re.IGNORECASE
    needle = re.compile(pattern if regex else re.escape(pattern), flags)
    snippets: list[CodeSnippet] = []
    for info in iter_source_files(repo_index.repo_root):
        rel = info.path.relative_to(repo_index.repo_root).as_posix()
        if path_glob and not fnmatch.fnmatch(rel, path_glob):
            continue
        snippets.extend(_matching_lines(info.path, repo_index.repo_root, needle))
        if len(snippets) >= max_results:
            return snippets[:max_results]
    return snippets


async def _semantic_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(
        semantic_search, context.repo_index, query, top_k, context.config
    )
    return ToolObservation(tool="semantic_search", query=query, snippets=snippets)


async def _lexical_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(lexical_search, context.repo_index, query, top_k)
    return ToolObservation(tool="lexical_search", query=query, snippets=snippets)


async def _ripgrep_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(
        ripgrep_search, context.repo_index, query, top_k, context.config
    )
    return ToolObservation(tool="ripgrep_search", query=query, snippets=snippets)


async def _grep_search_handler(
    context: ToolContext, kwargs: dict[str, Any]
) -> ToolObservation:
    query = derive_query(kwargs)
    top_k = int(kwargs.get("top_k", context.config.top_k))
    snippets = await asyncio.to_thread(
        grep_search,
        context.repo_index,
        kwargs.get("pattern") or "",
        kwargs.get("path_glob"),
        bool(kwargs.get("regex", True)),
        bool(kwargs.get("case_sensitive", False)),
        top_k,
    )
    return ToolObservation(tool="grep_search", query=query, snippets=snippets)


ACTION_HANDLERS: dict[str, ToolHandler] = {
    "grep_search": _grep_search_handler,
    "semantic_search": _semantic_search_handler,
    "lexical_search": _lexical_search_handler,
    "ripgrep_search": _ripgrep_search_handler,
}


def make_tools(context: ToolContext) -> list:
    @tool("grep_search")
    async def grep_search_tool(
        pattern: str,
        runtime: ToolRuntime,
        path_glob: str | None = None,
        regex: bool = True,
        case_sensitive: bool = False,
    ) -> Command:
        """Search source files by exact text or regex."""
        return await run_tool_with_runtime(
            "grep_search",
            runtime,
            _grep_search_handler,
            context,
            pattern=pattern,
            path_glob=path_glob,
            regex=regex,
            case_sensitive=case_sensitive,
        )

    @tool("semantic_search")
    async def semantic_search_tool(query: str, runtime: ToolRuntime) -> Command:
        """Semantic code search."""
        return await run_tool_with_runtime(
            "semantic_search", runtime, _semantic_search_handler, context, query=query
        )

    @tool("lexical_search")
    async def lexical_search_tool(query: str, runtime: ToolRuntime) -> Command:
        """Lexical BM25 code search."""
        return await run_tool_with_runtime(
            "lexical_search", runtime, _lexical_search_handler, context, query=query
        )

    @tool("ripgrep_search")
    async def ripgrep_search_tool(query: str, runtime: ToolRuntime) -> Command:
        """Ripgrep-based code search."""
        return await run_tool_with_runtime(
            "ripgrep_search", runtime, _ripgrep_search_handler, context, query=query
        )

    return [
        grep_search_tool,
        semantic_search_tool,
        lexical_search_tool,
        ripgrep_search_tool,
    ]


def _matching_lines(path: Path, repo_root: Path, pattern: re.Pattern[str]) -> list[CodeSnippet]:
    text = read_text(path, max_chars=250000)
    rel = path.relative_to(repo_root).as_posix()
    snippets: list[CodeSnippet] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not pattern.search(line):
            continue
        snippets.append(
            CodeSnippet(
                path=rel,
                start_line=line_number,
                end_line=line_number,
                text=line,
                language=LANGUAGE_BY_EXTENSION.get(path.suffix.lower()),
            )
        )
    return snippets
