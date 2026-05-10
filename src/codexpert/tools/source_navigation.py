from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

from codexpert.core.types import CodeSnippet
from codexpert.memory.chunking import EXCLUDED_DIRS, LANGUAGE_BY_EXTENSION, iter_source_files
from codexpert.utils.text import read_text, simple_tokenize

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


def git_history_search(
    repo_index,
    query: str | None = None,
    path: str | None = None,
    symbol: str | None = None,
    max_results: int = 10,
) -> list[CodeSnippet]:
    if not (repo_index.repo_root / ".git").exists():
        return []

    command = [
        "git",
        "log",
        f"--max-count={max_results}",
        "--date=short",
        "--pretty=%h %ad %s",
    ]
    if symbol:
        command.append(f"-S{symbol}")
    if query:
        command.extend(["--grep", query])
    if path:
        command.extend(["--", path])

    result = subprocess.run(
        command,
        cwd=repo_index.repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    return [
        CodeSnippet(
            path=path or ".git",
            start_line=index,
            end_line=index,
            text=line,
            language="git",
        )
        for index, line in enumerate(result.stdout.splitlines(), start=1)
    ]


def _safe_repo_path(repo_root: Path, path: str) -> Path:
    candidate = (repo_root / path).resolve()
    root = repo_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {path}")
    return candidate


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


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
