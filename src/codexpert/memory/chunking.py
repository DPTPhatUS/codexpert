from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tree_sitter_languages import get_parser

from codexpert.core.config import ChunkingConfig
from codexpert.utils.paths import repo_relative_path
from codexpert.utils.text import chunk_text_by_chars, read_text


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cs": "c_sharp",
}

TARGET_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "method_definition", "class_declaration"},
    "typescript": {"function_declaration", "method_definition", "class_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {"method_declaration", "class_declaration", "struct_declaration"},
}

NAME_NODE_TYPES = {"identifier", "name", "property_identifier", "type_identifier"}

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
}


class CodeChunk(BaseModel):
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class FileInfo:
    path: Path
    language: str


def iter_source_files(repo_root: Path) -> list[FileInfo]:
    files: list[FileInfo] = []
    for path in repo_root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDED_DIRS:
                continue
            continue
        if _is_excluded(path):
            continue
        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if not language:
            continue
        files.append(FileInfo(path=path, language=language))
    return files


def chunk_repository(repo_root: Path, repo_id: str, config: ChunkingConfig) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for info in iter_source_files(repo_root):
        chunks.extend(chunk_file(info.path, repo_root, repo_id, config, info.language))
    return chunks


def chunk_file(
    path: Path,
    repo_root: Path,
    repo_id: str,
    config: ChunkingConfig,
    language: str | None = None,
) -> list[CodeChunk]:
    language = language or LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
    if not language:
        return []

    source = read_text(path, config.max_file_chars)
    if not source.strip():
        return []

    source_bytes = source.encode("utf-8")
    parser = get_parser(language)
    tree = parser.parse(source_bytes)
    nodes = _collect_nodes(tree.root_node, language)

    rel_path = repo_relative_path(repo_root, path)
    chunks: list[CodeChunk] = []
    for node in nodes:
        chunk_text = source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        )
        if not chunk_text.strip():
            continue
        chunk_id = _chunk_id(repo_id, rel_path, node.start_point[0] + 1, node.end_point[0] + 1)
        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                text=chunk_text,
                symbol=_extract_symbol(node, source_bytes),
                language=language,
            )
        )

    if not chunks:
        return _fallback_chunks(source, repo_id, rel_path, config, language)

    return chunks


def _fallback_chunks(
    source: str,
    repo_id: str,
    rel_path: str,
    config: ChunkingConfig,
    language: str,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for index, chunk_text in enumerate(
        chunk_text_by_chars(source, config.max_chunk_chars, config.min_chunk_chars)
    ):
        start_line = _line_for_offset(source, source.find(chunk_text))
        end_line = start_line + chunk_text.count("\n")
        chunk_id = _chunk_id(repo_id, rel_path, start_line, end_line, suffix=str(index))
        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                path=rel_path,
                start_line=start_line,
                end_line=end_line,
                text=chunk_text,
                symbol=None,
                language=language,
            )
        )
    return chunks


def _collect_nodes(root, language: str) -> list[Any]:
    target_types = TARGET_NODE_TYPES.get(language, set())
    if not target_types:
        return []
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in target_types:
            nodes.append(node)
        stack.extend(node.children)
    return nodes


def _extract_symbol(node, source_bytes: bytes) -> str | None:
    for child in node.children:
        if child.type in NAME_NODE_TYPES:
            return source_bytes[child.start_byte : child.end_byte].decode(
                "utf-8", errors="ignore"
            )
    return None


def _chunk_id(repo_id: str, rel_path: str, start_line: int, end_line: int, suffix: str | None = None) -> str:
    base = f"{repo_id}:{rel_path}:{start_line}-{end_line}"
    if suffix:
        return f"{base}:{suffix}"
    return base


def _line_for_offset(source: str, offset: int) -> int:
    if offset <= 0:
        return 1
    return source.count("\n", 0, offset) + 1


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)
