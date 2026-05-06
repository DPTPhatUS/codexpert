from __future__ import annotations

import ast
import re
from pathlib import Path

import networkx as nx

from codexpert.memory.chunking import LANGUAGE_BY_EXTENSION
from codexpert.utils.paths import repo_relative_path
from codexpert.utils.text import read_text


IMPORT_PATTERNS = {
    "javascript": re.compile(r"import\s+.*?from\s+[\"'](?P<module>[^\"']+)[\"']"),
    "typescript": re.compile(r"import\s+.*?from\s+[\"'](?P<module>[^\"']+)[\"']"),
    "c": re.compile(r"#include\s+[<\"](?P<module>[^>\"]+)[>\"]"),
    "cpp": re.compile(r"#include\s+[<\"](?P<module>[^>\"]+)[>\"]"),
    "c_sharp": re.compile(r"using\s+(?P<module>[A-Za-z0-9_.]+)\s*;")
}


def build_dependency_graph(repo_root: Path, files: list[Path]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for path in files:
        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if not language:
            continue
        rel_path = repo_relative_path(repo_root, path)
        graph.add_node(rel_path, type="file")
        dependencies = _extract_imports(path, language)
        for dep in dependencies:
            graph.add_node(dep, type="module")
            graph.add_edge(rel_path, dep, type="imports")
        if language == "python":
            _add_python_call_edges(path, rel_path, graph)
    return graph


def _extract_imports(path: Path, language: str) -> list[str]:
    source = read_text(path, max_chars=200000)
    if language == "python":
        return _python_imports(source)
    pattern = IMPORT_PATTERNS.get(language)
    if not pattern:
        return []
    return [match.group("module") for match in pattern.finditer(source)]


def _python_imports(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _add_python_call_edges(path: Path, rel_path: str, graph: nx.DiGraph) -> None:
    source = read_text(path, max_chars=200000)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    function_defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        caller = f"{rel_path}::{node.name}"
        graph.add_node(caller, type="function")
        for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
            callee = _call_name(call)
            if callee and callee in function_defs:
                graph.add_node(f"{rel_path}::{callee}", type="function")
                graph.add_edge(caller, f"{rel_path}::{callee}", type="calls")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None
