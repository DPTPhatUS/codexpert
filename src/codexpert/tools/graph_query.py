from __future__ import annotations

from codexpert.memory.storage import RepoStorage


def graph_query(repo_index, query: str, limit: int = 10) -> str:
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
