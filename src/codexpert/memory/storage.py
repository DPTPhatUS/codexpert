from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx
from networkx.readwrite import json_graph

from codexpert.memory.chunking import CodeChunk

_CHROMA_CLIENTS: dict[Path, chromadb.PersistentClient] = {}


class RepoStorage:
    def __init__(self, base_dir: Path, repo_id: str) -> None:
        self.repo_id = repo_id
        self.repo_dir = base_dir / repo_id
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    @property
    def chroma_dir(self) -> Path:
        return self.repo_dir / "chroma"

    @property
    def chunks_path(self) -> Path:
        return self.repo_dir / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.repo_dir / "bm25.pkl"

    @property
    def graph_path(self) -> Path:
        return self.repo_dir / "graph.json"

    @property
    def meta_path(self) -> Path:
        return self.repo_dir / "repo.json"

    def has_index(self) -> bool:
        return (
            self.chunks_path.exists()
            and self.bm25_path.exists()
            and self.graph_path.exists()
            and self.chroma_dir.exists()
        )

    def save_chunks(self, chunks: list[CodeChunk]) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.model_dump()) + "\n")

    def load_chunks(self) -> list[CodeChunk]:
        if not self.chunks_path.exists():
            return []
        chunks: list[CodeChunk] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunks.append(CodeChunk.model_validate_json(line))
        return chunks

    def save_bm25(self, bm25_index: Any) -> None:
        with self.bm25_path.open("wb") as handle:
            pickle.dump(bm25_index, handle)

    def load_bm25(self) -> Any:
        with self.bm25_path.open("rb") as handle:
            return pickle.load(handle)

    def save_graph(self, graph: nx.DiGraph) -> None:
        data = json_graph.node_link_data(graph)
        with self.graph_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def load_graph(self) -> nx.DiGraph:
        with self.graph_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return json_graph.node_link_graph(data, directed=True)

    def save_meta(self, payload: dict[str, Any]) -> None:
        with self.meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def load_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_chroma_collection(self, embedding_function, create: bool = True):
        client = _get_chroma_client(self.chroma_dir)
        collection_name = f"codexpert_{self.repo_id}"
        if create:
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            return client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )


def _get_chroma_client(path: Path) -> chromadb.PersistentClient:
    if path not in _CHROMA_CLIENTS:
        path.mkdir(parents=True, exist_ok=True)
        _CHROMA_CLIENTS[path] = chromadb.PersistentClient(path=str(path))
    return _CHROMA_CLIENTS[path]
