from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from codexpert.core.config import AgentConfig
from codexpert.memory.chunking import CodeChunk
from codexpert.memory.embeddings import build_embedding_function
from codexpert.memory.graph import build_dependency_graph
from codexpert.memory.storage import RepoStorage
from codexpert.utils.text import simple_tokenize


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]


def build_indexes(
    chunks: list[CodeChunk],
    repo_root: Path,
    storage: RepoStorage,
    config: AgentConfig,
) -> None:
    storage.save_chunks(chunks)
    _build_bm25(chunks, storage)
    _build_graph(repo_root, chunks, storage)
    if config.enable_vector_index:
        _build_vector_store(chunks, storage, config)


def _build_bm25(chunks: list[CodeChunk], storage: RepoStorage) -> None:
    tokenized = [simple_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    index = BM25Index(bm25=bm25, chunk_ids=[chunk.chunk_id for chunk in chunks])
    storage.save_bm25(index)


def _build_vector_store(
    chunks: list[CodeChunk],
    storage: RepoStorage,
    config: AgentConfig,
) -> None:
    embedding_fn = build_embedding_function(config.embedding)
    collection = storage.get_chroma_collection(embedding_fn, create=True)
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol or "",
            "language": chunk.language or "",
        }
        for chunk in chunks
    ]
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=documents,
        metadatas=metadatas,
    )


def _build_graph(repo_root: Path, chunks: list[CodeChunk], storage: RepoStorage) -> None:
    files = sorted({repo_root / chunk.path for chunk in chunks})
    graph = build_dependency_graph(repo_root, files)
    storage.save_graph(graph)
