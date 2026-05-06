from __future__ import annotations

from codexpert.core.types import CodeSnippet
from codexpert.memory.indexing import BM25Index
from codexpert.utils.text import simple_tokenize


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
