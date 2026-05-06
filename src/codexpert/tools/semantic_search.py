from __future__ import annotations

from codexpert.core.types import CodeSnippet
from codexpert.memory.embeddings import build_embedding_function


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
        include=["documents", "metadatas", "ids"],
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
