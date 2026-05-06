from __future__ import annotations

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from codexpert.core.config import EmbeddingConfig


def build_embedding_function(config: EmbeddingConfig):
    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbeddingFunction(model_name=config.model)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
