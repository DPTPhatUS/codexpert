from __future__ import annotations

from codexpert.core.config import AgentConfig
from codexpert.memory.chunking import CodeChunk
from codexpert.memory import indexing


def test_vector_metadata_uses_chroma_safe_values(monkeypatch) -> None:
    captured: dict = {}

    class FakeCollection:
        def add(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakeStorage:
        def get_chroma_collection(self, embedding_function, create: bool = True):
            return FakeCollection()

    monkeypatch.setattr(indexing, "build_embedding_function", lambda config: object())
    chunks = [
        CodeChunk(
            chunk_id="chunk-1",
            path="main.py",
            start_line=1,
            end_line=1,
            text="print('x')",
            symbol=None,
            language=None,
        )
    ]

    indexing._build_vector_store(chunks, FakeStorage(), AgentConfig())

    assert captured["metadatas"] == [
        {
            "path": "main.py",
            "start_line": 1,
            "end_line": 1,
            "symbol": "",
            "language": "",
        }
    ]
