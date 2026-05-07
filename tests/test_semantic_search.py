from __future__ import annotations

from types import SimpleNamespace

from codexpert.tools import semantic_search as semantic_module


def test_semantic_search_uses_chroma_supported_include(monkeypatch) -> None:
    captured: dict = {}

    class FakeCollection:
        def query(self, **kwargs):
            captured.update(kwargs)
            return {
                "documents": [["text"]],
                "metadatas": [
                    [
                        {
                            "path": "main.py",
                            "start_line": 1,
                            "end_line": 1,
                            "symbol": "",
                            "language": "python",
                        }
                    ]
                ],
            }

    class FakeStorage:
        def get_chroma_collection(self, embedding_function, create: bool = False):
            return FakeCollection()

    monkeypatch.setattr(semantic_module, "build_embedding_function", lambda config: object())
    repo_index = SimpleNamespace(storage=FakeStorage())
    config = SimpleNamespace(enable_vector_index=True, embedding=SimpleNamespace())

    snippets = semantic_module.semantic_search(repo_index, "query", 3, config)

    assert captured["include"] == ["documents", "metadatas"]
    assert snippets[0].path == "main.py"
