from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from codexpert.memory.chunking import CodeChunk
from codexpert.tools.navigation import (
    definition_lookup,
    doc_search,
    list_files,
    read_file,
    references_search,
    symbol_search,
)
from codexpert.tools.search import grep_search


class _Storage:
    def load_chunks(self) -> list[CodeChunk]:
        return [
            CodeChunk(
                chunk_id="repo:pkg/app.py:1-2",
                path="pkg/app.py",
                start_line=1,
                end_line=2,
                text="def connect_db():\n    return 'sqlite'",
                symbol="connect_db",
                language="python",
            )
        ]


def _repo_index(path: Path) -> SimpleNamespace:
    return SimpleNamespace(repo_root=path, storage=_Storage())


def test_file_listing_read_and_grep_return_citation_ready_snippets() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "pkg" / "app.py"
        source.parent.mkdir()
        source.write_text(
            "def connect_db():\n    return 'sqlite'\n\nvalue = connect_db()\n",
            encoding="utf-8",
        )
        index = _repo_index(root)

        listed = list_files(index, path="pkg", pattern="*.py")
        assert [snippet.path for snippet in listed] == ["pkg/app.py"]

        read = read_file(index, "pkg/app.py", 1, 2)
        assert read[0].path == "pkg/app.py"
        assert read[0].start_line == 1
        assert "connect_db" in read[0].text

        hits = grep_search(index, "connect_db", max_results=5)
        assert [hit.start_line for hit in hits] == [1, 4]


def test_symbol_definition_reference_and_doc_search() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "pkg" / "app.py"
        source.parent.mkdir()
        source.write_text("def connect_db():\n    return 'sqlite'\n", encoding="utf-8")
        (root / "README.md").write_text(
            "The database connection is created in pkg/app.py.\n",
            encoding="utf-8",
        )
        index = _repo_index(root)

        symbols = symbol_search(index, "connect")
        assert symbols[0].symbol == "connect_db"

        definitions = definition_lookup(index, "connect_db")
        assert definitions[0].path == "pkg/app.py"

        references = references_search(index, "connect_db")
        assert references[0].start_line == 1

        docs = doc_search(index, "database connection")
        assert docs[0].path == "README.md"
