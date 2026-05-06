from pathlib import Path
from tempfile import TemporaryDirectory

from codexpert.core.config import ChunkingConfig
from codexpert.memory.chunking import chunk_file


def test_chunk_file_python():
    with TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        path = repo_root / "sample.py"
        path.write_text(
            """
def add(a, b):
    return a + b

class Thing:
    def run(self):
        return add(1, 2)
""".strip(),
            encoding="utf-8",
        )
        chunks = chunk_file(path, repo_root, "repo", ChunkingConfig())
        assert chunks
        assert chunks[0].path == "sample.py"
