from pathlib import Path
from tempfile import TemporaryDirectory

from codexpert.core.config import AgentConfig, RepoSpec
from codexpert.memory.ingest import ensure_repo_index
from codexpert.tools.lexical_search import lexical_search


def test_ingest_smoke():
    with TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        repo_root = Path(tmp) / "repo"
        repo_root.mkdir()
        (repo_root / "main.py").write_text(
            """
def connect_db():
    return "sqlite://"

def handler():
    return connect_db()
""".strip(),
            encoding="utf-8",
        )

        config = AgentConfig(
            workspace_dir=workspace,
            enable_vector_index=False,
            ripgrep_enabled=False,
        )
        repo = RepoSpec.from_source(str(repo_root), repo_id="local-repo")
        index = ensure_repo_index(repo, config)
        snippets = lexical_search(index, "connect_db", 3)
        assert snippets
