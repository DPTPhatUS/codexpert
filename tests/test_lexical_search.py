from pathlib import Path
from tempfile import TemporaryDirectory

from codexpert.core.config import AgentConfig, RepoSpec
from codexpert.memory.ingest import ensure_repo_index
from codexpert.tools.search import lexical_search


def test_lexical_search_finds_symbol():
    with TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        repo_root = Path(tmp) / "repo"
        repo_root.mkdir()
        (repo_root / "utils.py").write_text(
            """
def make_client():
    return "client"
""".strip(),
            encoding="utf-8",
        )

        config = AgentConfig(
            workspace_dir=workspace,
            enable_vector_index=False,
            ripgrep_enabled=False,
        )
        repo = RepoSpec.from_source(str(repo_root), repo_id="local-utils")
        index = ensure_repo_index(repo, config)
        results = lexical_search(index, "make_client", 5)
        assert any("make_client" in snippet.text for snippet in results)
