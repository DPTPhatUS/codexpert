from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codexpert.core.config import AgentConfig, RepoSpec
from codexpert.memory.chunking import chunk_repository
from codexpert.memory.indexing import build_indexes
from codexpert.memory.repo_resolver import RepoResolver
from codexpert.memory.storage import RepoStorage


@dataclass(frozen=True)
class RepoIndex:
    repo_id: str
    repo_root: Path
    storage: RepoStorage


def ensure_repo_index(repo: RepoSpec, config: AgentConfig) -> RepoIndex:
    resolver = RepoResolver(config.repo_cache_dir)
    resolved = resolver.resolve(repo)
    storage = RepoStorage(config.index_dir, repo.repo_id)
    if storage.has_index():
        return RepoIndex(repo.repo_id, resolved.local_path, storage)

    chunks = chunk_repository(resolved.local_path, repo.repo_id, config.chunking)
    build_indexes(chunks, resolved.local_path, storage, config)
    storage.save_meta({"repo_id": repo.repo_id, "repo_root": str(resolved.local_path)})
    return RepoIndex(repo.repo_id, resolved.local_path, storage)


def load_existing_index(repo_id: str, config: AgentConfig) -> RepoIndex:
    storage = RepoStorage(config.index_dir, repo_id)
    if not storage.has_index():
        raise FileNotFoundError(
            f"Index not found for repo_id={repo_id}. Run async_ingest first."
        )
    meta = storage.load_meta()
    repo_root = Path(meta.get("repo_root", config.repo_cache_dir / repo_id))
    return RepoIndex(repo_id, repo_root, storage)
