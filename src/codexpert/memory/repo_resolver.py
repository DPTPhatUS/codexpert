from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from codexpert.core.config import RepoSpec


@dataclass(frozen=True)
class ResolvedRepo:
    repo_id: str
    local_path: Path
    source: str
    is_github: bool


class RepoResolver:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def resolve(self, spec: RepoSpec) -> ResolvedRepo:
        if _is_github_url(spec.source):
            local_path = self.cache_dir / spec.repo_id
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                _clone_repo(spec.source, local_path)
            if spec.ref:
                _checkout_ref(local_path, spec.ref)
            return ResolvedRepo(spec.repo_id, local_path, spec.source, True)

        local_path = Path(spec.source).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Repository path not found: {local_path}")
        return ResolvedRepo(spec.repo_id, local_path, spec.source, False)


def _is_github_url(source: str) -> bool:
    return source.startswith("https://github.com/") or source.startswith("git@github.com:")


def _clone_repo(source: str, destination: Path) -> None:
    result = subprocess.run(
        ["git", "clone", "--depth", "1", source, str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")


def _checkout_ref(repo_path: Path, ref: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "checkout", ref],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout failed: {result.stderr}")
