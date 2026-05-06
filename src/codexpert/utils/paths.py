from __future__ import annotations

from pathlib import Path


def repo_relative_path(repo_root: Path, path: Path) -> str:
    return str(path.relative_to(repo_root))
