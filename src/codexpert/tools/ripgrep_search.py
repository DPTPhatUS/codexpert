from __future__ import annotations

import shutil
import subprocess

from codexpert.core.types import CodeSnippet


def ripgrep_search(repo_index, query: str, top_k: int, config) -> list[CodeSnippet]:
    if not config.ripgrep_enabled:
        return []
    if not shutil.which("rg"):
        raise RuntimeError("ripgrep (rg) is required but not installed")

    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--max-count",
        str(top_k),
        query,
        ".",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_index.repo_root),
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"rg failed: {result.stderr}")

    snippets: list[CodeSnippet] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, line_number, text = parts
        if path.startswith("./"):
            path = path[2:]
        snippets.append(
            CodeSnippet(
                path=path,
                start_line=int(line_number),
                end_line=int(line_number),
                text=text,
            )
        )
    return snippets
