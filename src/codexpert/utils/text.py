from __future__ import annotations

import re
from pathlib import Path


def read_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def simple_tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"\W+", text.lower()) if token]


def chunk_text_by_chars(text: str, max_chars: int, min_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if len(chunk) < min_chars and chunks:
            chunks[-1] += chunk
        else:
            chunks.append(chunk)
        start = end
    return chunks
