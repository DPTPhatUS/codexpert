from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    MISTRAL = "mistral"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.GEMINI
    model: str = "gemini-1.5-pro"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    max_output_tokens: int = 2048
    timeout_s: int = 60


class EmbeddingConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


class ChromaConfig(BaseModel):
    persist_path: Path | None = None


class ChunkingConfig(BaseModel):
    max_chunk_chars: int = 4000
    min_chunk_chars: int = 200
    max_file_chars: int = 200000


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace_dir: Path = Field(default_factory=lambda: Path.home() / ".codexpert")
    ripgrep_enabled: bool = True
    max_iterations: int = 6
    top_k: int = 5
    trace_tags: list[str] = Field(default_factory=lambda: ["codexpert"])
    agent_system_prompt: str = (
        "You are Codexpert, a codebase question-answering agent. "
        "Use the available repository search tools to gather evidence before answering "
        "questions about code. Prefer precise file, symbol, and error-message searches. "
        "Do not guess about implementation details when tools can inspect the repository."
    )
    final_answer_system_prompt: str = (
        "You are Codexpert, generating the final answer for a repository question. "
        "Base the answer only on observed repository evidence. Cite file ranges when "
        "evidence is used, and do not invent paths, line numbers, APIs, or behavior."
    )
    enable_vector_index: bool = True
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def _coerce_workspace_dir(cls, value: Any) -> Path:
        return Path(value).expanduser()

    @property
    def repo_cache_dir(self) -> Path:
        return self.workspace_dir / "repos"

    @property
    def index_dir(self) -> Path:
        return self.workspace_dir / "indexes"

    @property
    def chroma_dir(self) -> Path:
        if self.chroma.persist_path:
            return self.chroma.persist_path
        return self.index_dir / "chroma"


class RepoSpec(BaseModel):
    repo_id: str
    source: str
    ref: str | None = None

    @classmethod
    def from_source(cls, source: str, repo_id: str | None = None) -> "RepoSpec":
        resolved_id = repo_id or _slugify_repo_id(source)
        return cls(repo_id=resolved_id, source=source)


def _slugify_repo_id(source: str) -> str:
    cleaned = source.strip().rstrip("/")
    cleaned = cleaned.replace("https://", "").replace("http://", "")
    cleaned = cleaned.replace("git@", "").replace(":", "/")
    cleaned = cleaned.replace("github.com/", "")
    cleaned = cleaned.replace(".", "-").replace("/", "-")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "-")
    return cleaned or "repo"
