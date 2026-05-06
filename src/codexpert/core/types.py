from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from codexpert.core.config import RepoSpec


class CodeSnippet(BaseModel):
    path: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None
    language: str | None = None


class Citation(BaseModel):
    path: str
    start_line: int
    end_line: int


class ToolObservation(BaseModel):
    tool: str
    query: str
    snippets: list[CodeSnippet] = Field(default_factory=list)
    notes: str | None = None


class ExecutionTraceStep(BaseModel):
    iteration: int
    action: str
    query: str
    result_count: int
    notes: str | None = None


class QueryRequest(BaseModel):
    repo_id: str
    question: str
    top_k: int = 5
    history: list[str] = Field(default_factory=list)
    repo: RepoSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[ExecutionTraceStep] = Field(default_factory=list)
