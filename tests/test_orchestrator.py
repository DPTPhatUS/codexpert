from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

import codexpert.core.orchestrator as orchestrator_module
from codexpert.core.orchestrator import FinalAnswer, ReActOrchestrator
from codexpert.core.types import CodeSnippet, QueryRequest, QueryResponse, ToolObservation


class FakeStructuredLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> FinalAnswer:
        self.calls.append(list(messages))
        return FinalAnswer(answer=self.answer)


class FakeChatModel:
    def __init__(self, final_answer: str = "done") -> None:
        self.final_answer = final_answer
        self.bound_tools: list[object] = []
        self.calls: list[list[object]] = []
        self.structured = FakeStructuredLLM(final_answer)

    def bind_tools(self, tools: list[object]) -> "FakeChatModel":
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages: list[object]) -> AIMessage:
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return AIMessage(
                content="need search",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "args": {"query": "alpha"},
                        "id": f"call-{uuid4()}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="enough evidence", tool_calls=[])

    def with_structured_output(self, _schema: type[FinalAnswer]) -> FakeStructuredLLM:
        return self.structured


class FakeTools:
    def __init__(self, _repo_index: object | None = None, _config: object | None = None) -> None:
        pass

    def as_langchain_tools(self) -> list[object]:
        @tool
        async def semantic_search(
            query: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """fake semantic search"""
            obs = ToolObservation(
                tool="semantic_search",
                query=query,
                snippets=[
                    CodeSnippet(
                        path="pkg/file_1.py",
                        start_line=10,
                        end_line=12,
                        text="x = 1",
                    )
                ],
                notes="hit",
            )
            trace = {
                "iteration": 0,
                "action": "semantic_search",
                "query": query,
                "result_count": 1,
                "notes": "hit",
            }
            return Command(
                update={
                    "messages": [ToolMessage(content="ok", tool_call_id=tool_call_id)],
                    "observations": [obs],
                    "trace": [trace],
                    "iterations": 1,
                }
            )

        return [semantic_search]


def _config(max_iterations: int = 4) -> SimpleNamespace:
    return SimpleNamespace(top_k=3, max_iterations=max_iterations)


@pytest.mark.asyncio
async def test_compiled_run_uses_toolnode_and_returns_query_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator_module, "ToolRegistry", FakeTools)
    llm = FakeChatModel(final_answer="final answer")
    orchestrator = ReActOrchestrator(llm, repo_index=object(), config=_config())

    response = await orchestrator.run(
        QueryRequest(repo_id="repo", question="what?", history=["old"], top_k=2)
    )

    assert isinstance(response, QueryResponse)
    assert response.answer == "final answer"
    assert [step.action for step in response.trace] == ["semantic_search"]
    assert [step.query for step in response.trace] == ["alpha"]
    assert [c.path for c in response.citations] == ["pkg/file_1.py"]
    assert llm.bound_tools
    first_messages = llm.calls[0]
    assert isinstance(first_messages[0], SystemMessage)
    assert "codebase question-answering agent" in first_messages[0].content
    assert isinstance(first_messages[1], HumanMessage)
    assert "History:\nold" in first_messages[1].content
    final_messages = llm.structured.calls[0]
    assert isinstance(final_messages[0], SystemMessage)
    assert "Base the answer only on observed repository evidence" in final_messages[0].content
    assert isinstance(final_messages[1], HumanMessage)
    final_prompt = final_messages[1].content
    assert "Question:\nwhat?" in final_prompt
    assert "pkg/file_1.py:10-12" in final_prompt
    assert "text: x = 1" in final_prompt
    assert "Available citations (use only these when citing):" in final_prompt
    assert "- [pkg/file_1.py:10-12]" in final_prompt
    assert "cite inline using format [path:start-end]" in final_prompt
    assert "Do not invent file paths or line ranges." in final_prompt


@pytest.mark.asyncio
async def test_guard_stops_after_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_module, "ToolRegistry", FakeTools)
    llm = FakeChatModel(final_answer="maxed out")
    orchestrator = ReActOrchestrator(llm, repo_index=object(), config=_config(max_iterations=1))

    response = await orchestrator.run(QueryRequest(repo_id="repo", question="what?", top_k=2))

    assert response.answer == "maxed out"
    assert len(llm.calls) == 1
    assert len(response.trace) == 1
