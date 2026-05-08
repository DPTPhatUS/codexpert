from __future__ import annotations

import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
from langchain_core.messages import ToolMessage

# Avoid unrelated package init import error during test collection.
if "codexpert.llm.gemini" not in sys.modules:
    gemini_stub = ModuleType("codexpert.llm.gemini")
    gemini_stub.GeminiClient = object
    sys.modules["codexpert.llm.gemini"] = gemini_stub

from codexpert.core.types import CodeSnippet
from codexpert.tools.registry import ToolRegistry


class _FakeRuntime:
    def __init__(self, top_k: int = 3, iterations: int = 2, tool_call_id: str = "tc-1") -> None:
        self.state = {"top_k": top_k, "iterations": iterations}
        self.tool_call_id = tool_call_id


@pytest.mark.asyncio
async def test_as_langchain_tools_exposes_expected_tool_names() -> None:
    registry = ToolRegistry(
        repo_index=object(),
        config=SimpleNamespace(top_k=5, ripgrep_enabled=True, enable_vector_index=False),
    )

    tools = registry.as_langchain_tools()

    assert [tool.name for tool in tools] == [
        "semantic_search",
        "lexical_search",
        "ripgrep_search",
        "graph_query",
    ]


@pytest.mark.asyncio
async def test_runtime_tool_returns_command_update_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(top_k=7, ripgrep_enabled=True, enable_vector_index=True)
    registry = ToolRegistry(repo_index=object(), config=config)

    def fake_lexical_search(_repo_index, _query: str, _top_k: int) -> list[CodeSnippet]:
        return [
            CodeSnippet(path="a.py", start_line=1, end_line=2, text="x"),
            CodeSnippet(path="b.py", start_line=3, end_line=4, text="y"),
        ]

    monkeypatch.setattr("codexpert.tools.registry.lexical_search", fake_lexical_search)

    lexical_tool = next(tool for tool in registry.as_langchain_tools() if tool.name == "lexical_search")
    runtime = _FakeRuntime(top_k=4, iterations=6, tool_call_id="call-42")

    command = await lexical_tool.coroutine(query="needle", runtime=runtime)
    update = command.update

    assert set(update.keys()) == {"observations", "trace", "iterations", "messages"}
    assert update["iterations"] == 1
    assert update["observations"][0].tool == "lexical_search"
    assert update["observations"][0].query == "needle"
    assert update["trace"][0].iteration == 6
    assert update["trace"][0].action == "lexical_search"
    assert update["trace"][0].result_count == 2
    assert isinstance(update["messages"][0], ToolMessage)
    assert update["messages"][0].tool_call_id == "call-42"
    assert "lexical_search hits=2" in str(update["messages"][0].content)
