from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from codexpert.core.citations import format_citations
from codexpert.core.types import (
    Citation,
    CodeSnippet,
    ExecutionTraceStep,
    QueryRequest,
    QueryResponse,
    ToolObservation,
)
from codexpert.tools.registry import ToolRegistry


DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are Codexpert, a codebase question-answering agent. "
    "Use repository tools to gather evidence before answering implementation "
    "questions. Prefer grep_search for exact identifiers and errors, read_file "
    "for cited line ranges, symbol and reference tools for code navigation, "
    "semantic_search for conceptual discovery, and doc_search for repository docs. "
    "Stop when you have "
    "enough citation-ready evidence. Do not guess about implementation details "
    "when tools can inspect the repository."
)

DEFAULT_FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are Codexpert, generating the final answer for a repository question. "
    "Base the answer only on observed repository evidence. Cite file ranges when "
    "evidence is used, and do not invent paths, line numbers, APIs, or behavior."
)


class FinalAnswer(BaseModel):
    answer: str


def _merge_evidence(left: list[CodeSnippet], right: list[CodeSnippet]) -> list[CodeSnippet]:
    merged: list[CodeSnippet] = []
    seen: set[tuple[str, int, int, str | None]] = set()
    for snippet in [*left, *right]:
        key = (snippet.path, snippet.start_line, snippet.end_line, snippet.symbol)
        if key in seen:
            continue
        seen.add(key)
        merged.append(snippet)
    return merged


class ReActState(TypedDict):
    repo_id: str
    question: str
    top_k: int
    max_iterations: int
    messages: Annotated[list[AnyMessage], add_messages]
    observations: Annotated[list[ToolObservation], operator.add]
    evidence_set: Annotated[list[CodeSnippet], _merge_evidence]
    trace: Annotated[list[ExecutionTraceStep], operator.add]
    iterations: Annotated[int, operator.add]
    answer: str | None


class ReActOrchestrator:
    def __init__(self, llm: BaseChatModel, repo_index: Any, config: Any) -> None:
        self.llm = llm
        self.repo_index = repo_index
        self.config = config
        self.tools = ToolRegistry(repo_index, config)
        self.langchain_tools = self.tools.as_langchain_tools()
        self._tools_node = ToolNode(self.langchain_tools)
        self.graph = self._build_graph()

    async def run(self, request: QueryRequest) -> QueryResponse:
        content = request.question
        if request.history:
            rendered_history = "\n".join(request.history)
            content = f"{request.question}\n\nHistory:\n{rendered_history}"

        state: ReActState = {
            "repo_id": request.repo_id,
            "question": request.question,
            "top_k": request.top_k or self.config.top_k,
            "iterations": 0,
            "max_iterations": self.config.max_iterations,
            "messages": [HumanMessage(content=content)],
            "observations": [],
            "evidence_set": [],
            "answer": None,
            "trace": [],
        }
        final_state = await self.graph.ainvoke(state, self._trace_config(request, state))
        evidence = self._limit_evidence(final_state.get("evidence_set", []))
        citations = _collect_citations(evidence)
        return QueryResponse(
            answer=final_state.get("answer") or "",
            citations=citations,
            trace=final_state.get("trace", []),
        )

    def _build_graph(self):
        graph = StateGraph(ReActState)
        graph.add_node("guard", self._guard)
        graph.add_node("model", self._model)
        graph.add_node("tools", self._tools_node)
        graph.add_node("evidence_check", self._evidence_check)
        graph.add_node("final", self._final)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", self._route_from_guard)
        graph.add_conditional_edges("model", self._route_from_model)
        graph.add_edge("tools", "evidence_check")
        graph.add_edge("evidence_check", "guard")
        graph.add_edge("final", END)
        return graph.compile()

    async def _guard(self, state: ReActState) -> dict[str, Any]:
        return {}

    async def _evidence_check(self, state: ReActState) -> dict[str, Any]:
        max_snippets = int(getattr(self.config, "max_evidence_snippets", 20))
        snippets = [
            snippet
            for observation in state["observations"]
            for snippet in observation.snippets
        ][:max_snippets]
        return {"evidence_set": snippets}

    def _route_from_guard(self, state: ReActState) -> str:
        if state["iterations"] >= state["max_iterations"]:
            return "final"
        return "model"

    async def _model(self, state: ReActState) -> dict[str, Any]:
        bound = self.llm.bind_tools(self.langchain_tools)
        response = await bound.ainvoke(
            [
                SystemMessage(content=self._agent_system_prompt()),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    def _route_from_model(self, state: ReActState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "final"

    async def _final(self, state: ReActState) -> dict[str, Any]:
        structured = self.llm.with_structured_output(FinalAnswer)
        evidence = self._limit_evidence(state.get("evidence_set", []))
        citations = _collect_citations(evidence)
        rendered_citations = "\n".join(f"- {value}" for value in format_citations(citations))
        if not rendered_citations:
            rendered_citations = "- (none)"

        rendered_evidence = _render_evidence(evidence)
        rendered_observations = _render_observations(state["observations"])
        final_prompt = (
            f"Question:\n{state['question']}\n\n"
            "Citation-ready evidence:\n"
            f"{rendered_evidence}\n\n"
            "Tool trace and notes:\n"
            f"{rendered_observations}\n\n"
            "Available citations (use only these when citing):\n"
            f"{rendered_citations}\n\n"
            "Citation instructions:\n"
            "- If evidence is used, cite inline using format [path:start-end].\n"
            "- Use only citations from the available list above.\n"
            "- Do not invent file paths or line ranges.\n"
            "- If no citation is available, say repository evidence was not found.\n"
        )

        answer = await structured.ainvoke(
            [
                SystemMessage(content=self._final_answer_system_prompt()),
                HumanMessage(content=final_prompt),
            ]
        )
        return {"answer": answer.answer}

    def _agent_system_prompt(self) -> str:
        return getattr(self.config, "agent_system_prompt", DEFAULT_AGENT_SYSTEM_PROMPT)

    def _final_answer_system_prompt(self) -> str:
        return getattr(
            self.config,
            "final_answer_system_prompt",
            DEFAULT_FINAL_ANSWER_SYSTEM_PROMPT,
        )

    def _trace_config(self, request: QueryRequest, state: ReActState) -> dict[str, Any]:
        metadata = {
            "repo_id": request.repo_id,
            "top_k": state["top_k"],
            "max_iterations": state["max_iterations"],
            **request.metadata,
        }
        return {
            "run_name": "codexpert.query",
            "tags": list(getattr(self.config, "trace_tags", ["codexpert"])),
            "metadata": metadata,
        }

    def _limit_evidence(self, evidence: list[CodeSnippet]) -> list[CodeSnippet]:
        max_snippets = int(getattr(self.config, "max_evidence_snippets", 20))
        return evidence[:max_snippets]


def _collect_citations(snippets: list[CodeSnippet]) -> list[Citation]:
    citations: list[Citation] = []
    seen = set()
    for snippet in snippets:
        key = (snippet.path, snippet.start_line, snippet.end_line)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                path=snippet.path,
                start_line=snippet.start_line,
                end_line=snippet.end_line,
            )
        )
    return citations


def _render_evidence(snippets: list[CodeSnippet]) -> str:
    if not snippets:
        return "- (none)"

    lines: list[str] = []
    for index, snippet in enumerate(snippets, start=1):
        label = f"{index}. {snippet.path}:{snippet.start_line}-{snippet.end_line}"
        metadata = []
        if snippet.symbol:
            metadata.append(f"symbol={snippet.symbol}")
        if snippet.language:
            metadata.append(f"language={snippet.language}")
        if metadata:
            label = f"{label} ({', '.join(metadata)})"
        lines.append(label)
        if snippet.text:
            lines.append(f"   text: {snippet.text}")
    return "\n".join(lines)


def _render_observations(observations: list[ToolObservation]) -> str:
    if not observations:
        return "- (none)"

    lines: list[str] = []
    for index, obs in enumerate(observations, start=1):
        lines.append(f"{index}. tool={obs.tool} query={obs.query}")
        if obs.notes:
            lines.append(f"   notes: {obs.notes}")
        if not obs.snippets:
            lines.append("   snippets: (none)")
            continue
        for snippet in obs.snippets:
            lines.append(
                "   - "
                f"{snippet.path}:{snippet.start_line}-{snippet.end_line}"
            )
            if snippet.text:
                lines.append(f"     text: {snippet.text}")
    return "\n".join(lines)
