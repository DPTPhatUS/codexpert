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
    ExecutionTraceStep,
    QueryRequest,
    QueryResponse,
    ToolObservation,
)
from codexpert.tools.registry import ToolRegistry


DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are Codexpert, a codebase question-answering agent. "
    "Use the available repository search tools to gather evidence before answering "
    "questions about code. Prefer precise file, symbol, and error-message searches. "
    "Do not guess about implementation details when tools can inspect the repository."
)

DEFAULT_FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are Codexpert, generating the final answer for a repository question. "
    "Base the answer only on observed repository evidence. Cite file ranges when "
    "evidence is used, and do not invent paths, line numbers, APIs, or behavior."
)


class FinalAnswer(BaseModel):
    answer: str


class ReActState(TypedDict):
    repo_id: str
    question: str
    top_k: int
    max_iterations: int
    messages: Annotated[list[AnyMessage], add_messages]
    observations: Annotated[list[ToolObservation], operator.add]
    trace: Annotated[list[ExecutionTraceStep], operator.add]
    iterations: Annotated[int, operator.add]
    answer: str | None


class ReActOrchestrator:
    def __init__(self, llm: BaseChatModel, repo_index: Any, config: Any) -> None:
        self.llm = llm
        self.repo_index = repo_index
        self.config = config
        self.tools = ToolRegistry(repo_index, config)
        self._tools_node = ToolNode(self.tools.as_langchain_tools())
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
            "answer": None,
            "trace": [],
        }
        final_state = await self.graph.ainvoke(state)
        citations = _collect_citations(final_state["observations"])
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
        graph.add_node("final", self._final)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", self._route_from_guard)
        graph.add_conditional_edges("model", self._route_from_model)
        graph.add_edge("tools", "guard")
        graph.add_edge("final", END)
        return graph.compile()

    async def _guard(self, state: ReActState) -> dict[str, Any]:
        return {}

    def _route_from_guard(self, state: ReActState) -> str:
        if state["iterations"] >= state["max_iterations"]:
            return "final"
        return "model"

    async def _model(self, state: ReActState) -> dict[str, Any]:
        bound = self.llm.bind_tools(self.tools.as_langchain_tools())
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
        citations = _collect_citations(state["observations"])
        rendered_citations = "\n".join(f"- {value}" for value in format_citations(citations))
        if not rendered_citations:
            rendered_citations = "- (none)"

        rendered_observations = _render_observations(state["observations"])
        final_prompt = (
            f"Question:\n{state['question']}\n\n"
            "Observed evidence:\n"
            f"{rendered_observations}\n\n"
            "Available citations (use only these when citing):\n"
            f"{rendered_citations}\n\n"
            "Citation instructions:\n"
            "- If evidence is used, cite inline using format [path:start-end].\n"
            "- Use only citations from the available list above.\n"
            "- Do not invent file paths or line ranges.\n"
            "- If no citation is available, answer without citations.\n"
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


def _collect_citations(observations: list[ToolObservation]) -> list[Citation]:
    citations: list[Citation] = []
    seen = set()
    for obs in observations:
        for snippet in obs.snippets:
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
