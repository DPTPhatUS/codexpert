from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from codexpert.core.citations import format_citations
from codexpert.core.types import (
    Citation,
    ExecutionTraceStep,
    QueryRequest,
    QueryResponse,
    ToolObservation,
)
from codexpert.llm.base import LLMClient
from codexpert.tools.registry import ToolRegistry


class ReActDecision(BaseModel):
    action: str
    query: str | None = None
    notes: str | None = None
    final: str | None = None


class ReActState(TypedDict):
    repo_id: str
    question: str
    history: list[str]
    top_k: int
    iterations: int
    max_iterations: int
    observations: list[ToolObservation]
    decision: ReActDecision | None
    answer: str | None
    trace: list[ExecutionTraceStep]


class ReActOrchestrator:
    def __init__(self, llm: LLMClient, repo_index: Any, config: Any) -> None:
        self.llm = llm
        self.repo_index = repo_index
        self.config = config
        self.tools = ToolRegistry(repo_index, config)
        self.graph = self._build_graph()

    async def run(self, request: QueryRequest) -> QueryResponse:
        state: ReActState = {
            "repo_id": request.repo_id,
            "question": request.question,
            "history": request.history,
            "top_k": request.top_k or self.config.top_k,
            "iterations": 0,
            "max_iterations": self.config.max_iterations,
            "observations": [],
            "decision": None,
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
        graph.add_node("reason", self._reason)
        graph.add_node("act", self._act)
        graph.add_node("final", self._final)
        graph.set_entry_point("reason")
        graph.add_conditional_edges("reason", self._route_from_reason)
        graph.add_edge("act", "reason")
        graph.add_edge("final", END)
        return graph.compile()

    async def _reason(self, state: ReActState) -> ReActState:
        if state["iterations"] >= state["max_iterations"]:
            decision = ReActDecision(action="final", notes="max_iterations_reached")
            state["decision"] = decision
            return state

        prompt = _reason_prompt(state["question"], state["history"], state["observations"])
        raw = await self.llm.complete(prompt)
        decision = _parse_decision(raw)
        state["decision"] = decision
        return state

    def _route_from_reason(self, state: ReActState) -> str:
        decision = state.get("decision")
        if not decision or decision.action == "final":
            return "final"
        return "act"

    async def _act(self, state: ReActState) -> ReActState:
        decision = state.get("decision")
        if not decision or not decision.query:
            state["decision"] = ReActDecision(action="final", notes="no_action")
            return state

        observation = await self.tools.run(decision.action, decision.query, state["top_k"])
        state["observations"].append(observation)
        state["trace"].append(
            ExecutionTraceStep(
                iteration=state["iterations"],
                action=decision.action,
                query=decision.query,
                result_count=len(observation.snippets),
                notes=observation.notes,
            )
        )
        state["iterations"] += 1
        return state

    async def _final(self, state: ReActState) -> ReActState:
        prompt = _final_prompt(state["question"], state["observations"])
        answer = await self.llm.complete(prompt)
        state["answer"] = answer
        return state


def _parse_decision(raw: str) -> ReActDecision:
    try:
        payload = json.loads(_extract_json(raw))
        return ReActDecision.model_validate(payload)
    except Exception:
        return ReActDecision(action="final", final=raw)


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON payload found")
    return text[start : end + 1]


def _reason_prompt(question: str, history: list[str], observations: list[ToolObservation]) -> str:
    rendered_obs = _render_observations(observations)
    rendered_history = "\n".join(history)
    return (
        "You are codexpert, a codebase QA agent. Decide the next action to answer the question.\n"
        "Available tools: semantic_search, lexical_search, ripgrep_search, graph_query, final.\n"
        "Respond with a JSON object {\"action\": ..., \"query\": ..., \"notes\": ...}.\n"
        "Use action=final when you can answer.\n\n"
        f"Question: {question}\n\n"
        f"History:\n{rendered_history}\n\n"
        f"Observations:\n{rendered_obs}\n"
    )


def _final_prompt(question: str, observations: list[ToolObservation]) -> str:
    rendered_obs = _render_observations(observations, include_text=True)
    citations = format_citations(_collect_citations(observations))
    citation_block = "\n".join(citations)
    return (
        "You are codexpert. Answer the question using the observations.\n"
        "Include citations in the form [path:start-end] when referencing code.\n"
        f"Question: {question}\n\n"
        f"Observations:\n{rendered_obs}\n\n"
        f"Available citations:\n{citation_block}\n"
    )


def _render_observations(
    observations: list[ToolObservation],
    include_text: bool = False,
) -> str:
    if not observations:
        return "(none)"
    rendered = []
    for obs in observations:
        header = f"- Tool: {obs.tool} | Query: {obs.query}"
        snippets = []
        for snippet in obs.snippets:
            if include_text:
                snippets.append(
                    f"  - {snippet.path}:{snippet.start_line}-{snippet.end_line}\n"
                    f"    {snippet.text.strip()}"
                )
            else:
                snippets.append(
                    f"  - {snippet.path}:{snippet.start_line}-{snippet.end_line}"
                )
        note = f"  Notes: {obs.notes}" if obs.notes else ""
        rendered.append("\n".join([header] + snippets + ([note] if note else [])))
    return "\n".join(rendered)


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
