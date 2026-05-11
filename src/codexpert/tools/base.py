from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from codexpert.core.types import ExecutionTraceStep
from codexpert.core.types import ToolObservation


@dataclass(frozen=True)
class ToolContext:
    repo_index: Any
    config: Any


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolObservation]]


def derive_query(kwargs: dict[str, Any]) -> str:
    return str(
        kwargs.get("query")
        or kwargs.get("pattern")
        or kwargs.get("symbol")
        or kwargs.get("path")
        or ""
    )


async def run_action(
    context: ToolContext,
    handlers: dict[str, ToolHandler],
    action: str,
    **kwargs: Any,
) -> ToolObservation:
    handler = handlers.get(action)
    if handler is None:
        return ToolObservation(tool=action, query=derive_query(kwargs), notes="unknown tool")
    return await handler(context, kwargs)


async def run_tool_with_runtime(
    action: str,
    runtime: ToolRuntime | None,
    handler: ToolHandler,
    context: ToolContext,
    **kwargs: Any,
) -> Command:
    state = runtime.state if runtime else {}
    top_k = state.get("top_k", context.config.top_k)
    iteration = state.get("iterations", 0)
    query = derive_query(kwargs)
    try:
        observation = await handler(context, {**kwargs, "top_k": top_k})
    except Exception as exc:
        observation = ToolObservation(
            tool=action,
            query=query,
            notes=f"{type(exc).__name__}: {exc}",
        )
    trace_step = ExecutionTraceStep(
        iteration=iteration,
        action=action,
        query=observation.query,
        result_count=len(observation.snippets),
        notes=observation.notes,
    )
    tool_message = ToolMessage(
        tool_call_id=runtime.tool_call_id if runtime else "",
        content=render_tool_message(
            observation,
            max_chars=int(getattr(context.config, "max_tool_message_chars", 4000)),
        ),
    )
    return Command(
        update={
            "observations": [observation],
            "evidence_set": observation.snippets,
            "trace": [trace_step],
            "iterations": 1,
            "messages": [tool_message],
        }
    )


def render_tool_message(observation: ToolObservation, *, max_chars: int) -> str:
    if observation.snippets:
        lines = [
            f"{observation.tool} "
            f"hits={len(observation.snippets)} "
            f"query={observation.query}"
        ]
        for index, snippet in enumerate(observation.snippets[:5], start=1):
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
                lines.append(_indent(_truncate(snippet.text, 900)))
        return _truncate("\n".join(lines), max_chars)
    if observation.notes:
        return _truncate(
            f"{observation.tool} query={observation.query} notes={observation.notes}",
            max_chars,
        )
    return f"{observation.tool} no_results"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...<truncated>"


def _indent(text: str) -> str:
    return "\n".join(f"   {line}" for line in text.splitlines())
