from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from codexpert import AgentConfig, CodexpertAgent, QueryRequest, RepoSpec
from codexpert.core.config import LLMConfig, LLMProvider


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_provider(value: str) -> LLMProvider:
    normalized = value.strip().lower()
    allowed = ", ".join(member.value for member in LLMProvider)
    if normalized == "mistral" and "mistral" not in {member.value for member in LLMProvider}:
        raise ValueError(
            "Invalid CODEXPERT_LLM_PROVIDER='mistral': provider not available in this build. "
            f"Supported providers: {allowed}"
        )
    try:
        return LLMProvider(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid CODEXPERT_LLM_PROVIDER={value!r}. Expected one of: gemini, openai, mistral"
        ) from exc


def _default_model(provider: LLMProvider) -> str:
    if provider == LLMProvider.OPENAI:
        return "gpt-4o-mini"
    if provider == LLMProvider.GEMINI:
        return "gemini-1.5-pro"
    if provider.value == "mistral":
        return "mistral-large-latest"
    raise ValueError(f"Unsupported provider: {provider.value}")


def _build_config() -> AgentConfig:
    defaults = AgentConfig()
    provider_value = os.getenv("CODEXPERT_LLM_PROVIDER", "gemini")
    provider = _parse_provider(provider_value)
    model = os.getenv("CODEXPERT_LLM_MODEL")
    if not model:
        model = _default_model(provider)
    llm = LLMConfig(provider=provider, model=model)
    workspace = os.getenv("CODEXPERT_WORKSPACE_DIR")
    workspace_dir = Path(workspace).expanduser() if workspace else defaults.workspace_dir
    return AgentConfig(
        workspace_dir=workspace_dir,
        llm=llm,
        top_k=_int_env("CODEXPERT_TOP_K", defaults.top_k),
        max_iterations=_int_env("CODEXPERT_MAX_ITERATIONS", defaults.max_iterations),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codexpert against a GitHub repo")
    parser.add_argument("url", help="GitHub repo URL")
    parser.add_argument("--repo-id", default=None, help="Repo ID namespace")
    parser.add_argument("--ref", default=None, help="Git ref (branch or tag)")
    parser.add_argument(
        "--question",
        default="How is authentication handled?",
        help="Question to ask",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = _parse_args()
    config = _build_config()
    agent = CodexpertAgent(config)

    repo = RepoSpec.from_source(args.url, repo_id=args.repo_id)
    if args.ref:
        repo = repo.model_copy(update={"ref": args.ref})
    await agent.async_ingest(repo)

    request = QueryRequest(repo_id=repo.repo_id, question=args.question, top_k=config.top_k)
    response = await agent.async_query(request)

    print("Answer:")
    print(response.answer)
    if response.citations:
        print("\nCitations:")
        for citation in response.citations:
            print(f"- {citation.path}:{citation.start_line}-{citation.end_line}")


if __name__ == "__main__":
    asyncio.run(main())
