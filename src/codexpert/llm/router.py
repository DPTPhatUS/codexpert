from __future__ import annotations

from codexpert.core.config import LLMConfig, LLMProvider
from codexpert.llm.base import LLMClient
from codexpert.llm.gemini import GeminiClient
from codexpert.llm.openai import OpenAIClient


def build_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == LLMProvider.OPENAI:
        model = config.model
        api_key_env = config.api_key_env
        if api_key_env == "GEMINI_API_KEY":
            api_key_env = "OPENAI_API_KEY"
        if model == "gemini-1.5-pro":
            model = "gpt-4o-mini"
        resolved = config.model_copy(update={"model": model, "api_key_env": api_key_env})
        return OpenAIClient(resolved)
    return GeminiClient(config)
