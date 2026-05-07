from __future__ import annotations

from codexpert.core.config import LLMConfig, LLMProvider
from codexpert.llm.base import LLMClient
from codexpert.llm.gemini import GeminiClient
from codexpert.llm.mistral import MistralClient
from codexpert.llm.openai import OpenAIClient


_PROVIDER_DEFAULTS: dict[LLMProvider, tuple[str, str]] = {
    LLMProvider.GEMINI: ("gemini-1.5-pro", "GEMINI_API_KEY"),
    LLMProvider.OPENAI: ("gpt-4o-mini", "OPENAI_API_KEY"),
    LLMProvider.MISTRAL: ("mistral-large-latest", "MISTRAL_API_KEY"),
}


def resolve_llm_config(config: LLMConfig) -> LLMConfig:
    default_model, default_api_key_env = _PROVIDER_DEFAULTS[config.provider]
    resolved_model = config.model
    resolved_api_key_env = config.api_key_env

    if resolved_model == LLMConfig.model_fields["model"].default:
        resolved_model = default_model

    if resolved_api_key_env == LLMConfig.model_fields["api_key_env"].default:
        resolved_api_key_env = default_api_key_env

    return config.model_copy(
        update={"model": resolved_model, "api_key_env": resolved_api_key_env}
    )


def build_llm_client(config: LLMConfig) -> LLMClient:
    resolved = resolve_llm_config(config)
    if resolved.provider == LLMProvider.OPENAI:
        return OpenAIClient(resolved)
    if resolved.provider == LLMProvider.MISTRAL:
        return MistralClient(resolved)
    return GeminiClient(resolved)
