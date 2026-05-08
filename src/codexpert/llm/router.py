from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from codexpert.core.config import LLMConfig, LLMProvider


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


def build_llm_client(config: LLMConfig) -> BaseChatModel:
    resolved = resolve_llm_config(config)
    if resolved.provider == LLMProvider.OPENAI:
        from codexpert.llm.openai import build_openai_chat_model

        return build_openai_chat_model(resolved)
    if resolved.provider == LLMProvider.MISTRAL:
        from codexpert.llm.mistral import build_mistral_chat_model

        return build_mistral_chat_model(resolved)
    from codexpert.llm.gemini import build_gemini_chat_model

    return build_gemini_chat_model(resolved)
