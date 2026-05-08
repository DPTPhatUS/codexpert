from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from codexpert.core.config import LLMConfig


def build_openai_chat_model(config: LLMConfig) -> BaseChatModel:
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {config.api_key_env}")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        timeout=config.timeout_s,
    )
