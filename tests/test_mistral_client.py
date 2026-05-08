from __future__ import annotations

import sys
from types import ModuleType

import pytest

from codexpert.core.config import LLMConfig, LLMProvider
from codexpert.llm.router import build_llm_client


def _config() -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.MISTRAL,
        model="mistral-small-latest",
        api_key_env="MISTRAL_API_KEY",
        max_output_tokens=256,
    )


def test_mistral_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Missing API key in env var MISTRAL_API_KEY"):
        build_llm_client(_config())


def test_mistral_client_builds_langchain_chat_model(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeChatMistralAIAllKw:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    module = ModuleType("langchain_mistralai")
    module.ChatMistralAI = FakeChatMistralAIAllKw
    monkeypatch.setitem(sys.modules, "langchain_mistralai", module)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    build_llm_client(_config())

    assert calls == [
        {
            "model": "mistral-small-latest",
            "api_key": "test-key",
            "temperature": 0.2,
            "max_tokens": 256,
            "timeout": 60,
        }
    ]
