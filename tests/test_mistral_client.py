from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from codexpert.core.config import LLMConfig, LLMProvider
from codexpert.llm.mistral import MistralClient


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
        MistralClient(_config())


@pytest.mark.asyncio
async def test_mistral_client_sends_system_and_user_messages(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeChat:
        async def complete_async(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
            )

    class FakeMistral:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.chat = FakeChat()

    client_module = ModuleType("mistralai.client")
    client_module.Mistral = FakeMistral
    package_module = ModuleType("mistralai")
    package_module.client = client_module
    monkeypatch.setitem(sys.modules, "mistralai", package_module)
    monkeypatch.setitem(sys.modules, "mistralai.client", client_module)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    client = MistralClient(_config())
    result = await client.complete("user prompt", system="system prompt")

    assert result == "done"
    assert calls == [
        {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            "temperature": 0.2,
            "max_tokens": 256,
            "stream": False,
            "response_format": {"type": "text"},
            "timeout_ms": 60000,
        }
    ]
