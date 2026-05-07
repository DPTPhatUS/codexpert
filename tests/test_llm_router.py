from __future__ import annotations

import importlib.util
from pathlib import Path

from codexpert.core.config import LLMConfig, LLMProvider
from codexpert.llm.router import resolve_llm_config


def _load_demo_local():
    path = Path(__file__).resolve().parents[1] / "scripts" / "demo_local.py"
    spec = importlib.util.spec_from_file_location("demo_local", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_openai_defaults() -> None:
    config = LLMConfig(provider=LLMProvider.OPENAI)

    resolved = resolve_llm_config(config)

    assert resolved.model == "gpt-4o-mini"
    assert resolved.api_key_env == "OPENAI_API_KEY"


def test_resolve_mistral_defaults() -> None:
    config = LLMConfig(provider=LLMProvider.MISTRAL)

    resolved = resolve_llm_config(config)

    assert resolved.model == "mistral-large-latest"
    assert resolved.api_key_env == "MISTRAL_API_KEY"


def test_resolve_mistral_preserves_explicit_model_and_env() -> None:
    config = LLMConfig(
        provider=LLMProvider.MISTRAL,
        model="codestral-latest",
        api_key_env="CUSTOM_MISTRAL_KEY",
    )

    resolved = resolve_llm_config(config)

    assert resolved.model == "codestral-latest"
    assert resolved.api_key_env == "CUSTOM_MISTRAL_KEY"


def test_parse_provider_accepts_mistral() -> None:
    assert _load_demo_local()._parse_provider("mistral") == LLMProvider.MISTRAL
