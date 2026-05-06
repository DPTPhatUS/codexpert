from __future__ import annotations

import asyncio
import os

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from codexpert.core.config import LLMConfig
from codexpert.llm.base import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {config.api_key_env}")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.model)
        self.generation_config = GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

    async def complete(self, prompt: str, system: str | None = None) -> str:
        def _call() -> str:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            return response.text or ""

        return await asyncio.to_thread(_call)
