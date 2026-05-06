from __future__ import annotations

import asyncio
import os

from google import genai
from google.genai import types

from codexpert.core.config import LLMConfig
from codexpert.llm.base import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {config.api_key_env}")
        self.client = genai.Client(api_key=api_key)
        self.model = config.model
        self.generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

    async def complete(self, prompt: str, system: str | None = None) -> str:
        def _call() -> str:
            contents = prompt if not system else f"{system}\n\n{prompt}"
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self.generation_config,
            )
            return response.text or ""

        return await asyncio.to_thread(_call)
