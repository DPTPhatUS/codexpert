from __future__ import annotations

import os

from codexpert.core.config import LLMConfig
from codexpert.llm.base import LLMClient


class MistralClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {config.api_key_env}")
        from mistralai.client import Mistral

        self.client = Mistral(api_key=api_key)
        self.model = config.model
        self.temperature = config.temperature
        self.max_output_tokens = config.max_output_tokens
        self.timeout_ms = config.timeout_s * 1000

    async def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.complete_async(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stream=False,
            response_format={"type": "text"},
            timeout_ms=self.timeout_ms,
        )
        return response.choices[0].message.content or ""
