"""OpenAI LLM client wrapper.

Provides chat completion (with structured output) and embedding generation.
Model names come from config, not hardcoded.
"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings


class LLMClient:
    """Thin wrapper around OpenAI chat + embeddings APIs."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.openai_chat_model
        self.embedding_model = settings.openai_embedding_model
        self.embedding_dimensions = settings.embedding_dimensions

    # -- Chat --

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, int, int]:
        """Send a chat completion request.

        Returns (content, prompt_tokens, completion_tokens).
        """
        kwargs: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        return content, usage.prompt_tokens, usage.completion_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_structured(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], int, int, int]:
        """Send a chat completion with structured JSON output.

        Returns (parsed_json, prompt_tokens, completion_tokens, latency_ms).
        """
        resp = self._client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        import json

        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        usage = resp.usage
        return parsed, usage.prompt_tokens, usage.completion_tokens, 0

    def chat_structured_timed(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], int, int, int]:
        """Like chat_structured but measures wall-clock latency."""
        t0 = time.time()
        parsed, pt, ct, _ = self.chat_structured(
            messages, schema=schema, temperature=temperature
        )
        latency_ms = int((time.time() - t0) * 1000)
        return parsed, pt, ct, latency_ms

    # -- Embeddings --

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        resp = self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            dimensions=self.embedding_dimensions,
        )
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


# Singleton
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
