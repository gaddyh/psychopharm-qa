"""Reranker — simple LLM-based reranking of retrieved passages."""

from __future__ import annotations

import logging
from typing import Any

from ..llm.client import get_llm_client

logger = logging.getLogger(__name__)


def rerank(query: str, passages: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Rerank passages using an LLM scoring prompt.

    V2: no arbitrary truncation of passage text for scoring. The LLM sees
    the full passage (chunks are already <=700 tokens from ingestion).
    """
    if len(passages) <= top_k:
        return passages

    client = get_llm_client()

    # Build a scoring prompt — full passage text, no truncation
    passages_text = ""
    for i, p in enumerate(passages):
        passages_text += f"\n[{i}] {p['text']}\n"

    prompt = f"""Rate the relevance of each passage to the question on a scale of 0-10.
Return JSON: {{"scores": [score0, score1, ...]}}

Question: {query}

Passages:{passages_text}

Rate each passage's relevance (0-10):"""

    try:
        messages = [{"role": "user", "content": prompt}]
        resp, _, _, _ = client.chat_structured(
            messages,
            schema={
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {"type": "number"},
                    }
                },
                "required": ["scores"],
                "additionalProperties": False,
            },
        )
        scores = resp.get("scores", [])
        # Pair and sort
        scored = list(zip(passages, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]
    except Exception as e:
        logger.warning(f"Reranking failed ({e}), returning original order")
        return passages[:top_k]
