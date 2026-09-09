"""Kaplan hybrid retrieval — vector + full-text + metadata filter → RRF fusion."""

from __future__ import annotations

import logging
from typing import Any

from ..llm.client import get_llm_client
from ..repositories import chunks as chunk_repo

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    fts_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked lists using Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    items: dict[int, dict[str, Any]] = {}

    for rank, r in enumerate(vector_results):
        cid = r["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in items:
            items[cid] = r

    for rank, r in enumerate(fts_results):
        cid = r["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in items:
            items[cid] = r

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    result = []
    for cid in sorted_ids:
        item = items[cid].copy()
        item["fused_score"] = scores[cid]
        result.append(item)
    return result


class KaplanRetriever:
    """Hybrid retrieval interface for Kaplan Chapter 33."""

    def __init__(self, top_k: int = 10, source_document_id: int | None = None):
        self.top_k = top_k
        self.source_document_id = source_document_id

    def retrieve(
        self,
        query: str,
        drug_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant Kaplan passages using hybrid search.

        Args:
            query: The question or search query.
            drug_ids: Optional drug IDs for metadata filtering.

        Returns:
            Fused and ranked list of passage dicts.
        """
        client = get_llm_client()

        # Vector search
        query_embedding = client.embed_one(query)
        drug_id = drug_ids[0] if drug_ids else None
        vector_results = chunk_repo.vector_search(
            query_embedding, limit=self.top_k, drug_id=drug_id,
            source_document_id=self.source_document_id,
        )

        # Full-text search
        fts_results = chunk_repo.fulltext_search(
            query, limit=self.top_k, drug_id=drug_id,
            source_document_id=self.source_document_id,
        )

        # Fuse with RRF
        fused = reciprocal_rank_fusion(vector_results, fts_results)

        return fused[: self.top_k]
