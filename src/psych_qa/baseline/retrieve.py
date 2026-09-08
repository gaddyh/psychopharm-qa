"""Vector-only retrieval for the naive baseline.

No FTS, no RRF, no rerank, no metadata filter.
Filters by scope_key + pipeline version in entity_metadata.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session
from ..llm.client import get_llm_client
from .ingest import PIPELINE_VERSION, SCOPE_KEY, SOURCE_TYPE


def vector_search(
    query_embedding: list[float],
    limit: int = 20,
    scope_key: str = SCOPE_KEY,
    pipeline: str = PIPELINE_VERSION,
) -> list[dict[str, Any]]:
    """Vector similarity search using pgvector.

    Filters by source_documents.scope_key and entity_metadata.pipeline
    to isolate the baseline pipeline from the amisulpride POC.

    Args:
        query_embedding: Query embedding vector.
        limit: Maximum number of results.
        scope_key: Source document scope key.
        pipeline: Pipeline version in entity_metadata.

    Returns:
        List of chunk dicts with id, text, page info, score, rank.
    """
    session = get_session()
    try:
        query_str = """
            WITH q AS (SELECT CAST(:emb AS vector) AS qv)
            SELECT dc.id, dc.text, dc.physical_pdf_page,
                   dc.entity_metadata,
                   dc.chunk_index,
                   1 - (de.embedding <=> q.qv) AS score
            FROM document_chunks dc
            JOIN document_embeddings de ON de.chunk_id = dc.id
            JOIN source_documents sd ON dc.source_document_id = sd.id
            CROSS JOIN q
            WHERE sd.is_active = true
              AND sd.scope_key = :scope_key
              AND dc.entity_metadata->>'pipeline' = :pipeline
              AND de.model = :model
            ORDER BY de.embedding <=> q.qv
            LIMIT :limit
        """
        params: dict[str, Any] = {
            "emb": str(query_embedding),
            "scope_key": scope_key,
            "pipeline": pipeline,
            "model": get_llm_client().embedding_model,
            "limit": limit,
        }

        rows = session.execute(sqltext(query_str), params).fetchall()
        results = []
        for rank, r in enumerate(rows):
            metadata = r[3] or {}
            results.append(
                {
                    "rank": rank,
                    "id": r[0],
                    "text": r[1],
                    "physical_pdf_page": r[2],
                    "page_start": metadata.get("page_start", r[2]),
                    "page_end": metadata.get("page_end", r[2]),
                    "chunk_index": r[4],
                    "score": float(r[5]),
                    "evidence_id": f"kaplan_chunk_{r[0]}",
                }
            )
        return results
    finally:
        session.close()


class BaselineRetriever:
    """Vector-only retriever for the naive baseline."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Embed query and return top-k chunks by vector similarity."""
        client = get_llm_client()
        query_embedding = client.embed_one(query)
        return vector_search(query_embedding, limit=self.top_k)
