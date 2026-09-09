"""Chunks repository — Kaplan vector/keyword search."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session
from ..llm.client import get_llm_client


def vector_search(query_embedding: list[float], limit: int = 10, drug_id: int | None = None,
                   source_document_id: int | None = None) -> list[dict[str, Any]]:
    """Vector similarity search using pgvector."""
    session = get_session()
    try:
        # Use CAST(... AS vector) to avoid :param::type syntax conflict
        query_str = """
            WITH q AS (SELECT CAST(:emb AS vector) AS qv)
            SELECT dc.id, dc.text, dc.physical_pdf_page, dc.entity_metadata,
                   dc.chunk_index, dc.section_id,
                   1 - (de.embedding <=> q.qv) as score
            FROM document_chunks dc
            JOIN document_embeddings de ON de.chunk_id = dc.id
            JOIN source_documents sd ON dc.source_document_id = sd.id
            CROSS JOIN q
            WHERE sd.is_active = true AND de.model = :model
        """
        params: dict[str, Any] = {
            "emb": str(query_embedding),
            "model": get_llm_client().embedding_model,
        }
        if drug_id:
            query_str += " AND dc.entity_metadata->'drug_ids' @> CAST(:did AS jsonb)"
            params["did"] = f'[{drug_id}]'
        if source_document_id:
            query_str += " AND dc.source_document_id = :sdid"
            params["sdid"] = source_document_id
        query_str += " ORDER BY de.embedding <=> q.qv LIMIT :limit"
        params["limit"] = limit

        rows = session.execute(sqltext(query_str), params).fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "physical_pdf_page": r[2],
                "entity_metadata": r[3],
                "chunk_index": r[4],
                "section_id": r[5],
                "score": float(r[6]),
                "evidence_id": f"kaplan_chunk_{r[0]}",
            }
            for r in rows
        ]
    finally:
        session.close()


def fulltext_search(query: str, limit: int = 10, drug_id: int | None = None,
                    source_document_id: int | None = None) -> list[dict[str, Any]]:
    """PostgreSQL full-text search using tsvector."""
    session = get_session()
    try:
        query_str = """
            SELECT dc.id, dc.text, dc.physical_pdf_page, dc.entity_metadata,
                   dc.chunk_index, dc.section_id,
                   ts_rank(dc.search_vector, plainto_tsquery('english', :q)) as score
            FROM document_chunks dc
            JOIN source_documents sd ON dc.source_document_id = sd.id
            WHERE sd.is_active = true
            AND dc.search_vector @@ plainto_tsquery('english', :q)
        """
        params: dict[str, Any] = {"q": query}
        if drug_id:
            query_str += " AND dc.entity_metadata->'drug_ids' @> CAST(:did AS jsonb)"
            params["did"] = f'[{drug_id}]'
        if source_document_id:
            query_str += " AND dc.source_document_id = :sdid"
            params["sdid"] = source_document_id
        query_str += " ORDER BY score DESC LIMIT :limit"
        params["limit"] = limit

        rows = session.execute(sqltext(query_str), params).fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "physical_pdf_page": r[2],
                "entity_metadata": r[3],
                "chunk_index": r[4],
                "section_id": r[5],
                "score": float(r[6]),
                "evidence_id": f"kaplan_chunk_{r[0]}",
            }
            for r in rows
        ]
    finally:
        session.close()


def get_chunk_by_id(chunk_id: int) -> dict[str, Any] | None:
    """Get a single chunk by ID."""
    session = get_session()
    try:
        row = session.execute(
            sqltext("""
                SELECT dc.id, dc.text, dc.physical_pdf_page, dc.entity_metadata,
                       dc.chunk_index, dc.section_id
                FROM document_chunks dc
                WHERE dc.id = :id
            """),
            {"id": chunk_id},
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "text": row[1],
            "physical_pdf_page": row[2],
            "entity_metadata": row[3],
            "chunk_index": row[4],
            "section_id": row[5],
            "evidence_id": f"kaplan_chunk_{row[0]}",
        }
    finally:
        session.close()
