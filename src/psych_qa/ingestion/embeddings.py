"""Embedding generation for Kaplan chunks.

Uses OpenAI text-embedding-3-small (1536-dim).
Caches by input_hash to avoid re-embedding unchanged chunks.
Embeddings stored in document_embeddings table (separate from chunks).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session
from ..llm.client import get_llm_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def embed_kaplan_chunks(source_doc_id: int | None = None) -> dict[str, Any]:
    """Generate embeddings for all Kaplan chunks that don't have them yet.

    Args:
        source_doc_id: If set, only embed chunks from this source document.

    Returns:
        Summary dict with counts.
    """
    client = get_llm_client()
    session = get_session()

    try:
        # Find chunks without embeddings
        if source_doc_id:
            query = sqltext("""
                SELECT dc.id, dc.text, dc.input_hash
                FROM document_chunks dc
                WHERE dc.source_document_id = :sdid
                AND NOT EXISTS (
                    SELECT 1 FROM document_embeddings de
                    WHERE de.chunk_id = dc.id AND de.model = :model
                )
                ORDER BY dc.id
            """)
            rows = session.execute(query, {"sdid": source_doc_id, "model": client.embedding_model}).fetchall()
        else:
            query = sqltext("""
                SELECT dc.id, dc.text, dc.input_hash
                FROM document_chunks dc
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_embeddings de
                    WHERE de.chunk_id = dc.id AND de.model = :model
                )
                ORDER BY dc.id
            """)
            rows = session.execute(query, {"model": client.embedding_model}).fetchall()

        total = len(rows)
        logger.info(f"Embedding {total} chunks with {client.embedding_model}...")

        embedded = 0
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            texts = [r[1] for r in batch]

            # Generate embeddings
            embeddings = client.embed(texts)

            # Insert
            for (chunk_id, chunk_text, input_hash), embedding in zip(batch, embeddings):
                session.execute(
                    sqltext("""
                        INSERT INTO document_embeddings (chunk_id, model, dimensions, input_hash, embedding, embedded_at)
                        VALUES (:cid, :model, :dims, :ihash, :emb, :ts)
                        ON CONFLICT (chunk_id, model) DO NOTHING
                    """),
                    {
                        "cid": chunk_id,
                        "model": client.embedding_model,
                        "dims": client.embedding_dimensions,
                        "ihash": input_hash,
                        "emb": str(embedding),
                        "ts": datetime.utcnow(),
                    },
                )
                embedded += 1

            session.commit()
            logger.info(f"Embedded {min(i + BATCH_SIZE, total)}/{total}")

        return {"total": total, "embedded": embedded, "model": client.embedding_model}

    except Exception as e:
        session.rollback()
        logger.error(f"Embedding failed: {e}")
        raise
    finally:
        session.close()
