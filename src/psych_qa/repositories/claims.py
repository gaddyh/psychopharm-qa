"""Claims repository — Stahl/NbN source assertions."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session


def get_claims_for_drug(
    drug_id: int,
    source: str | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get all claims for a drug from active source documents.

    Args:
        drug_id: The drug ID.
        source: Filter by source type ('nbn' or 'stahl'). None = all.
        categories: Filter by categories. None = all.
    """
    session = get_session()
    try:
        query = """
            SELECT sc.id, sc.category, sc.text, sc.attributes, sc.locator,
                   sd.source_type, sc.drug_variant_id
            FROM source_claims sc
            JOIN source_documents sd ON sc.source_document_id = sd.id
            WHERE sc.drug_id = :did AND sd.is_active = true
        """
        params: dict[str, Any] = {"did": drug_id}
        if source:
            query += " AND sd.source_type = :source"
            params["source"] = source
        if categories:
            query += " AND sc.category = ANY(:cats)"
            params["cats"] = categories
        query += " ORDER BY sd.source_type, sc.category, sc.id"

        rows = session.execute(sqltext(query), params).fetchall()
        return [
            {
                "id": r[0],
                "category": r[1],
                "text": r[2],
                "attributes": r[3],
                "locator": r[4],
                "source": r[5],
                "drug_variant_id": r[6],
                "evidence_id": f"{r[5]}_claim_{r[0]}",
            }
            for r in rows
        ]
    finally:
        session.close()


def get_claim_by_id(claim_id: int) -> dict[str, Any] | None:
    """Get a single claim by ID."""
    session = get_session()
    try:
        row = session.execute(
            sqltext("""
                SELECT sc.id, sc.category, sc.text, sc.attributes, sc.locator,
                       sd.source_type, sc.drug_id
                FROM source_claims sc
                JOIN source_documents sd ON sc.source_document_id = sd.id
                WHERE sc.id = :id
            """),
            {"id": claim_id},
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "category": row[1],
            "text": row[2],
            "attributes": row[3],
            "locator": row[4],
            "source": row[5],
            "drug_id": row[6],
            "evidence_id": f"{row[5]}_claim_{row[0]}",
        }
    finally:
        session.close()
