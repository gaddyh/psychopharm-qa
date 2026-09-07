"""Drug repository — deterministic drug queries."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session


def find_by_slug(slug: str) -> dict[str, Any] | None:
    """Find a drug by its slug."""
    session = get_session()
    try:
        row = session.execute(
            sqltext("SELECT id, canonical_name, slug, gtopdb_ligand_id, constituents FROM drugs WHERE slug = :s"),
            {"s": slug},
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "canonical_name": row[1], "slug": row[2], "gtopdb_ligand_id": row[3], "constituents": row[4]}
    finally:
        session.close()


def find_by_name(name: str) -> dict[str, Any] | None:
    """Find a drug by canonical name or alias. Returns candidates."""
    session = get_session()
    try:
        # Try slug first
        import re
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        row = session.execute(
            sqltext("SELECT id, canonical_name, slug FROM drugs WHERE slug = :s"),
            {"s": slug},
        ).fetchone()
        if row:
            return {"id": row[0], "canonical_name": row[1], "slug": row[2]}

        # Try alias
        row = session.execute(
            sqltext("""
                SELECT d.id, d.canonical_name, d.slug
                FROM drugs d
                JOIN drug_aliases a ON a.drug_id = d.id
                WHERE a.normalized_alias = :na
                LIMIT 1
            """),
            {"na": slug},
        ).fetchone()
        if row:
            return {"id": row[0], "canonical_name": row[1], "slug": row[2]}
        return None
    finally:
        session.close()


def get_variants(drug_id: int) -> list[dict[str, Any]]:
    """Get all variants for a drug."""
    session = get_session()
    try:
        rows = session.execute(
            sqltext("SELECT id, variant_key, label FROM drug_variants WHERE drug_id = :d ORDER BY variant_key"),
            {"d": drug_id},
        ).fetchall()
        return [{"id": r[0], "variant_key": r[1], "label": r[2]} for r in rows]
    finally:
        session.close()
