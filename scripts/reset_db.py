#!/usr/bin/env python3
"""Reset the local POC database: export existing data, drop, recreate, re-ingest.

This is a destructive operation intended ONLY for the local POC environment.
It exports answer_traces, doctor_feedback, and conversations to a timestamped
JSON backup, records source-document metadata, drops the schema, recreates it
from the finalized ORM models, and re-ingests all sources.

Usage:
    python scripts/reset_db.py --confirm-local-reset

Safety:
    - Refuses to run unless settings.environment == "local"
    - Requires explicit --confirm-local-reset flag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text as sqltext

from psych_qa.config import PROJECT_ROOT, get_settings
from psych_qa.db.connection import get_engine
from psych_qa.db.tables import Base, init_db


def _export_table(conn, table: str) -> list[dict]:
    """Export all rows from a table as list of dicts."""
    rows = conn.execute(sqltext(f"SELECT * FROM {table} ORDER BY id")).fetchall()
    cols = list(rows[0]._mapping.keys()) if rows else []
    return [dict(zip(cols, r)) for r in rows]


def _serialize(obj):
    """Make JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def export_existing_data(backup_path: Path) -> None:
    """Export traces, feedback, conversations, and source-doc metadata to JSON."""
    engine = get_engine()
    with engine.connect() as conn:
        backup = {
            "exported_at": datetime.utcnow().isoformat(),
            "source_documents": _export_table(conn, "source_documents"),
            "ingestion_runs": _export_table(conn, "ingestion_runs"),
            "conversations": _export_table(conn, "conversations"),
            "answer_traces": _export_table(conn, "answer_traces"),
            "doctor_feedback": _export_table(conn, "doctor_feedback"),
        }
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w") as f:
        json.dump(_serialize(backup), f, indent=2, default=str)
    print(f"  Exported {len(backup['answer_traces'])} traces, "
          f"{len(backup['doctor_feedback'])} feedback, "
          f"{len(backup['conversations'])} conversations")
    print(f"  Backup written to: {backup_path}")


def drop_and_recreate() -> None:
    """Drop all tables and recreate from ORM models."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("  Dropped all tables.")
    init_db()
    print("  Recreated schema from ORM models (init_db).")


def reingest_all() -> dict:
    """Re-ingest NbN, Stahl (amisulpride), and Kaplan with embeddings."""
    from psych_qa.ingestion.pipeline import run_all
    print("  Ingesting NbN + Stahl (amisulpride) + Kaplan ...")
    results = run_all()
    for source, result in results.items():
        rec = result.get("records_written", result.get("chunks_written", "?"))
        print(f"    {source}: {rec} records written")
    return results


def main():
    parser = argparse.ArgumentParser(description="Reset local POC database")
    parser.add_argument(
        "--confirm-local-reset",
        action="store_true",
        help="Required flag to confirm the destructive reset",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not args.confirm_local_reset:
        print("ERROR: --confirm-local-reset flag is required.")
        print("This is a destructive operation. Run:")
        print("  python scripts/reset_db.py --confirm-local-reset")
        sys.exit(1)

    if settings.environment != "local":
        print(f"ERROR: Database reset is allowed only in local environment.")
        print(f"  Current environment: {settings.environment}")
        print(f"  Set ENVIRONMENT=local in .env to allow.")
        sys.exit(1)

    print(f"Resetting local POC database: {settings.database_url}")
    print()

    # 1. Export existing data
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = PROJECT_ROOT / "data" / "backups" / f"db_backup_{timestamp}.json"
    print(f"[1/3] Exporting existing data ...")
    export_existing_data(backup_path)

    # 2. Drop and recreate
    print(f"\n[2/3] Dropping and recreating schema ...")
    drop_and_recreate()

    # 3. Re-ingest
    print(f"\n[3/3] Re-ingesting all sources ...")
    reingest_all()

    print(f"\nDone. Database reset and re-ingested.")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
