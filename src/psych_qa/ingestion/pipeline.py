"""Ingestion pipeline orchestrator.

Usage:
    python -m psych_qa.ingestion.pipeline --source nbn
    python -m psych_qa.ingestion.pipeline --source stahl --drug amisulpride
    python -m psych_qa.ingestion.pipeline --source kaplan
    python -m psych_qa.ingestion.pipeline --source all
"""

from __future__ import annotations

import argparse
import logging
import sys

from .embeddings import embed_kaplan_chunks
from .kaplan import ingest_kaplan
from .nbn import ingest_nbn
from .stahl import ingest_stahl

logger = logging.getLogger(__name__)


def run_nbn() -> dict:
    return ingest_nbn()


def run_stahl(drug: str | None = None) -> dict:
    return ingest_stahl(drug_filter=drug)


def run_kaplan() -> dict:
    result = ingest_kaplan()
    # Also embed the chunks
    embed_result = embed_kaplan_chunks(source_doc_id=result["source_document_id"])
    result["embeddings"] = embed_result
    return result


def run_all() -> dict:
    results = {}
    logger.info("=== Ingesting NbN ===")
    results["nbn"] = run_nbn()
    logger.info("=== Ingesting Stahl (amisulpride) ===")
    results["stahl"] = run_stahl(drug="amisulpride")
    logger.info("=== Ingesting Kaplan ===")
    results["kaplan"] = run_kaplan()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument(
        "--source",
        choices=["nbn", "stahl", "kaplan", "all"],
        default="all",
        help="Which source to ingest",
    )
    parser.add_argument("--drug", default=None, help="Drug filter for Stahl (slug)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s")

    if args.source == "nbn":
        result = run_nbn()
    elif args.source == "stahl":
        result = run_stahl(drug=args.drug)
    elif args.source == "kaplan":
        result = run_kaplan()
    else:
        result = run_all()

    print(result)


if __name__ == "__main__":
    main()
