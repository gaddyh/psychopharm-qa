#!/usr/bin/env python3
"""Ingest Kaplan Chapter 33 with the naive baseline pipeline.

Fixed-size 500-token chunks, 400-token step (100 overlap).
Idempotent: re-running replaces existing baseline-v0 chunks.
"""
import argparse
import logging

from psych_qa.baseline.ingest import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_STEP_TOKENS,
    ingest_kaplan_baseline,
)
from psych_qa.ingestion.embeddings import embed_kaplan_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Kaplan baseline")
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--step-tokens", type=int, default=DEFAULT_STEP_TOKENS)
    parser.add_argument("--start-page", type=int, default=None)
    parser.add_argument("--end-page", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    page_range = None
    if args.start_page is not None and args.end_page is not None:
        page_range = (args.start_page, args.end_page)

    result = ingest_kaplan_baseline(
        chunk_tokens=args.chunk_tokens,
        step_tokens=args.step_tokens,
        page_range=page_range,
    )
    print("\n--- Ingest Summary ---")
    for k, v in result.items():
        if k == "avg_chunk_tokens":
            print(f"  {k}: {v:.1f}")
        elif k == "multi_page_pct":
            print(f"  {k}: {v:.1f}%")
        else:
            print(f"  {k}: {v}")

    print("\n--- Embedding ---")
    embed_result = embed_kaplan_chunks(source_doc_id=result["source_document_id"])
    print(f"  embedded: {embed_result['embedded']}/{embed_result['total']}")
    print(f"  model: {embed_result['model']}")
