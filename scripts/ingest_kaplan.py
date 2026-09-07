#!/usr/bin/env python3
"""Ingest Kaplan Chapter 33."""
import logging
from psych_qa.ingestion.kaplan import ingest_kaplan
from psych_qa.ingestion.embeddings import embed_kaplan_chunks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ingest_kaplan()
    embed_result = embed_kaplan_chunks(source_doc_id=result["source_document_id"])
    result["embeddings"] = embed_result
    print(result)
