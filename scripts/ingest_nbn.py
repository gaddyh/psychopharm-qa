#!/usr/bin/env python3
"""Ingest NbN data."""
import logging
from psych_qa.ingestion.nbn import ingest_nbn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ingest_nbn()
    print(result)
