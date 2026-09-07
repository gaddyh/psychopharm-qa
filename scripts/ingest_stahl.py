#!/usr/bin/env python3
"""Ingest Stahl data."""
import argparse
import logging
from psych_qa.ingestion.stahl import ingest_stahl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", default=None, help="Drug slug filter")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = ingest_stahl(drug_filter=args.drug)
    print(result)
