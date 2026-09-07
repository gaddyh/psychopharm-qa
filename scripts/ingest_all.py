#!/usr/bin/env python3
"""Run complete ingestion pipeline."""
import logging
from psych_qa.ingestion.pipeline import run_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    results = run_all()
    for source, result in results.items():
        print(f"\n{source}: {result}")
