#!/usr/bin/env python3
"""Run retrieval comparison: baseline vs complex pipeline on the same gold set.

Three strategies compared:
  1. baseline-v0: vector-only (naive 500-token chunks)
  2. complex-hybrid: vector + FTS + RRF (section-aware chunks)
  3. complex-rerank: vector + FTS + RRF + LLM rerank

Usage:
  python scripts/run_comparison_eval.py
  python scripts/run_comparison_eval.py --skip-rerank  # faster, no LLM rerank
  COLUMNS=200 python scripts/run_comparison_eval.py   # wider tables
"""
import argparse
import logging

from psych_qa.evaluation.comparison_runner import run_comparison_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison retrieval eval")
    parser.add_argument("--gold-set", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--skip-rerank", action="store_true",
                        help="Skip LLM rerank pipeline (faster)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_comparison_eval(
        gold_set_path=args.gold_set,
        top_k=args.top_k,
        skip_rerank=args.skip_rerank,
    )
