#!/usr/bin/env python3
"""Run retrieval evaluation against the OCD gold set.

Gold question → Kaplan vector search → Top 5/10/20 chunks →
compare with gold evidence (page + anchor) → Evidence Recall@K.
"""
import argparse
import logging

from psych_qa.evaluation.retrieval_runner import run_retrieval_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval eval")
    parser.add_argument("--gold-set", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_retrieval_eval(gold_set_path=args.gold_set, top_k=args.top_k)
