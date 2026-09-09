"""Comparison runner: baseline vs complex pipeline on the same gold set.

Runs three retrieval strategies against the same 7 gold questions:
  1. Baseline: vector-only (naive 500-token chunks)
  2. Complex: vector + FTS + RRF (section-aware chunks)
  3. Complex + rerank: vector + FTS + RRF + LLM rerank

Page translation: original chunks use book pages (+9785 offset from
chapter-only 0-based pages). The adapter translates book pages back to
chapter pages so the same matcher works unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import text as sqltext

from ..baseline.retrieve import BaselineRetriever
from ..config import get_settings
from ..db.connection import get_session
from ..retrieval.chapter_search import KaplanRetriever
from ..retrieval.reranker import rerank
from .gold_loader import GoldSet, load_gold_set
from .metrics import (
    AggregateMetrics,
    CaseResult,
    compute_aggregate_metrics,
    compute_case_result,
)

logger = logging.getLogger(__name__)
console = Console()

MATCHER_VERSION = "page-anchor-v1"
K_VALUES = [5, 10, 20]
BOOK_PAGE_OFFSET = 9785  # book_page = chapter_page + 9785


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def _gold_set_hash(yaml_path: Path) -> str:
    h = hashlib.sha256()
    with open(yaml_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def _translate_book_pages(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate book page numbers to chapter 0-based page numbers.

    The original Kaplan ingestion used book page numbers (9786-11299).
    The gold set uses 0-based chapter page numbers (0-1508).
    Offset: chapter_page = book_page - 9785.
    """
    translated = []
    for chunk in chunks:
        c = dict(chunk)
        book_page = c.get("physical_pdf_page", 0)
        chapter_page = book_page - BOOK_PAGE_OFFSET
        c["page_start"] = chapter_page
        c["page_end"] = chapter_page
        # Also extract from metadata if available
        md = c.get("entity_metadata") or {}
        if "page_start" in md:
            c["page_start"] = md["page_start"] - BOOK_PAGE_OFFSET
        if "page_end" in md:
            c["page_end"] = md["page_end"] - BOOK_PAGE_OFFSET
        translated.append(c)
    return translated


class ComplexRetriever:
    """Hybrid retriever: vector + FTS + RRF, no drug filtering."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.retriever = KaplanRetriever(top_k=top_k)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        results = self.retriever.retrieve(query, drug_ids=None)
        return _translate_book_pages(results)


class ComplexRerankedRetriever:
    """Hybrid retriever + LLM rerank, no drug filtering."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.retriever = KaplanRetriever(top_k=top_k)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        results = self.retriever.retrieve(query, drug_ids=None)
        # Rerank with LLM (fetch top_k * 2 candidates, rerank to top_k)
        reranked = rerank(query, results, top_k=self.top_k)
        return _translate_book_pages(reranked)


def _run_pipeline(
    name: str,
    retriever,
    gold_set: GoldSet,
    k_values: list[int] = K_VALUES,
) -> tuple[list[CaseResult], AggregateMetrics]:
    """Run a single pipeline against all gold cases."""
    case_results: list[CaseResult] = []

    for case in gold_set.cases:
        logger.info(f"  [{name}] {case.case_id}: {case.question[:80]}...")
        t0 = time.time()
        retrieved = retriever.retrieve(case.question)
        latency_ms = int((time.time() - t0) * 1000)

        for chunk in retrieved:
            chunk["retrieval_latency_ms"] = latency_ms

        cr = compute_case_result(case, retrieved, gold_set, k_values=k_values)
        case_results.append(cr)

        logger.info(
            f"    {case.case_id}: crit@10={cr.critical_recall_at(10):.1%}, "
            f"pass={cr.retrieval_pass}"
        )

    agg = compute_aggregate_metrics(case_results)
    return case_results, agg


def run_comparison_eval(
    gold_set_path: Path | str | None = None,
    top_k: int = 20,
    output_path: Path | None = None,
    skip_rerank: bool = False,
) -> dict[str, Any]:
    """Run baseline vs complex pipeline comparison.

    Args:
        gold_set_path: Path to gold set YAML.
        top_k: Number of chunks to retrieve per question.
        output_path: Where to save results JSON.
        skip_rerank: Skip the LLM rerank pipeline (faster, no API cost).

    Returns:
        Full comparison results dict.
    """
    settings = get_settings()
    if gold_set_path is None:
        gold_set_path = settings.evals_dir / "kaplan_ocd_gold_set_v0.1.yaml"
    gold_set_path = Path(gold_set_path)

    logger.info(f"Loading gold set: {gold_set_path}")
    gold_set = load_gold_set(gold_set_path)
    logger.info(f"Loaded {len(gold_set.cases)} cases, {len(gold_set.evidence_catalog)} evidence items")

    pipelines: dict[str, Any] = {
        "baseline-v0": BaselineRetriever(top_k=top_k),
        "complex-hybrid": ComplexRetriever(top_k=top_k),
    }
    if not skip_rerank:
        pipelines["complex-rerank"] = ComplexRerankedRetriever(top_k=top_k)

    all_results: dict[str, dict[str, Any]] = {}

    for name, retriever in pipelines.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running pipeline: {name}")
        logger.info(f"{'='*60}")
        case_results, agg = _run_pipeline(name, retriever, gold_set)
        all_results[name] = {
            "case_results": case_results,
            "aggregate": agg,
        }

    # Build results dict
    results = {
        "gold_set_id": gold_set.meta.id,
        "gold_set_hash": _gold_set_hash(gold_set_path),
        "git_commit": _git_commit(),
        "matcher_version": MATCHER_VERSION,
        "embedding_model": settings.openai_embedding_model,
        "top_k": top_k,
        "k_values": K_VALUES,
        "ran_at": datetime.utcnow().isoformat() + "Z",
        "pipelines": {},
    }

    for name, data in all_results.items():
        agg = data["aggregate"]
        case_results = data["case_results"]
        results["pipelines"][name] = {
            "aggregate": {
                "n_cases": agg.n_cases,
                "retrieval_passes": agg.retrieval_passes,
                "macro_critical_recall_at_5": agg.macro_critical_recall_at_5,
                "macro_critical_recall_at_10": agg.macro_critical_recall_at_10,
                "macro_critical_recall_at_20": agg.macro_critical_recall_at_20,
                "macro_all_recall_at_5": agg.macro_all_recall_at_5,
                "macro_all_recall_at_10": agg.macro_all_recall_at_10,
                "macro_all_recall_at_20": agg.macro_all_recall_at_20,
                "micro_critical_recall_at_10": agg.micro_critical_recall_at_10,
                "micro_all_recall_at_10": agg.micro_all_recall_at_10,
                "total_critical_pairs_at_10": agg.total_critical_pairs_at_10,
                "matched_critical_pairs_at_10": agg.matched_critical_pairs_at_10,
            },
            "cases": [
                {
                    "case_id": cr.case_id,
                    "retrieval_pass": cr.retrieval_pass,
                    "critical_recall_at_5": cr.critical_recall_at(5),
                    "critical_recall_at_10": cr.critical_recall_at(10),
                    "critical_recall_at_20": cr.critical_recall_at(20),
                    "all_recall_at_5": cr.all_recall_at(5),
                    "all_recall_at_10": cr.all_recall_at(10),
                    "all_recall_at_20": cr.all_recall_at(20),
                    "missed_critical_at_10": cr.missed_critical_at_10,
                }
                for cr in case_results
            ],
        }

    # Save
    if output_path is None:
        output_path = settings.evals_dir / "results"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"comparison_retrieval_{timestamp}.json"
    output_file = output_path / filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_file}")

    latest_path = output_path / "comparison_retrieval_latest.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print rich comparison tables
    _print_comparison_tables(results, all_results, gold_set, output_file)

    return results


def _print_comparison_tables(
    results: dict[str, Any],
    all_results: dict[str, dict[str, Any]],
    gold_set: GoldSet,
    output_file: Path,
) -> None:
    """Print side-by-side rich comparison tables."""

    # --- Provenance panel ---
    prov_lines = []
    for k in ["gold_set_id", "gold_set_hash", "git_commit", "matcher_version", "embedding_model"]:
        prov_lines.append(f"[cyan]{k:20s}[/cyan] {results[k]}")
    prov_lines.append(f"[cyan]top_k:              [/cyan] {results['top_k']}")
    prov_lines.append(f"[cyan]ran_at:            [/cyan] {results['ran_at']}")
    console.print(Panel("\n".join(prov_lines), title="Provenance", border_style="blue"))

    pipeline_names = list(results["pipelines"].keys())

    # --- Aggregate comparison table ---
    agg_table = Table(title="Aggregate Metrics Comparison", border_style="blue", show_lines=False)
    agg_table.add_column("Metric", style="cyan", no_wrap=True)
    for name in pipeline_names:
        agg_table.add_column(name, justify="right")

    metrics_rows = [
        ("Retrieval passes", "retrieval_passes", "{}/{}", "n_cases"),
        ("", None, None, None),
        ("Macro critical Recall@5", "macro_critical_recall_at_5", "{:.1%}", None),
        ("Macro critical Recall@10", "macro_critical_recall_at_10", "{:.1%}", None),
        ("Macro critical Recall@20", "macro_critical_recall_at_20", "{:.1%}", None),
        ("", None, None, None),
        ("Macro all-evidence Recall@5", "macro_all_recall_at_5", "{:.1%}", None),
        ("Macro all-evidence Recall@10", "macro_all_recall_at_10", "{:.1%}", None),
        ("Macro all-evidence Recall@20", "macro_all_recall_at_20", "{:.1%}", None),
        ("", None, None, None),
        ("Micro critical Recall@10", "micro_critical_recall_at_10", "{:.1%}", None),
        ("Micro all-evidence Recall@10", "micro_all_recall_at_10", "{:.1%}", None),
    ]

    for label, key, fmt, denom_key in metrics_rows:
        if key is None:
            agg_table.add_row("", *["" for _ in pipeline_names])
            continue
        row_vals = [label]
        for name in pipeline_names:
            agg = results["pipelines"][name]["aggregate"]
            val = agg[key]
            if denom_key:
                row_vals.append(fmt.format(val, agg[denom_key]))
            else:
                row_vals.append(fmt.format(val))
        # Highlight the primary metric
        style = "bold green" if "Recall@10" in label and "critical" in label else ""
        agg_table.add_row(*row_vals, style=style)

    console.print()
    console.print(agg_table)

    # --- Per-case comparison table ---
    case_table = Table(title="Per-Case Critical Recall@10 Comparison", border_style="blue", show_lines=True)
    case_table.add_column("Case ID", style="cyan", no_wrap=True)
    case_table.add_column("Route", style="dim", no_wrap=True)
    for name in pipeline_names:
        case_table.add_column(f"{name}\nPass", justify="center")
        case_table.add_column(f"{name}\nCrit@10", justify="right")
        case_table.add_column(f"{name}\nCrit@20", justify="right")

    n_cases = len(results["pipelines"][pipeline_names[0]]["cases"])
    for i in range(n_cases):
        case_id = results["pipelines"][pipeline_names[0]]["cases"][i]["case_id"]
        route = next(c.route for c in gold_set.cases if c.case_id == case_id)
        row_vals = [case_id, route]
        for name in pipeline_names:
            case_data = results["pipelines"][name]["cases"][i]
            pass_str = "[green]PASS[/green]" if case_data["retrieval_pass"] else "[red]FAIL[/red]"
            crit_10 = case_data["critical_recall_at_10"]
            crit_20 = case_data["critical_recall_at_20"]
            row_vals.append(pass_str)
            row_vals.append(f"{crit_10:.0%}")
            row_vals.append(f"{crit_20:.0%}")
        case_table.add_row(*row_vals)

    console.print()
    console.print(case_table)

    # --- Missed evidence comparison ---
    missed_table = Table(title="Missed Critical Evidence @10 Comparison", border_style="red", show_lines=True)
    missed_table.add_column("Case", style="cyan", no_wrap=True)
    for name in pipeline_names:
        missed_table.add_column(name, style="red")

    for i in range(n_cases):
        case_id = results["pipelines"][pipeline_names[0]]["cases"][i]["case_id"]
        row_vals = [case_id]
        for name in pipeline_names:
            missed = results["pipelines"][name]["cases"][i]["missed_critical_at_10"]
            short = ", ".join(eid.replace("K33-P", "P") for eid in missed)
            if not short:
                short = "[green]—[/green]"
            row_vals.append(short)
        missed_table.add_row(*row_vals)

    console.print()
    console.print(missed_table)
    console.print()
    console.print(f"[dim]Results saved to:[/dim] [bold]{output_file}[/bold]")
