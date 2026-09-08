"""Retrieval evaluation runner.

Orchestrates: load gold set → embed questions → vector search →
match evidence → compute metrics → save results with full provenance.
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

from ..baseline.retrieve import BaselineRetriever
from ..config import get_settings
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


def _git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def _gold_set_hash(yaml_path: Path) -> str:
    """SHA-256 hash of the gold set file."""
    h = hashlib.sha256()
    with open(yaml_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def _corpus_hash() -> str | None:
    """Get content hash of the active baseline source document from DB."""
    from sqlalchemy import text as sqltext
    from ..db.connection import get_session
    from ..baseline.ingest import SCOPE_KEY, SOURCE_TYPE

    session = get_session()
    try:
        row = session.execute(
            sqltext(
                "SELECT content_hash FROM source_documents "
                "WHERE source_type = :st AND scope_key = :sk AND is_active = true"
            ),
            {"st": SOURCE_TYPE, "sk": SCOPE_KEY},
        ).fetchone()
        return row[0][:16] if row else None
    finally:
        session.close()


def run_retrieval_eval(
    gold_set_path: Path | str | None = None,
    top_k: int = 20,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full retrieval evaluation.

    Args:
        gold_set_path: Path to the gold set YAML.
        top_k: Number of chunks to retrieve per question.
        output_path: Where to save results JSON. Defaults to evals/results/.

    Returns:
        Full results dict with provenance, per-case results, and aggregate metrics.
    """
    settings = get_settings()
    if gold_set_path is None:
        gold_set_path = settings.evals_dir / "kaplan_ocd_gold_set_v0.1.yaml"
    gold_set_path = Path(gold_set_path)

    # Load gold set
    logger.info(f"Loading gold set: {gold_set_path}")
    gold_set = load_gold_set(gold_set_path)
    logger.info(f"Loaded {len(gold_set.cases)} cases, {len(gold_set.evidence_catalog)} evidence items")

    # Retrieve
    retriever = BaselineRetriever(top_k=top_k)
    case_results: list[CaseResult] = []

    for case in gold_set.cases:
        logger.info(f"Retrieving for {case.case_id}: {case.question[:80]}...")
        t0 = time.time()
        retrieved = retriever.retrieve(case.question)
        latency_ms = int((time.time() - t0) * 1000)

        # Stamp latency on each chunk
        for chunk in retrieved:
            chunk["retrieval_latency_ms"] = latency_ms

        cr = compute_case_result(case, retrieved, gold_set, k_values=K_VALUES)
        case_results.append(cr)

        logger.info(
            f"  {case.case_id}: critical recall@10={cr.critical_recall_at(10):.1%}, "
            f"pass={cr.retrieval_pass}, "
            f"missed={cr.missed_critical_at_10}"
        )

    # Aggregate
    agg = compute_aggregate_metrics(case_results)

    # Build results with provenance
    results = {
        "gold_set_id": gold_set.meta.id,
        "gold_set_hash": _gold_set_hash(gold_set_path),
        "corpus_hash": _corpus_hash(),
        "git_commit": _git_commit(),
        "pipeline_version": "baseline-v0",
        "matcher_version": MATCHER_VERSION,
        "embedding_model": settings.openai_embedding_model,
        "chunk_tokens": 500,
        "overlap_tokens": 100,
        "top_k": top_k,
        "k_values": K_VALUES,
        "ran_at": datetime.utcnow().isoformat() + "Z",
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
            "total_all_pairs_at_10": agg.total_all_pairs_at_10,
            "matched_all_pairs_at_10": agg.matched_all_pairs_at_10,
        },
        "cases": [
            {
                "case_id": cr.case_id,
                "route": cr.route,
                "question": cr.question,
                "retrieval_pass": cr.retrieval_pass,
                "critical_recall_at_5": cr.critical_recall_at(5),
                "critical_recall_at_10": cr.critical_recall_at(10),
                "critical_recall_at_20": cr.critical_recall_at(20),
                "all_recall_at_5": cr.all_recall_at(5),
                "all_recall_at_10": cr.all_recall_at(10),
                "all_recall_at_20": cr.all_recall_at(20),
                "matched_critical_at_10": sorted(cr.matched_critical_at_k.get(10, set())),
                "missed_critical_at_10": cr.missed_critical_at_10,
                "matched_all_at_10": sorted(cr.matched_all_at_k.get(10, set())),
                "missed_all_at_10": cr.missed_all_at_10,
                "retrieved_chunks": [
                    {
                        "rank": c["rank"],
                        "id": c["id"],
                        "score": c["score"],
                        "page_start": c["page_start"],
                        "page_end": c["page_end"],
                        "text_preview": c["text"][:200],
                    }
                    for c in cr.retrieved_chunks
                ],
            }
            for cr in case_results
        ],
    }

    # Save
    if output_path is None:
        output_path = settings.evals_dir / "results"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"baseline_retrieval_{timestamp}.json"
    output_file = output_path / filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_file}")

    # Also write a 'latest' symlink/copy
    latest_path = output_path / "baseline_retrieval_latest.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print rich summary
    _print_rich_summary(results, case_results, agg, output_file, gold_set)

    return results


def _print_rich_summary(
    results: dict[str, Any],
    case_results: list[CaseResult],
    agg: AggregateMetrics,
    output_file: Path,
    gold_set: GoldSet,
) -> None:
    """Print rich tables with metrics, per-case results, and provenance."""

    # --- Provenance panel ---
    prov_lines = []
    for k in ["gold_set_id", "gold_set_hash", "corpus_hash", "git_commit",
              "pipeline_version", "matcher_version", "embedding_model"]:
        prov_lines.append(f"[cyan]{k:20s}[/cyan] {results[k]}")
    prov_lines.append(f"[cyan]chunk_tokens:       [/cyan] {results['chunk_tokens']}")
    prov_lines.append(f"[cyan]overlap_tokens:     [/cyan] {results['overlap_tokens']}")
    prov_lines.append(f"[cyan]top_k:              [/cyan] {results['top_k']}")
    prov_lines.append(f"[cyan]ran_at:            [/cyan] {results['ran_at']}")
    console.print(Panel("\n".join(prov_lines), title="Provenance", border_style="blue"))

    # --- Aggregate metrics table ---
    agg_table = Table(title="Aggregate Metrics", border_style="blue", show_lines=False)
    agg_table.add_column("Metric", style="cyan", no_wrap=True)
    agg_table.add_column("Value", justify="right", style="white")
    agg_table.add_column("Detail", justify="right", style="dim")

    passes = agg.retrieval_passes
    pass_style = "green" if passes == agg.n_cases else ("yellow" if passes > 0 else "red")

    rows = [
        ("Cases", str(agg.n_cases), ""),
        ("Retrieval passes", f"[{pass_style}]{passes}/{agg.n_cases}[/{pass_style}]", ""),
        ("", "", ""),
        ("Macro critical Recall@5", f"{agg.macro_critical_recall_at_5:.1%}", ""),
        ("Macro critical Recall@10", f"{agg.macro_critical_recall_at_10:.1%}", "PRIMARY"),
        ("Macro critical Recall@20", f"{agg.macro_critical_recall_at_20:.1%}", ""),
        ("", "", ""),
        ("Macro all-evidence Recall@5", f"{agg.macro_all_recall_at_5:.1%}", ""),
        ("Macro all-evidence Recall@10", f"{agg.macro_all_recall_at_10:.1%}", ""),
        ("Macro all-evidence Recall@20", f"{agg.macro_all_recall_at_20:.1%}", ""),
        ("", "", ""),
        ("Micro critical Recall@10", f"{agg.micro_critical_recall_at_10:.1%}",
         f"{agg.matched_critical_pairs_at_10}/{agg.total_critical_pairs_at_10}"),
        ("Micro all-evidence Recall@10", f"{agg.micro_all_recall_at_10:.1%}",
         f"{agg.matched_all_pairs_at_10}/{agg.total_all_pairs_at_10}"),
    ]
    for metric, value, detail in rows:
        if metric == "":
            agg_table.add_row("", "", "")
        else:
            style = "bold green" if "PRIMARY" in detail else ""
            agg_table.add_row(metric, value, detail, style=style)

    console.print()
    console.print(agg_table)

    # --- Per-case table ---
    case_table = Table(title="Per-Case Results", border_style="blue", show_lines=True)
    case_table.add_column("Case ID", style="cyan", no_wrap=True)
    case_table.add_column("Route", style="dim", no_wrap=True)
    case_table.add_column("Pass", justify="center")
    case_table.add_column("Crit@5", justify="right")
    case_table.add_column("Crit@10", justify="right")
    case_table.add_column("Crit@20", justify="right")
    case_table.add_column("All@5", justify="right")
    case_table.add_column("All@10", justify="right")
    case_table.add_column("All@20", justify="right")
    case_table.add_column("Missed Critical @10", style="red")

    for cr in case_results:
        pass_str = "[green]PASS[/green]" if cr.retrieval_pass else "[red]FAIL[/red]"
        missed = ", ".join(eid.replace("K33-P", "P") for eid in cr.missed_critical_at_10)
        if not missed:
            missed = "[green]—[/green]"

        case_table.add_row(
            cr.case_id,
            cr.route,
            pass_str,
            f"{cr.critical_recall_at(5):.0%}",
            f"{cr.critical_recall_at(10):.0%}",
            f"{cr.critical_recall_at(20):.0%}",
            f"{cr.all_recall_at(5):.0%}",
            f"{cr.all_recall_at(10):.0%}",
            f"{cr.all_recall_at(20):.0%}",
            missed,
        )

    console.print()
    console.print(case_table)

    # --- Missed evidence detail ---
    evidence_by_id = gold_set.evidence_by_id
    missed_table = Table(title="Missed Critical Evidence Detail", border_style="red", show_lines=True)
    missed_table.add_column("Case", style="cyan", no_wrap=True)
    missed_table.add_column("Evidence ID", style="red", no_wrap=True)
    missed_table.add_column("Pages", justify="center")
    missed_table.add_column("Anchor (first)", style="dim")

    for case_dict in results["cases"]:
        case_id = case_dict["case_id"]
        for eid in case_dict.get("missed_critical_at_10", []):
            ev = evidence_by_id.get(eid)
            if ev:
                pages_str = str(ev.pdf_pages)
                anchor = ev.anchor_text[0][:60] + "..." if ev.anchor_text else ""
                missed_table.add_row(case_id, eid, pages_str, anchor)

    console.print()
    console.print(missed_table)
    console.print()
    console.print(f"[dim]Results saved to:[/dim] [bold]{output_file}[/bold]")
