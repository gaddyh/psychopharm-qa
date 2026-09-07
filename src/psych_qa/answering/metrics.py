"""Run metrics — per-question evaluation metrics for the answering pipeline.

Computes metrics across 4 dimensions:
1. Retrieval: evidence gathered from each source
2. Citation: how the LLM used the evidence
3. Source coverage: did all 3 sources contribute?
4. Performance: latency breakdown

Metrics are stored in the answer trace (metrics JSONB column) and can be
aggregated across runs for evaluation.
"""

from __future__ import annotations

from typing import Any


def compute_run_metrics(
    question: str,
    understanding: dict[str, Any],
    evidence_package: dict[str, Any],
    answer: dict[str, Any],
    invalid_citation_ids: list[str],
    llm_latency_ms: int | None,
    total_latency_ms: int | None,
) -> dict[str, Any]:
    """Compute per-run metrics for a single question.

    Returns a flat dict of metrics suitable for JSONB storage and aggregation.
    """
    # --- Retrieval metrics ---
    drugs = evidence_package.get("drugs", {})
    nbn_provided = sum(len(d.get("nbn_evidence", [])) for d in drugs.values())
    stahl_provided = sum(len(d.get("stahl_evidence", [])) for d in drugs.values())
    kaplan_provided = len(evidence_package.get("kaplan_passages", []))
    total_evidence_provided = nbn_provided + stahl_provided + kaplan_provided

    drugs_found = sum(1 for d in drugs.values() if d.get("nbn_evidence") or d.get("stahl_evidence"))
    drugs_requested = len(understanding.get("drug_names", []))

    kaplan_scores = [p.get("attributes", {}).get("score", 0) for p in evidence_package.get("kaplan_passages", [])]
    kaplan_top_score = max(kaplan_scores) if kaplan_scores else 0.0
    kaplan_mean_score = sum(kaplan_scores) / len(kaplan_scores) if kaplan_scores else 0.0

    conflicts = len(evidence_package.get("conflicts_or_uncertainties", []))

    # --- Citation metrics ---
    citations = answer.get("citations", [])
    all_cited_ids: list[str] = []
    claims_with_citations = 0
    for c in citations:
        eids = c.get("evidence_ids", [])
        all_cited_ids.extend(eids)
        if eids:
            claims_with_citations += 1

    unique_cited_ids = set(all_cited_ids)
    total_claims = len(citations)
    citation_coverage = claims_with_citations / total_claims if total_claims > 0 else 0.0

    # Evidence utilization: how much of the provided evidence was actually cited
    evidence_utilization = len(unique_cited_ids) / total_evidence_provided if total_evidence_provided > 0 else 0.0

    # Invalid citations (hallucinated IDs)
    invalid_citation_count = len(invalid_citation_ids)
    citation_validity = 1.0 if invalid_citation_count == 0 else 1.0 - (invalid_citation_count / max(len(all_cited_ids), 1))

    # --- Source coverage ---
    nbn_cited = sum(1 for eid in unique_cited_ids if eid.startswith("nbn_"))
    stahl_cited = sum(1 for eid in unique_cited_ids if eid.startswith("stahl_"))
    kaplan_cited = sum(1 for eid in unique_cited_ids if eid.startswith("kaplan_"))
    sources_used = sum([nbn_cited > 0, stahl_cited > 0, kaplan_cited > 0])

    # --- Answer quality ---
    status = answer.get("status", "unknown")
    has_uncertainty = bool(answer.get("uncertainty"))
    has_clarification = bool(answer.get("clarification_question"))
    direct_answer_chars = len(answer.get("direct_answer", ""))
    explanation_chars = len(answer.get("explanation", ""))

    # --- Performance ---
    llm_ms = llm_latency_ms or 0
    total_ms = total_latency_ms or 0
    retrieval_ms = max(total_ms - llm_ms, 0) if total_ms else 0

    return {
        # Retrieval
        "drugs_requested": drugs_requested,
        "drugs_found": drugs_found,
        "drug_lookup_success_rate": drugs_found / drugs_requested if drugs_requested > 0 else 0.0,
        "nbn_evidence_provided": nbn_provided,
        "stahl_evidence_provided": stahl_provided,
        "kaplan_evidence_provided": kaplan_provided,
        "total_evidence_provided": total_evidence_provided,
        "kaplan_top_score": round(kaplan_top_score, 4),
        "kaplan_mean_score": round(kaplan_mean_score, 4),
        "conflicts_detected": conflicts,
        # Citation
        "total_claims": total_claims,
        "claims_with_citations": claims_with_citations,
        "citation_coverage": round(citation_coverage, 4),
        "unique_evidence_ids_cited": len(unique_cited_ids),
        "evidence_utilization": round(evidence_utilization, 4),
        "invalid_citation_count": invalid_citation_count,
        "citation_validity": round(citation_validity, 4),
        # Source coverage
        "nbn_cited": nbn_cited,
        "stahl_cited": stahl_cited,
        "kaplan_cited": kaplan_cited,
        "sources_used": sources_used,
        # Answer quality
        "status": status,
        "has_uncertainty": has_uncertainty,
        "has_clarification": has_clarification,
        "direct_answer_chars": direct_answer_chars,
        "explanation_chars": explanation_chars,
        # Performance
        "llm_latency_ms": llm_ms,
        "total_latency_ms": total_ms,
        "retrieval_latency_ms": retrieval_ms,
    }


def compute_aggregate_metrics(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across multiple runs.

    Returns summary stats (mean, min, max) for numeric metrics.
    """
    if not run_metrics:
        return {"total_runs": 0}

    numeric_keys = [
        "drug_lookup_success_rate", "nbn_evidence_provided", "stahl_evidence_provided",
        "kaplan_evidence_provided", "total_evidence_provided", "kaplan_top_score",
        "kaplan_mean_score", "conflicts_detected", "total_claims", "claims_with_citations",
        "citation_coverage", "unique_evidence_ids_cited", "evidence_utilization",
        "invalid_citation_count", "citation_validity", "nbn_cited", "stahl_cited",
        "kaplan_cited", "sources_used", "direct_answer_chars", "explanation_chars",
        "llm_latency_ms", "total_latency_ms", "retrieval_latency_ms",
    ]

    agg: dict[str, Any] = {"total_runs": len(run_metrics)}

    for key in numeric_keys:
        values = [r.get(key, 0) for r in run_metrics if key in r]
        if values:
            agg[f"{key}_mean"] = round(sum(values) / len(values), 4)
            agg[f"{key}_min"] = min(values)
            agg[f"{key}_max"] = max(values)

    # Status distribution
    statuses = [r.get("status", "unknown") for r in run_metrics]
    agg["status_distribution"] = {s: statuses.count(s) for s in set(statuses)}

    # Source usage distribution
    source_counts = [r.get("sources_used", 0) for r in run_metrics]
    agg["source_usage_distribution"] = {n: source_counts.count(n) for n in set(source_counts)}

    # Abstention rate
    agg["abstention_rate"] = statuses.count("abstained") / len(statuses)

    # Avg citation validity
    validities = [r.get("citation_validity", 1.0) for r in run_metrics]
    agg["avg_citation_validity"] = round(sum(validities) / len(validities), 4)

    return agg
