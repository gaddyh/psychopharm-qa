"""Run metrics — per-question evaluation metrics for the answering pipeline.

V2: computes metrics from claims-first answer format.
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
    entailment_verdicts: list[dict[str, Any]] | None = None,
    regeneration_attempts: int = 0,
    budget_diagnostics: dict[str, Any] | None = None,
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

    # --- Citation metrics (from claims) ---
    claims = answer.get("claims", [])
    all_cited_ids: list[str] = []
    claims_with_valid_citations = 0
    critical_claims = 0
    unsupported_critical_claims = 0

    valid_set = set(evidence_package.get("required_citations", []))

    for claim in claims:
        eids = claim.get("evidence_ids", [])
        all_cited_ids.extend(eids)
        valid_eids = [eid for eid in eids if eid in valid_set]
        has_valid = len(valid_eids) > 0
        if has_valid:
            claims_with_valid_citations += 1

        is_critical = claim.get("critical", False)
        if is_critical:
            critical_claims += 1
            if not has_valid:
                unsupported_critical_claims += 1

    unique_cited_ids = set(all_cited_ids)
    total_claims = len(claims)
    # Citation completeness: every claim must have valid evidence (target 100%)
    citation_completeness = claims_with_valid_citations / total_claims if total_claims > 0 else 0.0

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
    has_uncertainties = bool(answer.get("uncertainties"))
    has_clarification = bool(answer.get("clarification_question"))
    direct_answer_chars = len(answer.get("direct_answer", ""))
    explanation_chars = len(answer.get("explanation", ""))

    # --- Entailment metrics ---
    supported_claims = 0
    unsupported_specificity_count = 0
    contradicted_count = 0
    if entailment_verdicts:
        for v in entailment_verdicts:
            verdict = v.get("verdict", "")
            if verdict == "SUPPORTED":
                supported_claims += 1
            elif verdict == "UNSUPPORTED_SPECIFICITY":
                unsupported_specificity_count += 1
            elif verdict == "CONTRADICTED":
                contradicted_count += 1
        citation_entailment = supported_claims / total_claims if total_claims > 0 else 0.0
    else:
        citation_entailment = None

    # --- Performance ---
    llm_ms = llm_latency_ms or 0
    total_ms = total_latency_ms or 0
    retrieval_ms = max(total_ms - llm_ms, 0) if total_ms else 0

    metrics = {
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
        "claims_with_valid_citations": claims_with_valid_citations,
        "citation_completeness": round(citation_completeness, 4),
        "critical_claims": critical_claims,
        "unsupported_critical_claims": unsupported_critical_claims,
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
        "has_uncertainties": has_uncertainties,
        "has_clarification": has_clarification,
        "direct_answer_chars": direct_answer_chars,
        "explanation_chars": explanation_chars,
        # Entailment
        "citation_entailment": round(citation_entailment, 4) if citation_entailment is not None else None,
        "supported_claims": supported_claims,
        "unsupported_specificity_count": unsupported_specificity_count,
        "contradicted_count": contradicted_count,
        "regeneration_attempts": regeneration_attempts,
        # Performance
        "llm_latency_ms": llm_ms,
        "total_latency_ms": total_ms,
        "retrieval_latency_ms": retrieval_ms,
    }

    # Budget diagnostics
    if budget_diagnostics:
        metrics.update(budget_diagnostics)

    return metrics


def compute_aggregate_metrics(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across multiple runs."""
    if not run_metrics:
        return {"total_runs": 0}

    numeric_keys = [
        "drug_lookup_success_rate", "nbn_evidence_provided", "stahl_evidence_provided",
        "kaplan_evidence_provided", "total_evidence_provided", "kaplan_top_score",
        "kaplan_mean_score", "conflicts_detected", "total_claims",
        "claims_with_valid_citations", "citation_completeness",
        "critical_claims", "unsupported_critical_claims",
        "unique_evidence_ids_cited", "evidence_utilization",
        "invalid_citation_count", "citation_validity", "nbn_cited", "stahl_cited",
        "kaplan_cited", "sources_used", "direct_answer_chars", "explanation_chars",
        "supported_claims", "unsupported_specificity_count", "contradicted_count",
        "llm_latency_ms", "total_latency_ms", "retrieval_latency_ms",
    ]

    agg: dict[str, Any] = {"total_runs": len(run_metrics)}

    for key in numeric_keys:
        values = [r.get(key, 0) for r in run_metrics if key in r and r.get(key) is not None]
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

    # Avg citation completeness (target 100%)
    completeness = [r.get("citation_completeness", 0.0) for r in run_metrics]
    agg["avg_citation_completeness"] = round(sum(completeness) / len(completeness), 4)

    # Avg entailment (if available)
    entailments = [r.get("citation_entailment") for r in run_metrics if r.get("citation_entailment") is not None]
    if entailments:
        agg["avg_citation_entailment"] = round(sum(entailments) / len(entailments), 4)

    return agg
