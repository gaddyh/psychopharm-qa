"""Retrieval metrics: Evidence Recall@K.

Primary: macro critical_evidence_recall_at_10
Pass gate: all critical evidence found in top-10

Secondary:
  - macro all-evidence Recall@5/10/20
  - macro critical Recall@5/10/20
  - micro critical recall (case-evidence pairs)
  - micro all-evidence recall (case-evidence pairs)

Micro counts case-evidence pairs, not unique evidence IDs, because the
same evidence can be required by multiple cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gold_loader import GoldCase, GoldSet


@dataclass
class CaseResult:
    """Retrieval evaluation result for a single case."""

    case_id: str
    question: str
    route: str

    # Per-K matched evidence
    matched_critical_at_k: dict[int, set[str]] = field(default_factory=dict)
    matched_all_at_k: dict[int, set[str]] = field(default_factory=dict)

    # Full detail
    critical_evidence: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    missed_critical_at_10: list[str] = field(default_factory=list)
    missed_all_at_10: list[str] = field(default_factory=list)

    # Retrieved chunks
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def retrieval_pass(self) -> bool:
        """Pass = all critical evidence found in top-10."""
        return len(self.missed_critical_at_10) == 0

    def critical_recall_at(self, k: int) -> float:
        matched = self.matched_critical_at_k.get(k, set())
        total = set(self.critical_evidence)
        if not total:
            return 1.0
        return len(matched & total) / len(total)

    def all_recall_at(self, k: int) -> float:
        matched = self.matched_all_at_k.get(k, set())
        total = set(self.critical_evidence) | set(self.supporting_evidence)
        if not total:
            return 1.0
        return len(matched & total) / len(total)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all cases."""

    n_cases: int = 0
    retrieval_passes: int = 0

    # Macro averages (mean of per-case recall)
    macro_critical_recall_at_5: float = 0.0
    macro_critical_recall_at_10: float = 0.0
    macro_critical_recall_at_20: float = 0.0
    macro_all_recall_at_5: float = 0.0
    macro_all_recall_at_10: float = 0.0
    macro_all_recall_at_20: float = 0.0

    # Micro averages (total matched / total required across all case-evidence pairs)
    micro_critical_recall_at_10: float = 0.0
    micro_all_recall_at_10: float = 0.0

    # Raw counts
    total_critical_pairs_at_10: int = 0
    matched_critical_pairs_at_10: int = 0
    total_all_pairs_at_10: int = 0
    matched_all_pairs_at_10: int = 0

    def format_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Cases:                      {self.n_cases}",
            f"Retrieval passes:           {self.retrieval_passes}/{self.n_cases}",
            "",
            "Primary:",
            f"  Macro critical Recall@10: {self.macro_critical_recall_at_10:.1%}",
            "",
            "Secondary (macro):",
            f"  Critical Recall@5:         {self.macro_critical_recall_at_5:.1%}",
            f"  Critical Recall@10:        {self.macro_critical_recall_at_10:.1%}",
            f"  Critical Recall@20:        {self.macro_critical_recall_at_20:.1%}",
            f"  All-evidence Recall@5:     {self.macro_all_recall_at_5:.1%}",
            f"  All-evidence Recall@10:    {self.macro_all_recall_at_10:.1%}",
            f"  All-evidence Recall@20:    {self.macro_all_recall_at_20:.1%}",
            "",
            "Secondary (micro):",
            f"  Critical Recall@10:        {self.macro_critical_recall_at_10:.1%} "
            f"({self.matched_critical_pairs_at_10}/{self.total_critical_pairs_at_10})",
            f"  All-evidence Recall@10:    {self.micro_all_recall_at_10:.1%} "
            f"({self.matched_all_pairs_at_10}/{self.total_all_pairs_at_10})",
        ]
        return "\n".join(lines)


def compute_case_result(
    case: GoldCase,
    retrieved_chunks: list[dict[str, Any]],
    gold_set: GoldSet,
    k_values: list[int] = (5, 10, 20),
) -> CaseResult:
    """Compute retrieval metrics for a single case.

    Args:
        case: Gold case.
        retrieved_chunks: Retrieved chunks ordered by rank.
        gold_set: Loaded gold set.
        k_values: K values to compute recall at.

    Returns:
        CaseResult with matched/missed evidence at each K.
    """
    from .evidence_matcher import find_matched_evidence_at_k

    all_evidence = list(case.critical_evidence) + list(case.supporting_evidence)

    result = CaseResult(
        case_id=case.case_id,
        question=case.question,
        route=case.route,
        critical_evidence=list(case.critical_evidence),
        supporting_evidence=list(case.supporting_evidence),
        retrieved_chunks=retrieved_chunks,
    )

    for k in k_values:
        matched_critical = find_matched_evidence_at_k(
            retrieved_chunks, case.critical_evidence, gold_set, k
        )
        matched_all = find_matched_evidence_at_k(
            retrieved_chunks, all_evidence, gold_set, k
        )
        result.matched_critical_at_k[k] = matched_critical
        result.matched_all_at_k[k] = matched_all

    # Missed at 10
    matched_at_10 = result.matched_critical_at_k.get(10, set())
    result.missed_critical_at_10 = [
        eid for eid in case.critical_evidence if eid not in matched_at_10
    ]
    matched_all_at_10 = result.matched_all_at_k.get(10, set())
    result.missed_all_at_10 = [
        eid for eid in all_evidence if eid not in matched_all_at_10
    ]

    return result


def compute_aggregate_metrics(case_results: list[CaseResult]) -> AggregateMetrics:
    """Compute aggregate metrics across all cases.

    Args:
        case_results: List of per-case results.

    Returns:
        AggregateMetrics with macro and micro averages.
    """
    n = len(case_results)
    if n == 0:
        return AggregateMetrics()

    agg = AggregateMetrics(n_cases=n)
    agg.retrieval_passes = sum(1 for cr in case_results if cr.retrieval_pass)

    # Macro averages
    for k in (5, 10, 20):
        crit_recalls = [cr.critical_recall_at(k) for cr in case_results]
        all_recalls = [cr.all_recall_at(k) for cr in case_results]

        if k == 5:
            agg.macro_critical_recall_at_5 = sum(crit_recalls) / n
            agg.macro_all_recall_at_5 = sum(all_recalls) / n
        elif k == 10:
            agg.macro_critical_recall_at_10 = sum(crit_recalls) / n
            agg.macro_all_recall_at_10 = sum(all_recalls) / n
        elif k == 20:
            agg.macro_critical_recall_at_20 = sum(crit_recalls) / n
            agg.macro_all_recall_at_20 = sum(all_recalls) / n

    # Micro averages at K=10 (case-evidence pairs)
    total_critical = 0
    matched_critical = 0
    total_all = 0
    matched_all = 0

    for cr in case_results:
        crit_set = set(cr.critical_evidence)
        all_set = set(cr.critical_evidence) | set(cr.supporting_evidence)
        matched_at_10 = cr.matched_critical_at_k.get(10, set())
        matched_all_at_10 = cr.matched_all_at_k.get(10, set())

        total_critical += len(crit_set)
        matched_critical += len(matched_at_10 & crit_set)
        total_all += len(all_set)
        matched_all += len(matched_all_at_10 & all_set)

    agg.total_critical_pairs_at_10 = total_critical
    agg.matched_critical_pairs_at_10 = matched_critical
    agg.total_all_pairs_at_10 = total_all
    agg.matched_all_pairs_at_10 = matched_all

    agg.micro_critical_recall_at_10 = matched_critical / total_critical if total_critical else 1.0
    agg.micro_all_recall_at_10 = matched_all / total_all if total_all else 1.0

    return agg
