"""Unit tests for retrieval metrics.

Tests recall computation, pass/fail logic, macro and micro aggregation.
No DB required.
"""

from __future__ import annotations

from psych_qa.evaluation.gold_loader import (
    CorpusInfo,
    EvaluationPolicy,
    EvidenceItem,
    GoldCase,
    GoldSet,
    GoldSetMeta,
    RetrievalPolicy,
)
from psych_qa.evaluation.metrics import (
    CaseResult,
    compute_aggregate_metrics,
    compute_case_result,
)


def _make_gold_set(cases: list[GoldCase], evidence: list[EvidenceItem]) -> GoldSet:
    meta = GoldSetMeta(
        id="test",
        title="test",
        corpus=CorpusInfo(file_name="test.pdf", chapter=33, page_numbering="0-based"),
        evaluation_policy=EvaluationPolicy(
            retrieval=RetrievalPolicy(primary_metric="test"),
        ),
    )
    return GoldSet(meta=meta, evidence_catalog=evidence, cases=cases)


def _make_evidence(eid: str, pages: list[int], anchor: str) -> EvidenceItem:
    return EvidenceItem(evidence_id=eid, pdf_pages=pages, anchor_text=[anchor])


def _make_case(cid: str, critical: list[str], supporting: list[str] | None = None) -> GoldCase:
    return GoldCase(
        case_id=cid,
        route="test",
        question="q?",
        critical_evidence=critical,
        supporting_evidence=supporting or [],
    )


def _make_chunk(cid: int, page: int, text: str) -> dict:
    return {"id": cid, "page_start": page, "page_end": page, "text": text, "rank": 0, "score": 0.9}


class TestCaseResult:
    def test_retrieval_pass_all_critical_found(self):
        cr = CaseResult(
            case_id="C1",
            question="q?",
            route="test",
            critical_evidence=["E1", "E2"],
            missed_critical_at_10=[],
        )
        assert cr.retrieval_pass is True

    def test_retrieval_fail_missing_critical(self):
        cr = CaseResult(
            case_id="C1",
            question="q?",
            route="test",
            critical_evidence=["E1", "E2"],
            missed_critical_at_10=["E2"],
        )
        assert cr.retrieval_pass is False

    def test_critical_recall_at_k(self):
        cr = CaseResult(
            case_id="C1",
            question="q?",
            route="test",
            critical_evidence=["E1", "E2", "E3", "E4"],
            matched_critical_at_k={
                5: {"E1", "E2"},
                10: {"E1", "E2", "E3"},
                20: {"E1", "E2", "E3", "E4"},
            },
        )
        assert cr.critical_recall_at(5) == 0.5
        assert cr.critical_recall_at(10) == 0.75
        assert cr.critical_recall_at(20) == 1.0

    def test_all_recall_at_k(self):
        cr = CaseResult(
            case_id="C1",
            question="q?",
            route="test",
            critical_evidence=["E1"],
            supporting_evidence=["E2", "E3"],
            matched_all_at_k={
                5: {"E1"},
                10: {"E1", "E2"},
                20: {"E1", "E2", "E3"},
            },
        )
        assert cr.all_recall_at(5) == 1 / 3
        assert cr.all_recall_at(10) == 2 / 3
        assert cr.all_recall_at(20) == 1.0


class TestComputeCaseResult:
    def test_full_match(self):
        evidence = [
            _make_evidence("E1", [5], "risperidone"),
            _make_evidence("E2", [10], "aripiprazole"),
        ]
        case = _make_case("C1", ["E1", "E2"])
        gs = _make_gold_set([case], evidence)

        chunks = [
            _make_chunk(1, 5, "Risperidone is a D2 antagonist"),
            _make_chunk(2, 10, "Aripiprazole is a partial agonist"),
        ]

        cr = compute_case_result(case, chunks, gs)
        assert cr.retrieval_pass is True
        assert cr.critical_recall_at(10) == 1.0
        assert cr.missed_critical_at_10 == []

    def test_partial_match(self):
        evidence = [
            _make_evidence("E1", [5], "risperidone"),
            _make_evidence("E2", [10], "aripiprazole"),
        ]
        case = _make_case("C1", ["E1", "E2"])
        gs = _make_gold_set([case], evidence)

        chunks = [
            _make_chunk(1, 5, "Risperidone is a D2 antagonist"),
            _make_chunk(2, 20, "Some unrelated text"),
        ]

        cr = compute_case_result(case, chunks, gs)
        assert cr.retrieval_pass is False
        assert cr.critical_recall_at(10) == 0.5
        assert cr.missed_critical_at_10 == ["E2"]

    def test_wrong_page_no_match(self):
        """Page overlap required — anchor on wrong page should not match."""
        evidence = [_make_evidence("E1", [5], "risperidone")]
        case = _make_case("C1", ["E1"])
        gs = _make_gold_set([case], evidence)

        chunks = [_make_chunk(1, 99, "Risperidone is a D2 antagonist")]

        cr = compute_case_result(case, chunks, gs)
        assert cr.retrieval_pass is False
        assert cr.critical_recall_at(10) == 0.0


class TestAggregateMetrics:
    def test_macro_critical_recall(self):
        results = [
            CaseResult(
                case_id="C1", question="q?", route="test",
                critical_evidence=["E1", "E2"],
                matched_critical_at_k={10: {"E1", "E2"}},
                missed_critical_at_10=[],
            ),
            CaseResult(
                case_id="C2", question="q?", route="test",
                critical_evidence=["E1", "E2", "E3"],
                matched_critical_at_k={10: {"E1"}},
                missed_critical_at_10=["E2", "E3"],
            ),
        ]
        agg = compute_aggregate_metrics(results)
        # C1: 2/2 = 1.0, C2: 1/3 = 0.333
        assert agg.macro_critical_recall_at_10 == pytest_approx((1.0 + 1 / 3) / 2)
        assert agg.retrieval_passes == 1
        assert agg.n_cases == 2

    def test_micro_critical_recall(self):
        results = [
            CaseResult(
                case_id="C1", question="q?", route="test",
                critical_evidence=["E1", "E2"],
                matched_critical_at_k={10: {"E1", "E2"}},
                missed_critical_at_10=[],
            ),
            CaseResult(
                case_id="C2", question="q?", route="test",
                critical_evidence=["E1", "E2", "E3"],
                matched_critical_at_k={10: {"E1"}},
                missed_critical_at_10=["E2", "E3"],
            ),
        ]
        agg = compute_aggregate_metrics(results)
        # Total pairs: 2 + 3 = 5, matched: 2 + 1 = 3
        assert agg.total_critical_pairs_at_10 == 5
        assert agg.matched_critical_pairs_at_10 == 3
        assert agg.micro_critical_recall_at_10 == 3 / 5

    def test_micro_counts_pairs_not_unique_ids(self):
        """Same evidence in two cases should count as two pairs."""
        results = [
            CaseResult(
                case_id="C1", question="q?", route="test",
                critical_evidence=["E1"],
                matched_critical_at_k={10: {"E1"}},
                missed_critical_at_10=[],
            ),
            CaseResult(
                case_id="C2", question="q?", route="test",
                critical_evidence=["E1"],
                matched_critical_at_k={10: {"E1"}},
                missed_critical_at_10=[],
            ),
        ]
        agg = compute_aggregate_metrics(results)
        assert agg.total_critical_pairs_at_10 == 2
        assert agg.matched_critical_pairs_at_10 == 2

    def test_empty_results(self):
        agg = compute_aggregate_metrics([])
        assert agg.n_cases == 0


def pytest_approx(expected):
    """Simple approximate comparison helper."""
    class Approx:
        def __eq__(self, other):
            return abs(expected - other) < 1e-6
    return Approx()
