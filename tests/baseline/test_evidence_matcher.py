"""Unit tests for the evidence matcher.

Tests page overlap, anchor text matching, normalization,
and cross-chunk anchor support. No DB required.
"""

from __future__ import annotations

from psych_qa.evaluation.evidence_matcher import (
    anchor_in_text,
    find_matched_evidence_at_k,
    match_chunk_to_evidence,
    match_chunk_to_evidence_cross,
    match_retrieved_to_evidence,
    normalize_text,
    pages_overlap,
)
from psych_qa.evaluation.gold_loader import EvidenceItem, GoldSet, GoldCase, GoldSetMeta, CorpusInfo, RetrievalPolicy, EvaluationPolicy


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("hello    world\n\n\tworld") == "hello world world"

    def test_fix_hyphenated_line_break(self):
        assert normalize_text("seroto-\nnin") == "serotonin"

    def test_fix_hyphenated_line_break_word(self):
        assert normalize_text("dopamine-\nreceptor") == "dopaminereceptor"

    def test_remove_non_printable(self):
        # Non-printable chars are removed (not replaced with space)
        assert normalize_text("hello\x00world") == "helloworld"


class TestPagesOverlap:
    def test_overlap_single_page(self):
        assert pages_overlap(5, 5, [5]) is True

    def test_no_overlap(self):
        assert pages_overlap(1, 3, [10, 11]) is False

    def test_overlap_range(self):
        assert pages_overlap(3, 7, [5, 6]) is True

    def test_overlap_boundary(self):
        assert pages_overlap(5, 10, [10]) is True

    def test_no_overlap_adjacent(self):
        assert pages_overlap(1, 5, [6, 7]) is False


class TestAnchorInText:
    def test_exact_match(self):
        assert anchor_in_text("risperidone is helpful", "risperidone is helpful for OCD") is True

    def test_case_insensitive(self):
        assert anchor_in_text("RISPERIDONE", "Risperidone is a drug") is True

    def test_no_match(self):
        assert anchor_in_text("aripiprazole", "risperidone is a drug") is False

    def test_hyphenated_break_in_source(self):
        """Source text has a hyphenated line break that should be fixed."""
        assert anchor_in_text("serotonin", "seroto-\nnin is important") is True

    def test_whitespace_normalization(self):
        assert anchor_in_text("hello world", "hello    world") is True


class TestMatchChunkToEvidence:
    def _make_evidence(self, pages: list[int], anchors: list[str]) -> EvidenceItem:
        return EvidenceItem(
            evidence_id="E1",
            pdf_pages=pages,
            anchor_text=anchors,
        )

    def _make_chunk(self, page_start: int, page_end: int, text: str) -> dict:
        return {
            "id": 1,
            "page_start": page_start,
            "page_end": page_end,
            "text": text,
        }

    def test_page_and_anchor_match(self):
        ev = self._make_evidence([5], ["risperidone"])
        chunk = self._make_chunk(5, 5, "Risperidone is a D2 antagonist")
        assert match_chunk_to_evidence(chunk, ev) is True

    def test_page_match_but_no_anchor(self):
        ev = self._make_evidence([5], ["aripiprazole"])
        chunk = self._make_chunk(5, 5, "Risperidone is a D2 antagonist")
        assert match_chunk_to_evidence(chunk, ev) is False

    def test_anchor_match_but_wrong_page(self):
        ev = self._make_evidence([5], ["risperidone"])
        chunk = self._make_chunk(10, 10, "Risperidone is a D2 antagonist")
        assert match_chunk_to_evidence(chunk, ev) is False

    def test_multi_page_chunk_matches(self):
        ev = self._make_evidence([5], ["risperidone"])
        chunk = self._make_chunk(3, 7, "Risperidone is a D2 antagonist")
        # page 5 is in range(3, 8), so this matches
        assert match_chunk_to_evidence(chunk, ev) is True

    def test_multi_page_evidence_matches(self):
        ev = self._make_evidence([5, 6, 7], ["risperidone"])
        chunk = self._make_chunk(6, 6, "Risperidone is a D2 antagonist")
        assert match_chunk_to_evidence(chunk, ev) is True


class TestCrossChunkAnchor:
    def _make_evidence(self, pages: list[int], anchors: list[str]) -> EvidenceItem:
        return EvidenceItem(evidence_id="E1", pdf_pages=pages, anchor_text=anchors)

    def test_single_chunk_sufficient(self):
        ev = self._make_evidence([5], ["risperidone"])
        chunk = {"id": 1, "page_start": 5, "page_end": 5, "text": "Risperidone is helpful"}
        assert match_chunk_to_evidence_cross(chunk, None, ev) is True

    def test_cross_chunk_anchor(self):
        """Anchor split across two adjacent chunks."""
        ev = self._make_evidence([5], ["risperidone is helpful for OCD"])
        chunk1 = {"id": 1, "page_start": 5, "page_end": 5, "text": "risperidone is"}
        chunk2 = {"id": 2, "page_start": 5, "page_end": 5, "text": "helpful for OCD treatment"}
        # Neither alone matches, but together they do
        assert match_chunk_to_evidence_cross(chunk1, chunk2, ev) is True

    def test_cross_chunk_wrong_page(self):
        """Cross-chunk with wrong page should not match."""
        ev = self._make_evidence([5], ["risperidone is helpful"])
        chunk1 = {"id": 1, "page_start": 10, "page_end": 10, "text": "risperidone is"}
        chunk2 = {"id": 2, "page_start": 10, "page_end": 10, "text": "helpful"}
        assert match_chunk_to_evidence_cross(chunk1, chunk2, ev) is False

    def test_no_next_chunk(self):
        ev = self._make_evidence([5], ["risperidone is helpful for OCD"])
        chunk = {"id": 1, "page_start": 5, "page_end": 5, "text": "risperidone is"}
        assert match_chunk_to_evidence_cross(chunk, None, ev) is False


class TestMatchRetrievedToEvidence:
    def _make_gold_set(self) -> GoldSet:
        meta = GoldSetMeta(
            id="test",
            title="test",
            corpus=CorpusInfo(file_name="test.pdf", chapter=33, page_numbering="0-based"),
            evaluation_policy=EvaluationPolicy(
                retrieval=RetrievalPolicy(primary_metric="test"),
            ),
        )
        evidence = [
            EvidenceItem(evidence_id="E1", pdf_pages=[5], anchor_text=["risperidone"]),
            EvidenceItem(evidence_id="E2", pdf_pages=[10], anchor_text=["aripiprazole"]),
            EvidenceItem(evidence_id="E3", pdf_pages=[15], anchor_text=["quetiapine"]),
        ]
        cases = [
            GoldCase(case_id="C1", route="test", question="q?", critical_evidence=["E1", "E2"]),
        ]
        return GoldSet(meta=meta, evidence_catalog=evidence, cases=cases)

    def test_match_multiple_evidence(self):
        gs = self._make_gold_set()
        chunks = [
            {"id": 1, "page_start": 5, "page_end": 5, "text": "Risperidone is a D2 antagonist"},
            {"id": 2, "page_start": 10, "page_end": 10, "text": "Aripiprazole is a partial agonist"},
            {"id": 3, "page_start": 20, "page_end": 20, "text": "Some unrelated text"},
        ]
        matches = match_retrieved_to_evidence(chunks, ["E1", "E2", "E3"], gs)
        assert matches["E1"] == [1]
        assert matches["E2"] == [2]
        assert matches["E3"] == []

    def test_find_matched_at_k(self):
        gs = self._make_gold_set()
        chunks = [
            {"id": 1, "page_start": 5, "page_end": 5, "text": "Risperidone is a D2 antagonist"},
            {"id": 2, "page_start": 20, "page_end": 20, "text": "unrelated"},
            {"id": 3, "page_start": 10, "page_end": 10, "text": "Aripiprazole is a partial agonist"},
        ]
        # At k=1, only E1 is matched
        matched_1 = find_matched_evidence_at_k(chunks, ["E1", "E2"], gs, 1)
        assert matched_1 == {"E1"}

        # At k=3, both E1 and E2 are matched
        matched_3 = find_matched_evidence_at_k(chunks, ["E1", "E2"], gs, 3)
        assert matched_3 == {"E1", "E2"}
