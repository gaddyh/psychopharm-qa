"""Unit tests for the gold set YAML loader.

Tests validation: duplicate evidence IDs, missing references, empty
critical, negative pages, evidence in both critical+supporting,
duplicate case IDs, missing anchor_text.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from psych_qa.evaluation.gold_loader import GoldSet, EvidenceItem, GoldCase, load_gold_set


def _write_yaml(tmp_path: Path, gold_set: dict, evidence_catalog: list, cases: list) -> Path:
    """Write a minimal gold set YAML."""
    data = {
        "gold_set": gold_set,
        "evidence_catalog": evidence_catalog,
        "cases": cases,
    }
    path = tmp_path / "test_gold.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _base_gold_set() -> dict:
    return {
        "id": "test",
        "title": "Test",
        "corpus": {
            "file_name": "test.pdf",
            "chapter": 33,
            "page_numbering": "0-based",
        },
        "evaluation_policy": {
            "retrieval": {"primary_metric": "critical_evidence_recall_at_10"},
        },
    }


def _base_evidence(eid: str = "E1", pages: list[int] | None = None) -> dict:
    return {
        "evidence_id": eid,
        "pdf_pages": pages or [1],
        "anchor_text": ["test anchor phrase"],
        "excerpt": "test excerpt",
    }


def _base_case(cid: str = "C1", critical: list[str] | None = None) -> dict:
    return {
        "case_id": cid,
        "route": "test",
        "question": "test question?",
        "critical_evidence": critical or ["E1"],
        "supporting_evidence": [],
        "required_claims": [],
        "forbidden_claims": [],
    }


class TestEvidenceItemValidation:
    def test_negative_page_rejected(self):
        with pytest.raises(Exception, match="negative"):
            EvidenceItem(evidence_id="E1", pdf_pages=[-1], anchor_text=["x"])

    def test_empty_pages_rejected(self):
        with pytest.raises(Exception, match="empty"):
            EvidenceItem(evidence_id="E1", pdf_pages=[], anchor_text=["x"])

    def test_empty_anchor_text_rejected(self):
        with pytest.raises(Exception, match="anchor_text"):
            EvidenceItem(evidence_id="E1", pdf_pages=[1], anchor_text=[])


class TestGoldCaseValidation:
    def test_empty_critical_rejected(self):
        with pytest.raises(Exception, match="critical_evidence"):
            GoldCase(case_id="C1", route="test", question="q?", critical_evidence=[])

    def test_evidence_overlap_rejected(self):
        with pytest.raises(Exception, match="both critical and supporting"):
            GoldCase(
                case_id="C1",
                route="test",
                question="q?",
                critical_evidence=["E1"],
                supporting_evidence=["E1"],
            )


class TestGoldSetCrossReferenceValidation:
    def test_duplicate_evidence_id_rejected(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            _base_gold_set(),
            [_base_evidence("E1"), _base_evidence("E1")],
            [_base_case("C1", ["E1"])],
        )
        with pytest.raises(Exception, match="Duplicate evidence_id"):
            load_gold_set(path)

    def test_duplicate_case_id_rejected(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            _base_gold_set(),
            [_base_evidence("E1")],
            [_base_case("C1", ["E1"]), _base_case("C1", ["E1"])],
        )
        with pytest.raises(Exception, match="Duplicate case_id"):
            load_gold_set(path)

    def test_missing_critical_reference_rejected(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            _base_gold_set(),
            [_base_evidence("E1")],
            [_base_case("C1", ["E_MISSING"])],
        )
        with pytest.raises(Exception, match="unknown ID"):
            load_gold_set(path)

    def test_missing_supporting_reference_rejected(self, tmp_path):
        case = _base_case("C1", ["E1"])
        case["supporting_evidence"] = ["E_MISSING"]
        path = _write_yaml(
            tmp_path,
            _base_gold_set(),
            [_base_evidence("E1")],
            [case],
        )
        with pytest.raises(Exception, match="unknown ID"):
            load_gold_set(path)

    def test_valid_gold_set_loads(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            _base_gold_set(),
            [_base_evidence("E1", [1, 2]), _base_evidence("E2", [3])],
            [_base_case("C1", ["E1", "E2"])],
        )
        gs = load_gold_set(path)
        assert len(gs.cases) == 1
        assert len(gs.evidence_catalog) == 2
        assert gs.evidence_by_id["E1"].pdf_pages == [1, 2]

    def test_all_evidence_for_case(self, tmp_path):
        case = _base_case("C1", ["E1"])
        case["supporting_evidence"] = ["E2"]
        path = _write_yaml(
            tmp_path,
            _base_gold_set(),
            [_base_evidence("E1"), _base_evidence("E2")],
            [case],
        )
        gs = load_gold_set(path)
        all_ev = gs.all_evidence_for_case(gs.cases[0])
        assert set(all_ev) == {"E1", "E2"}


class TestRealGoldSet:
    """Test that the actual gold set file loads and validates."""

    def test_real_gold_set_loads(self):
        from psych_qa.config import get_settings
        settings = get_settings()
        path = settings.evals_dir / "kaplan_ocd_gold_set_v0.1.yaml"
        if not path.exists():
            pytest.skip("Gold set file not found")
        gs = load_gold_set(path)
        assert gs.meta.id == "kaplan_ch33_ocd_gold_v0.1"
        assert len(gs.cases) == 7
        assert len(gs.evidence_catalog) == 23
        # All evidence items have anchor_text
        for item in gs.evidence_catalog:
            assert len(item.anchor_text) > 0, f"{item.evidence_id} has no anchor_text"
