"""Tests for question-specific sufficiency rules."""

from __future__ import annotations

import pytest

from psych_qa.answering.sufficiency import check_sufficiency


def _make_evidence_package(nbn_cats=None, stahl_cats=None, kaplan_count=0, drugs=None):
    """Build a synthetic evidence package."""
    nbn_cats = nbn_cats or []
    stahl_cats = stahl_cats or []
    drugs = drugs or {"amisulpride": True}

    drugs_dict = {}
    for drug_name in drugs:
        nbn_evidence = [{"category": c, "text": f"nbn {c}"} for c in nbn_cats]
        stahl_evidence = [{"category": c, "text": f"stahl {c}"} for c in stahl_cats]
        drugs_dict[drug_name] = {
            "nbn_evidence": nbn_evidence,
            "stahl_evidence": stahl_evidence,
        }

    return {
        "drugs": drugs_dict,
        "kaplan_passages": [{"text": f"kaplan {i}"} for i in range(kaplan_count)],
    }


class TestBasicSufficiency:
    def test_no_drugs(self):
        ep = {"drugs": {}, "kaplan_passages": []}
        ok, reason = check_sufficiency(ep, {"question_type": "general"})
        assert ok is False
        assert "No recognized drugs" in reason

    def test_no_evidence(self):
        ep = _make_evidence_package(nbn_cats=[], stahl_cats=[])
        ok, reason = check_sufficiency(ep, {"question_type": "general"})
        assert ok is False
        assert "No evidence" in reason

    def test_no_understanding_basic_ok(self):
        ep = _make_evidence_package(stahl_cats=["mechanism"])
        ok, reason = check_sufficiency(ep, None)
        assert ok is True
        assert reason is None


class TestDosingSufficiency:
    def test_dosing_with_dosage_range(self):
        ep = _make_evidence_package(stahl_cats=["dosage_range", "dosing_instructions"])
        ok, _ = check_sufficiency(ep, {"question_type": "dosing"})
        assert ok is True

    def test_dosing_with_dosing_only(self):
        ep = _make_evidence_package(stahl_cats=["dosing"])
        ok, _ = check_sufficiency(ep, {"question_type": "dosing"})
        assert ok is True

    def test_dosing_without_dosing_evidence(self):
        ep = _make_evidence_package(stahl_cats=["mechanism", "side_effects"])
        ok, reason = check_sufficiency(ep, {"question_type": "dosing"})
        assert ok is False
        assert "dosing" in reason.lower()


class TestInteractionsSufficiency:
    def test_interactions_with_evidence(self):
        ep = _make_evidence_package(stahl_cats=["interactions"])
        ok, _ = check_sufficiency(ep, {"question_type": "interactions"})
        assert ok is True

    def test_interactions_without_evidence(self):
        ep = _make_evidence_package(stahl_cats=["mechanism"])
        ok, reason = check_sufficiency(ep, {"question_type": "interactions"})
        assert ok is False
        assert "interactions" in reason.lower()


class TestMechanismSufficiency:
    def test_mechanism_with_mechanism_and_target(self):
        ep = _make_evidence_package(stahl_cats=["mechanism"], nbn_cats=["target"])
        ok, _ = check_sufficiency(ep, {"question_type": "mechanism"})
        assert ok is True

    def test_mechanism_with_mode_of_action_only(self):
        ep = _make_evidence_package(nbn_cats=["mode_of_action"])
        ok, _ = check_sufficiency(ep, {"question_type": "mechanism"})
        assert ok is True

    def test_mechanism_without_evidence(self):
        ep = _make_evidence_package(stahl_cats=["side_effects"])
        ok, reason = check_sufficiency(ep, {"question_type": "mechanism"})
        assert ok is False
        assert "mechanism" in reason.lower()


class TestPatientSpecific:
    def test_patient_specific_abstains(self):
        ep = _make_evidence_package(stahl_cats=["dosing"])
        ok, reason = check_sufficiency(ep, {
            "question_type": "dosing",
            "is_patient_specific": True,
        })
        assert ok is False
        assert "patient" in reason.lower()


class TestComparison:
    def test_comparison_needs_two_drugs(self):
        ep = _make_evidence_package(stahl_cats=["mechanism"], drugs={"amisulpride": True})
        ok, reason = check_sufficiency(ep, {
            "question_type": "comparison",
            "drug_names": ["amisulpride"],
        })
        assert ok is False
        assert "two drugs" in reason

    def test_comparison_both_drugs_have_evidence(self):
        ep = _make_evidence_package(
            stahl_cats=["mechanism"],
            drugs={"amisulpride": True, "haloperidol": True},
        )
        ok, _ = check_sufficiency(ep, {
            "question_type": "comparison",
            "drug_names": ["amisulpride", "haloperidol"],
        })
        assert ok is True

    def test_comparison_one_drug_missing_evidence(self):
        ep = {
            "drugs": {
                "amisulpride": {
                    "nbn_evidence": [{"category": "target"}],
                    "stahl_evidence": [{"category": "mechanism"}],
                },
                "unknown_drug": {
                    "nbn_evidence": [],
                    "stahl_evidence": [],
                },
            },
            "kaplan_passages": [],
        }
        ok, reason = check_sufficiency(ep, {
            "question_type": "comparison",
            "drug_names": ["amisulpride", "unknown_drug"],
        })
        assert ok is False
        assert "evidence" in reason.lower()


class TestSpecialPopulations:
    def test_special_populations_with_pregnancy(self):
        ep = _make_evidence_package(stahl_cats=["pregnancy"])
        ok, _ = check_sufficiency(ep, {"question_type": "special_populations"})
        assert ok is True

    def test_special_populations_with_renal(self):
        ep = _make_evidence_package(stahl_cats=["renal_impairment"])
        ok, _ = check_sufficiency(ep, {"question_type": "special_populations"})
        assert ok is True

    def test_special_populations_without_evidence(self):
        ep = _make_evidence_package(stahl_cats=["mechanism"])
        ok, reason = check_sufficiency(ep, {"question_type": "special_populations"})
        assert ok is False
