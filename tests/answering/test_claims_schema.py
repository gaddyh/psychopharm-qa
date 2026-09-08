"""Tests for the claims-first V2 answer schema, criticality, and prose assembly."""

from __future__ import annotations

import pytest

from psych_qa.answering.citations import (
    drop_unsupported_claims,
    resolve_citation,
    validate_citations,
    validate_every_claim_has_evidence,
)
from psych_qa.answering.criticality import (
    CRITICAL_CATEGORIES,
    assign_critical,
    is_critical_category,
)
from psych_qa.domain.models import AnswerClaim, assemble_prose


# ---------------------------------------------------------------------------
# Criticality tests
# ---------------------------------------------------------------------------

class TestCriticality:
    def test_critical_categories_defined(self):
        assert "dosing" in CRITICAL_CATEGORIES
        assert "pregnancy" in CRITICAL_CATEGORIES
        assert "contraindications" in CRITICAL_CATEGORIES
        assert "interaction" in CRITICAL_CATEGORIES

    def test_is_critical_category(self):
        assert is_critical_category("dosing") is True
        assert is_critical_category("mechanism") is False
        assert is_critical_category(None) is False

    def test_assign_critical_from_evidence(self):
        claims = [
            {"text": "Dose is 400-800 mg/day", "evidence_ids": ["stahl_claim_1"], "claim_role": "direct"},
            {"text": "Mechanism involves D2 blockade", "evidence_ids": ["stahl_claim_2"], "claim_role": "explanatory"},
        ]
        evidence_lookup = {
            "stahl_claim_1": {"category": "dosing", "source": "stahl"},
            "stahl_claim_2": {"category": "mechanism", "source": "stahl"},
        }
        result = assign_critical(claims, evidence_lookup)
        assert result[0]["critical"] is True  # dosing claim
        assert result[1]["critical"] is False  # mechanism claim

    def test_assign_critical_multiple_evidence(self):
        """A claim is critical if ANY of its cited evidence is critical."""
        claims = [
            {"text": "Some claim", "evidence_ids": ["stahl_claim_1", "stahl_claim_2"], "claim_role": "direct"},
        ]
        evidence_lookup = {
            "stahl_claim_1": {"category": "mechanism", "source": "stahl"},
            "stahl_claim_2": {"category": "pregnancy", "source": "stahl"},
        }
        result = assign_critical(claims, evidence_lookup)
        assert result[0]["critical"] is True  # pregnancy evidence makes it critical

    def test_assign_critical_no_evidence(self):
        claims = [
            {"text": "Some claim", "evidence_ids": [], "claim_role": "direct"},
        ]
        result = assign_critical(claims, {})
        assert result[0]["critical"] is False

    def test_llm_cannot_set_critical(self):
        """The LLM schema should not have a 'critical' field — it's app-assigned."""
        from psych_qa.llm.schemas import ANSWER_SCHEMA
        claim_item = ANSWER_SCHEMA["properties"]["claims"]["items"]
        props = claim_item["properties"]
        assert "critical" not in props, "LLM schema must not have 'critical' field"


# ---------------------------------------------------------------------------
# Citation validation tests
# ---------------------------------------------------------------------------

class TestCitationValidation:
    def test_validate_citations_all_valid(self):
        answer = {"claims": [
            {"text": "a", "evidence_ids": ["stahl_claim_1", "stahl_claim_2"]},
            {"text": "b", "evidence_ids": ["nbn_claim_3"]},
        ]}
        valid, invalid = validate_citations(answer, ["stahl_claim_1", "stahl_claim_2", "nbn_claim_3"])
        assert valid is True
        assert invalid == []

    def test_validate_citations_invalid_id(self):
        answer = {"claims": [
            {"text": "a", "evidence_ids": ["stahl_claim_1", "fake_id"]},
        ]}
        valid, invalid = validate_citations(answer, ["stahl_claim_1"])
        assert valid is False
        assert "fake_id" in invalid

    def test_validate_every_claim_has_evidence(self):
        answer = {"claims": [
            {"text": "a", "evidence_ids": ["stahl_claim_1"]},
            {"text": "b", "evidence_ids": []},
        ]}
        valid, missing = validate_every_claim_has_evidence(answer)
        assert valid is False
        assert 1 in missing

    def test_drop_unsupported_claims(self):
        answer = {"claims": [
            {"text": "supported", "evidence_ids": ["stahl_claim_1"]},
            {"text": "unsupported", "evidence_ids": ["fake_id"]},
            {"text": "mixed", "evidence_ids": ["stahl_claim_1", "fake_id"]},
        ]}
        result = drop_unsupported_claims(answer, ["stahl_claim_1"])
        assert len(result["claims"]) == 2
        texts = [c["text"] for c in result["claims"]]
        assert "supported" in texts
        assert "unsupported" not in texts
        assert "mixed" in texts
        # Mixed claim should have only the valid ID remaining
        mixed = next(c for c in result["claims"] if c["text"] == "mixed")
        assert mixed["evidence_ids"] == ["stahl_claim_1"]


# ---------------------------------------------------------------------------
# Prose assembly tests
# ---------------------------------------------------------------------------

class TestAssembleProse:
    def test_assemble_direct_and_explanation(self):
        claims = [
            AnswerClaim(text="Dose is 400-800 mg/day", evidence_ids=["stahl_claim_1"], claim_role="direct"),
            AnswerClaim(text="This is because of bioavailability", evidence_ids=["stahl_claim_2"], claim_role="explanatory"),
        ]
        direct, explanation = assemble_prose(claims)
        assert "Dose is 400-800 mg/day" in direct
        assert "[stahl_claim_1]" in direct
        assert "bioavailability" in explanation
        assert "[stahl_claim_2]" in explanation

    def test_assemble_critical_tag(self):
        claims = [
            AnswerClaim(text="Dose is 400-800 mg/day", evidence_ids=["stahl_claim_1"], claim_role="direct", critical=True),
        ]
        direct, _ = assemble_prose(claims)
        assert "⚠️" in direct

    def test_assemble_empty_claims(self):
        direct, explanation = assemble_prose([])
        assert direct == ""
        assert explanation == ""

    def test_assemble_caveat_role(self):
        claims = [
            AnswerClaim(text="Caution in pregnancy", evidence_ids=["stahl_claim_1"], claim_role="caveat"),
        ]
        _, explanation = assemble_prose(claims)
        assert "Caveat" in explanation
        assert "Caution in pregnancy" in explanation


# ---------------------------------------------------------------------------
# Resolve citation tests
# ---------------------------------------------------------------------------

class TestResolveCitation:
    def test_resolve_stahl_citation(self):
        evidence_lookup = {
            "stahl_claim_1": {
                "source": "stahl",
                "text": "Some claim text",
                "category": "mechanism",
                "locator": {"printed_book_page": 17, "physical_pdf_page": 33},
            },
        }
        result = resolve_citation("stahl_claim_1", evidence_lookup)
        assert result is not None
        assert result["source"] == "Stahl"
        assert result["page"] == 17
        assert result["category"] == "mechanism"

    def test_resolve_kaplan_citation(self):
        evidence_lookup = {
            "kaplan_chunk_500": {
                "source": "kaplan",
                "text": "Kaplan passage text",
                "locator": {"physical_pdf_page": 10050},
            },
        }
        result = resolve_citation("kaplan_chunk_500", evidence_lookup)
        assert result is not None
        assert result["source"] == "Kaplan Ch.33"
        assert result["page"] == 10050

    def test_resolve_unknown_citation(self):
        result = resolve_citation("nonexistent", {})
        assert result is None
