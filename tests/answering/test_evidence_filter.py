"""Tests for question-type evidence filtering."""

from __future__ import annotations

from psych_qa.answering.evidence_filter import (
    ALWAYS_ON_CATEGORIES,
    filter_claims_by_question_type,
    get_categories_for_question_type,
)


class TestGetCategories:
    def test_mechanism_categories(self):
        cats = get_categories_for_question_type("mechanism")
        assert "mechanism" in cats
        assert "mode_of_action" in cats
        assert "target" in cats
        # Always-on
        assert "class" in cats

    def test_dosing_categories(self):
        cats = get_categories_for_question_type("dosing")
        assert "dosage_range" in cats
        assert "dosing_instructions" in cats
        assert "renal_impairment" in cats
        # Always-on
        assert "mechanism" in cats

    def test_interactions_categories(self):
        cats = get_categories_for_question_type("interactions")
        assert "interactions" in cats
        assert "contraindications" in cats

    def test_general_returns_empty(self):
        """General/fallback returns empty set = all categories."""
        cats = get_categories_for_question_type("general")
        assert cats == set()

    def test_unknown_returns_empty(self):
        cats = get_categories_for_question_type("nonexistent")
        assert cats == set()

    def test_special_populations(self):
        cats = get_categories_for_question_type("special_populations")
        assert "pregnancy" in cats
        assert "breast_feeding" in cats
        assert "elderly" in cats
        assert "pediatric" in cats


class TestFilterClaims:
    def _make_claims(self, categories):
        return [{"category": c, "text": f"claim for {c}"} for c in categories]

    def test_filter_dosing(self):
        claims = self._make_claims(["dosage_range", "mechanism", "side_effects", "dosing_instructions"])
        result = filter_claims_by_question_type(claims, "dosing")
        cats = {c["category"] for c in result}
        assert "dosage_range" in cats
        assert "dosing_instructions" in cats
        assert "mechanism" in cats  # always-on
        assert "side_effects" not in cats  # not relevant to dosing

    def test_filter_general_returns_all(self):
        claims = self._make_claims(["dosage_range", "mechanism", "side_effects"])
        result = filter_claims_by_question_type(claims, "general")
        assert len(result) == 3  # all returned

    def test_filter_unknown_returns_all(self):
        claims = self._make_claims(["dosage_range", "mechanism"])
        result = filter_claims_by_question_type(claims, "nonexistent")
        assert len(result) == 2

    def test_filter_mechanism(self):
        claims = self._make_claims(["mechanism", "side_effects", "dosage_range", "neurotransmitter_effects"])
        result = filter_claims_by_question_type(claims, "mechanism")
        cats = {c["category"] for c in result}
        assert "mechanism" in cats
        assert "neurotransmitter_effects" in cats
        assert "side_effects" not in cats
        assert "dosage_range" not in cats

    def test_always_on_included(self):
        """Always-on categories should be included even for specialized question types."""
        claims = self._make_claims(["mechanism", "class", "target"])
        result = filter_claims_by_question_type(claims, "side_effects")
        cats = {c["category"] for c in result}
        # Always-on categories should be included
        assert "mechanism" in cats
        assert "class" in cats
        assert "target" in cats
