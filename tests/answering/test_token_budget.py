"""Tests for token-budget-aware Kaplan packing."""

from __future__ import annotations

import pytest

from psych_qa.answering.token_budget import (
    compute_kaplan_budget,
    pack_kaplan,
)


class TestComputeKaplanBudget:
    def test_basic_budget(self):
        budget = compute_kaplan_budget(
            system_prompt="You are a helpful assistant.",
            question="What is amisulpride?",
            schema_json='{"type": "object"}',
            selected_deterministic_evidence={"drugs": {}, "conflicts_or_uncertainties": []},
            model_context_tokens=128_000,
            reserved_output_tokens=2500,
            safety_margin=0.10,
        )
        # 128000 - small prompts - 2500 - 12800 = ~112000+
        assert budget > 100_000

    def test_budget_with_evidence(self):
        budget = compute_kaplan_budget(
            system_prompt="You are a helpful assistant.",
            question="What is amisulpride?",
            schema_json='{"type": "object"}',
            selected_deterministic_evidence={
                "drugs": {
                    "amisulpride": {
                        "nbn_evidence": [{"text": "x" * 1000}],
                        "stahl_evidence": [{"text": "y" * 1000}],
                    }
                },
                "conflicts_or_uncertainties": ["conflict"],
            },
            model_context_tokens=128_000,
            reserved_output_tokens=2500,
            safety_margin=0.10,
        )
        # Budget should be smaller due to evidence tokens
        assert budget > 90_000

    def test_budget_never_negative(self):
        """Budget should never go negative — return at least 0."""
        budget = compute_kaplan_budget(
            system_prompt="x" * 200_000,
            question="y" * 200_000,
            schema_json="z" * 200_000,
            selected_deterministic_evidence={"drugs": {}, "conflicts_or_uncertainties": []},
            model_context_tokens=1000,
        )
        assert budget >= 0


class TestPackKaplan:
    def _make_passages(self, n, tokens_each=100):
        """Make n passages with approximately tokens_each tokens each."""
        # ~4 chars per token
        text = "word " * (tokens_each * 1)
        return [{"evidence_id": f"kaplan_{i}", "text": text, "physical_pdf_page": 10000 + i} for i in range(n)]

    def test_pack_all_fit(self):
        passages = self._make_passages(3, tokens_each=100)
        selected, omitted, diag = pack_kaplan(passages, budget=1000, max_results=5)
        assert len(selected) == 3
        assert len(omitted) == 0
        assert diag["kaplan_passages_included"] == 3

    def test_pack_some_omitted(self):
        passages = self._make_passages(5, tokens_each=100)
        # Budget only fits 2 passages
        selected, omitted, diag = pack_kaplan(passages, budget=250, max_results=5)
        assert len(selected) == 2
        assert len(omitted) == 3
        assert diag["passages_omitted_for_budget"] == 3

    def test_pack_respects_max_results(self):
        passages = self._make_passages(10, tokens_each=50)
        selected, omitted, diag = pack_kaplan(passages, budget=10000, max_results=3)
        assert len(selected) == 3
        assert diag["kaplan_passages_included"] == 3

    def test_pack_no_truncation(self):
        """Passages should be sent complete, never truncated."""
        long_text = "This is a complete passage. " * 20
        passages = [{"evidence_id": "kaplan_1", "text": long_text, "physical_pdf_page": 10000}]
        selected, _, _ = pack_kaplan(passages, budget=10000, max_results=5)
        assert selected[0]["text"] == long_text  # exact same text, no truncation

    def test_pack_empty_passages(self):
        selected, omitted, diag = pack_kaplan([], budget=1000, max_results=5)
        assert selected == []
        assert diag["kaplan_candidates"] == 0

    def test_pack_diagnostics(self):
        passages = self._make_passages(3, tokens_each=100)
        selected, omitted, diag = pack_kaplan(passages, budget=1000, max_results=5)
        assert "input_token_budget" in diag
        assert "evidence_tokens_used" in diag
        assert "kaplan_candidates" in diag
        assert "kaplan_passages_included" in diag
        assert "passages_omitted_for_budget" in diag
