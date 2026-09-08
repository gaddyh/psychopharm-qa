"""Token-budget-aware evidence packing.

Computes the Kaplan token budget after accounting for Stahl+NbN evidence,
then greedily packs complete Kaplan child chunks into the remaining budget.
Never truncates a passage at an arbitrary character boundary — sends the
complete child chunk or omits it.
"""

from __future__ import annotations

from typing import Any

import tiktoken

from ..config import get_settings

# Tokenizer (same as Kaplan ingestion)
_ENCODER = tiktoken.encoding_for_model("gpt-4o")

# Defaults (overridable via config)
DEFAULT_MODEL_CONTEXT_TOKENS = 128_000
DEFAULT_RESERVED_OUTPUT_TOKENS = 2500
DEFAULT_SAFETY_MARGIN = 0.10
DEFAULT_MAX_KAPLAN_CANDIDATES = 10
DEFAULT_MAX_KAPLAN_RESULTS = 5
DEFAULT_MAX_CHILD_CHUNK_TOKENS = 700


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def compute_kaplan_budget(
    system_prompt: str,
    question: str,
    schema_json: str,
    selected_deterministic_evidence: dict[str, Any],
    model_context_tokens: int | None = None,
    reserved_output_tokens: int | None = None,
    safety_margin: float | None = None,
) -> int:
    """Compute the token budget available for Kaplan passages.

    total input budget
    − system prompt
    − question
    − schema
    − selected NbN/Stahl evidence
    − reserved output
    − safety margin
    = Kaplan budget
    """
    ctx = model_context_tokens or DEFAULT_MODEL_CONTEXT_TOKENS
    reserved = reserved_output_tokens or DEFAULT_RESERVED_OUTPUT_TOKENS
    margin = safety_margin or DEFAULT_SAFETY_MARGIN

    system_tokens = _count_tokens(system_prompt)
    question_tokens = _count_tokens(question)
    schema_tokens = _count_tokens(schema_json)

    # Count deterministic evidence tokens
    evidence_tokens = 0
    for drug_data in selected_deterministic_evidence.get("drugs", {}).values():
        for e in drug_data.get("nbn_evidence", []):
            evidence_tokens += _count_tokens(e.get("text", ""))
        for e in drug_data.get("stahl_evidence", []):
            evidence_tokens += _count_tokens(e.get("text", ""))

    # Also count conflicts/uncertainties
    for c in selected_deterministic_evidence.get("conflicts_or_uncertainties", []):
        evidence_tokens += _count_tokens(c)

    safety_tokens = int(ctx * margin)

    kaplan_budget = (
        ctx
        - system_tokens
        - question_tokens
        - schema_tokens
        - evidence_tokens
        - reserved
        - safety_tokens
    )

    return max(kaplan_budget, 0)


def pack_kaplan(
    passages: list[dict[str, Any]],
    budget: int,
    max_results: int = DEFAULT_MAX_KAPLAN_RESULTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Greedily pack complete Kaplan passages into the token budget.

    Never truncates a passage — sends the complete chunk or omits it.

    Args:
        passages: Ranked list of Kaplan passage dicts (highest relevance first).
        budget: Token budget for Kaplan passages.
        max_results: Maximum number of passages to include.

    Returns:
        (selected, omitted, diagnostics) where:
        - selected: passages that fit within the budget
        - omitted: passages that were skipped
        - diagnostics: dict with budget details
    """
    selected: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    remaining_budget = budget
    total_tokens_used = 0

    # Only consider passages up to max_results — strictly respect the limit
    for passage in passages[:max_results]:
        text = passage.get("text", "")
        passage_tokens = _count_tokens(text)

        if passage_tokens <= remaining_budget:
            selected.append(passage)
            remaining_budget -= passage_tokens
            total_tokens_used += passage_tokens
        else:
            omitted.append(passage)

    # Passages beyond max_results are always omitted
    for passage in passages[max_results:]:
        omitted.append(passage)

    diagnostics = {
        "input_token_budget": budget,
        "evidence_tokens_used": total_tokens_used,
        "kaplan_candidates": len(passages),
        "kaplan_passages_included": len(selected),
        "passages_omitted_for_budget": len(omitted),
    }

    return selected, omitted, diagnostics
