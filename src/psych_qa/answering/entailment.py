"""Claim-to-evidence entailment validation.

For each claim in the answer, check whether the cited evidence actually
supports the claim. Uses an LLM judge with a structured verdict.

Verdicts:
- SUPPORTED: evidence entails the claim
- UNSUPPORTED: evidence does not support the claim
- UNSUPPORTED_SPECIFICITY: evidence supports the general claim but the
  claim adds specificity not present in the evidence
- CONTRADICTED: evidence contradicts the claim
- MISSING_CITATION: claim has no evidence_ids (should be caught earlier)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..llm.client import get_llm_client
from ..llm.schemas import ENTAILMENT_SCHEMA

logger = logging.getLogger(__name__)

ENTAILMENT_PROMPT_VERSION = "entailment-v1"


def judge_claim(
    claim_text: str,
    evidence_texts: list[str],
    claim_role: str = "direct",
    is_critical: bool = False,
) -> dict[str, Any]:
    """Judge whether evidence supports a single claim.

    Returns dict with: verdict, reasoning, unsupported_specificity_detail
    """
    if not evidence_texts:
        return {
            "verdict": "MISSING_CITATION",
            "reasoning": "Claim has no cited evidence.",
            "unsupported_specificity_detail": None,
        }

    client = get_llm_client()

    evidence_block = "\n\n".join(
        f"[Evidence {i+1}] {text}" for i, text in enumerate(evidence_texts)
    )

    prompt = f"""You are a clinical evidence auditor. Determine whether the cited evidence supports the medical claim.

Claim: "{claim_text}"
Claim role: {claim_role}
Critical (clinical safety): {is_critical}

Cited Evidence:
{evidence_block}

Verdict rules:
- SUPPORTED: The evidence entails the claim. The claim does not add information beyond what the evidence states.
- UNSUPPORTED: The evidence does not support the claim at all.
- UNSUPPORTED_SPECIFICITY: The evidence supports the general idea, but the claim adds specific details (numbers, qualifiers, comparisons) not present in the evidence.
- CONTRADICTED: The evidence directly contradicts the claim.
- MISSING_CITATION: No evidence was provided.

Be strict. If the claim mentions a specific dose, mechanism, or comparison that is not in the evidence, mark as UNSUPPORTED_SPECIFICITY."""

    messages = [{"role": "user", "content": prompt}]

    try:
        result, _, _, _ = client.chat_structured(messages, schema=ENTAILMENT_SCHEMA)
        return result
    except Exception as e:
        logger.warning(f"Entailment judging failed ({e}), defaulting to UNSUPPORTED")
        return {
            "verdict": "UNSUPPORTED",
            "reasoning": f"Entailment judge failed: {e}",
            "unsupported_specificity_detail": None,
        }


def judge_claims_batch(
    claims: list[dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Judge entailment for all claims in an answer.

    Returns a list of verdict dicts, one per claim, with:
    - claim_index
    - claim_text
    - verdict
    - reasoning
    - unsupported_specificity_detail
    - is_critical
    - evidence_ids
    """
    verdicts: list[dict[str, Any]] = []

    for i, claim in enumerate(claims):
        claim_text = claim.get("text", "")
        evidence_ids = claim.get("evidence_ids", [])
        claim_role = claim.get("claim_role", "direct")
        is_critical = claim.get("critical", False)

        # Resolve evidence texts
        evidence_texts: list[str] = []
        for eid in evidence_ids:
            evidence = evidence_lookup.get(eid)
            if evidence:
                evidence_texts.append(evidence.get("text", ""))

        result = judge_claim(
            claim_text=claim_text,
            evidence_texts=evidence_texts,
            claim_role=claim_role,
            is_critical=is_critical,
        )

        verdicts.append({
            "claim_index": i,
            "claim_text": claim_text,
            "verdict": result.get("verdict", "UNSUPPORTED"),
            "reasoning": result.get("reasoning", ""),
            "unsupported_specificity_detail": result.get("unsupported_specificity_detail"),
            "is_critical": is_critical,
            "evidence_ids": evidence_ids,
        })

    return verdicts


def should_regenerate(verdicts: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Determine if the answer should be regenerated based on verdicts.

    Returns (should_regenerate, reasons).
    Regenerates if any critical claim is not SUPPORTED.
    """
    reasons: list[str] = []
    for v in verdicts:
        if v.get("is_critical") and v.get("verdict") != "SUPPORTED":
            reasons.append(
                f"Critical claim '{v['claim_text'][:60]}...' verdict: {v['verdict']}"
            )
    return len(reasons) > 0, reasons


def should_abstain_after_regeneration(verdicts: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Determine if the system should abstain after a failed regeneration.

    Returns (should_abstain, reasons).
    Abstains if any critical claim is still not SUPPORTED after regeneration.
    """
    return should_regenerate(verdicts)
