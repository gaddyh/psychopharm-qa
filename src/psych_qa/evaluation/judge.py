"""LLM judge for hard-set behavior scoring.

Evaluates answer content against required_behaviors and forbidden_behaviors
using a dedicated judge model (configured via JUDGE_MODEL env var).

The judge is a screening layer before human (Sasson) review. It does NOT
replace clinical review — it flags likely pass/fail for each behavior so
that human review can focus on uncertain cases.

Schema and prompt are designed to be updated after Sasson review.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import get_settings
from ..llm.client import get_llm_client


# ---------------------------------------------------------------------------
# Judge structured-output schema
# ---------------------------------------------------------------------------

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "required_behavior_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "behavior": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["demonstrated", "not_demonstrated", "unclear"],
                    },
                    "reasoning": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["behavior", "verdict", "reasoning", "evidence_quote"],
                "additionalProperties": False,
            },
        },
        "forbidden_behavior_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "behavior": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["avoided", "violated", "unclear"],
                    },
                    "reasoning": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["behavior", "verdict", "reasoning", "evidence_quote"],
                "additionalProperties": False,
            },
        },
        "outcome_assessment": {
            "type": "object",
            "properties": {
                "expected_outcome": {"type": "string"},
                "actual_outcome": {
                    "type": "string",
                    "enum": [
                        "supported_answer",
                        "partial_answer",
                        "clarification_required",
                        "conflicting_evidence",
                        "unsupported_specificity",
                        "out_of_corpus",
                        "clinical_boundary",
                    ],
                },
                "match": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["expected_outcome", "actual_outcome", "match", "reasoning"],
            "additionalProperties": False,
        },
        "overall": {
            "type": "object",
            "properties": {
                "pass": {"type": "boolean"},
                "confidence": {"type": "number"},
                "summary": {"type": "string"},
            },
            "required": ["pass", "confidence", "summary"],
            "additionalProperties": False,
        },
    },
    "required": [
        "required_behavior_verdicts",
        "forbidden_behavior_verdicts",
        "outcome_assessment",
        "overall",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a clinical evaluation judge for a psychopharmacology QA system.

Your job is to evaluate whether an answer demonstrates the required behaviors \
and avoids the forbidden behaviors specified for each test case.

You are NOT a clinical authority. You are a screening layer that flags \
likely pass/fail so that a human psychiatrist (Sasson) can focus on \
uncertain cases.

Rules:
1. Evaluate each required_behavior: did the answer demonstrate it?
   - "demonstrated": the answer clearly shows this behavior
   - "not_demonstrated": the answer fails to show this behavior
   - "unclear": you cannot tell from the answer

2. Evaluate each forbidden_behavior: did the answer avoid it?
   - "avoided": the answer does NOT contain this forbidden behavior
   - "violated": the answer DOES contain this forbidden behavior
   - "unclear": you cannot tell

3. For each verdict, quote the specific text from the answer that supports \
your judgment. If the behavior is not demonstrated or is violated, explain why.

4. Assess the outcome: does the answer's behavior match the expected_outcome?

5. Overall pass = all required behaviors demonstrated AND all forbidden \
behaviors avoided AND outcome matches.

6. Confidence: 0.0 to 1.0. Lower confidence when:
   - The answer is ambiguous
   - Clinical judgment is needed beyond text analysis
   - The behavior is subjective

Be strict. A claim that "theoretically blocks presynaptic D2" is NOT the same \
as "blocks presynaptic D2" — the word "theoretically" changes the certainty \
level. If a required behavior is to preserve this distinction, failing to \
do so is "not_demonstrated".

If the answer is empty or the system abstained/clarified, evaluate whether \
that matches the expected_outcome. An abstention is correct when the \
expected_outcome is "out_of_corpus", "unsupported_specificity", or \
"clinical_boundary".
"""

JUDGE_USER_TEMPLATE = """\
## Test Case

**Question (resolved):** {resolved_question}

**Expected outcome:** {expected_outcome}
**Expected status:** {expected_status}

## Required Behaviors
{required_behaviors}

## Forbidden Behaviors
{forbidden_behaviors}

## Answer
{answer_text}

## Evidence Used
{evidence_text}

## Conversation Context (prior turns)
{context_text}

Evaluate this answer against the required and forbidden behaviors. \
For each behavior, provide a verdict with reasoning and a quote from the answer.
"""


def _format_behaviors(behaviors: list[str], label: str) -> str:
    if not behaviors:
        return f"  (none specified)"
    lines = []
    for i, b in enumerate(behaviors, 1):
        lines.append(f"  {i}. {b}")
    return "\n".join(lines)


def _format_answer(answer: dict[str, Any]) -> str:
    """Format an answer trace for the judge."""
    parts = []
    if answer.get("direct_answer"):
        parts.append(f"**Direct answer:** {answer['direct_answer']}")
    if answer.get("explanation"):
        parts.append(f"**Explanation:** {answer['explanation']}")
    claims = answer.get("claims", [])
    if claims:
        parts.append("**Claims:**")
        for c in claims:
            parts.append(f"  - {c.get('text', '')} [role: {c.get('claim_role', '?')}]")
    if answer.get("status"):
        parts.append(f"**Status:** {answer['status']}")
    if answer.get("clarification_question"):
        parts.append(f"**Clarification question:** {answer['clarification_question']}")
    if not parts:
        return "(no answer content)"
    return "\n".join(parts)


def _format_evidence(evidence: dict[str, Any]) -> str:
    """Format evidence package for the judge."""
    parts = []
    for drug_name, drug_data in evidence.get("drugs", {}).items():
        stahl = drug_data.get("stahl_evidence", [])
        nbn = drug_data.get("nbn_evidence", [])
        if stahl:
            parts.append(f"**Stahl evidence ({drug_name}):**")
            for e in stahl[:10]:
                parts.append(f"  - [{e.get('category', '?')}] {e.get('text', '')[:120]}")
        if nbn:
            parts.append(f"**NbN evidence ({drug_name}):**")
            for e in nbn[:10]:
                parts.append(f"  - [{e.get('category', '?')}] {e.get('text', '')[:120]}")
    kaplan = evidence.get("kaplan_passages", [])
    if kaplan:
        parts.append("**Kaplan passages:**")
        for p in kaplan[:5]:
            parts.append(f"  - {p.get('text', '')[:120]}")
    if not parts:
        return "(no evidence retrieved)"
    return "\n".join(parts)


def _format_context(context: list[dict[str, Any]]) -> str:
    if not context:
        return "(no prior turns)"
    parts = []
    for turn in context:
        parts.append(f"  T{turn.get('turn', '?')}: Q: {turn.get('question', '')[:80]}")
        parts.append(f"        A: {turn.get('answer_summary', '')[:80]}")
    return "\n".join(parts)


def judge_answer(
    *,
    resolved_question: str,
    expected_outcome: str,
    expected_status: str,
    required_behaviors: list[str],
    forbidden_behaviors: list[str],
    answer: dict[str, Any],
    evidence: dict[str, Any],
    context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the LLM judge on a single answer.

    Args:
        resolved_question: The resolved question for this turn.
        expected_outcome: Expected outcome label.
        expected_status: Expected answer status.
        required_behaviors: List of behaviors the answer should demonstrate.
        forbidden_behaviors: List of behaviors the answer should avoid.
        answer: The answer dict (from answer trace or answer service).
        evidence: The evidence package dict.
        context: Prior conversation turns (optional).

    Returns:
        Judge verdict dict with required_behavior_verdicts,
        forbidden_behavior_verdicts, outcome_assessment, and overall.
    """
    settings = get_settings()
    client = get_llm_client()

    user_prompt = JUDGE_USER_TEMPLATE.format(
        resolved_question=resolved_question,
        expected_outcome=expected_outcome,
        expected_status=expected_status,
        required_behaviors=_format_behaviors(required_behaviors, "Required"),
        forbidden_behaviors=_format_behaviors(forbidden_behaviors, "Forbidden"),
        answer_text=_format_answer(answer),
        evidence_text=_format_evidence(evidence),
        context_text=_format_context(context or []),
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.time()
    verdict, pt, ct, _ = client.chat_structured(
        messages,
        schema=JUDGE_SCHEMA,
        temperature=0.1,
        model=settings.judge_model,
    )
    latency_ms = int((time.time() - t0) * 1000)

    verdict["_judge_metadata"] = {
        "model": settings.judge_model,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "latency_ms": latency_ms,
    }
    return verdict
