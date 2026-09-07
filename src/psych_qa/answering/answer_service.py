"""Answer service — orchestrates the full answer pipeline (claims-first V2).

parse question → drug lookup → Kaplan retrieval → evidence package →
LLM answer (claims) → assign criticality → validate citations →
entailment gate → assemble prose → compute metrics → save trace
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..llm.client import get_llm_client
from ..llm.schemas import ANSWER_SCHEMA
from ..repositories import conversations as conv_repo
from ..retrieval.question_parser import parse_question
from .citations import drop_unsupported_claims, validate_citations, validate_every_claim_has_evidence
from .criticality import assign_critical
from .evidence_builder import build_evidence_package
from .entailment import (
    ENTAILMENT_PROMPT_VERSION,
    judge_claims_batch,
    should_abstain_after_regeneration,
    should_regenerate,
)
from .metrics import compute_run_metrics
from .prompts import build_answer_prompt, build_evidence_prompt
from .sufficiency import check_sufficiency

logger = logging.getLogger(__name__)

# Version metadata stamped on every trace
ANSWER_SCHEMA_VERSION = 2
PROMPT_VERSION = "claims-v1"
RETRIEVER_VERSION = "hybrid-v1"


def _build_versions(evidence_package: dict[str, Any]) -> dict[str, Any]:
    """Build the trace version block for reproducibility."""
    client = get_llm_client()
    return {
        "answer_schema_version": ANSWER_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "entailment_prompt_version": ENTAILMENT_PROMPT_VERSION,
        "judge_model": client.chat_model,
        "retriever_version": RETRIEVER_VERSION,
        "corpus_versions": {},  # filled from source_documents if available
    }


def answer_question(question: str, clinician_id: str | None = None) -> dict[str, Any]:
    """Answer a psychopharmacology question with cited evidence.

    Returns:
        Dict with: question, understanding, evidence_package, answer, trace_id,
        llm_model, latency_ms, metrics, versions
    """
    t0 = time.time()

    # 1. Parse question
    logger.info(f"Parsing question: {question[:80]}...")
    understanding = parse_question(question)

    # 2. Build evidence package
    logger.info("Building evidence package...")
    evidence_package_model, raw_data = build_evidence_package(question, understanding)
    evidence_package = evidence_package_model.model_dump()

    # 3. Check sufficiency
    is_sufficient, reason = check_sufficiency(evidence_package, understanding)
    invalid_citation_ids: list[str] = []
    llm_latency_ms: int | None = None
    entailment_verdicts: list[dict[str, Any]] = []
    regeneration_attempts = 0

    if not is_sufficient:
        logger.info(f"Insufficient evidence: {reason}")
        answer = {
            "claims": [],
            "status": "abstained",
            "uncertainties": [reason] if reason else [],
            "clarification_question": None,
            "direct_answer": "",
            "explanation": reason or "Insufficient evidence to answer this question.",
        }
    else:
        # 4. Generate answer with LLM
        logger.info("Generating answer with LLM...")
        evidence_text = build_evidence_prompt(evidence_package)
        messages = build_answer_prompt(question, evidence_text)

        client = get_llm_client()
        answer, prompt_tokens, completion_tokens, llm_latency_ms = client.chat_structured_timed(
            messages, schema=ANSWER_SCHEMA
        )

        # 5. Assign criticality deterministically
        evidence_lookup = build_evidence_lookup(evidence_package)
        answer = _assign_criticality_to_answer(answer, evidence_lookup)

        # 6. Validate citations (ID existence)
        is_valid, invalid_citation_ids = validate_citations(answer, evidence_package.get("required_citations", []))
        if not is_valid:
            logger.warning(f"Invalid citation IDs: {invalid_citation_ids}")
            answer = drop_unsupported_claims(answer, evidence_package.get("required_citations", []))

        # 7. Validate every claim has evidence
        has_evidence, _ = validate_every_claim_has_evidence(answer)
        if not has_evidence:
            logger.warning("Some claims have no evidence_ids after validation")

        # 8. Entailment gate — judge claim-to-evidence support
        claims = answer.get("claims", [])
        if claims:
            entailment_verdicts = judge_claims_batch(claims, evidence_lookup)

            # Check if regeneration is needed (critical claims not SUPPORTED)
            regenerate, regen_reasons = should_regenerate(entailment_verdicts)
            if regenerate:
                regeneration_attempts = 1
                logger.info(f"Regenerating answer due to: {regen_reasons}")
                answer, _, _, regen_latency = client.chat_structured_timed(
                    messages, schema=ANSWER_SCHEMA
                )
                llm_latency_ms = (llm_latency_ms or 0) + (regen_latency or 0)
                answer = _assign_criticality_to_answer(answer, evidence_lookup)
                is_valid, invalid_citation_ids = validate_citations(
                    answer, evidence_package.get("required_citations", [])
                )
                if not is_valid:
                    answer = drop_unsupported_claims(answer, evidence_package.get("required_citations", []))
                entailment_verdicts = judge_claims_batch(answer.get("claims", []), evidence_lookup)

                # If still failing after regeneration, abstain
                abstain, abstain_reasons = should_abstain_after_regeneration(entailment_verdicts)
                if abstain:
                    logger.warning(f"Abstaining after regeneration: {abstain_reasons}")
                    answer = {
                        "claims": [],
                        "status": "abstained",
                        "uncertainties": [
                            "Answer could not be validated against evidence after regeneration."
                        ] + abstain_reasons,
                        "clarification_question": None,
                        "direct_answer": "",
                        "explanation": "The system generated an answer but could not verify that all critical claims are supported by the cited evidence.",
                    }

        # 9. Assemble prose from validated claims
        from ..domain.models import AnswerClaim, assemble_prose
        claim_models = [AnswerClaim(**c) for c in answer.get("claims", [])]
        direct_answer, explanation = assemble_prose(claim_models)
        answer["direct_answer"] = direct_answer
        answer["explanation"] = explanation

    total_latency_ms = int((time.time() - t0) * 1000)

    # 10. Build version metadata
    versions = _build_versions(evidence_package)

    # 11. Compute metrics
    budget_diagnostics = raw_data.get("budget_diagnostics")
    metrics = compute_run_metrics(
        question=question,
        understanding=understanding,
        evidence_package=evidence_package,
        answer=answer,
        invalid_citation_ids=invalid_citation_ids,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
        entailment_verdicts=entailment_verdicts if entailment_verdicts else None,
        regeneration_attempts=regeneration_attempts,
        budget_diagnostics=budget_diagnostics,
    )

    # 12. Save trace
    logger.info("Saving answer trace...")
    conversation_id = conv_repo.create_conversation(title=question[:80], clinician_id=clinician_id)
    trace_id = conv_repo.save_answer_trace(
        conversation_id=conversation_id,
        question=question,
        question_understanding=understanding,
        evidence_package=evidence_package,
        answer=answer,
        llm_model=get_llm_client().chat_model,
        llm_latency_ms=llm_latency_ms,
        metrics=metrics,
        versions=versions,
    )

    return {
        "trace_id": trace_id,
        "question": question,
        "understanding": understanding,
        "evidence_package": evidence_package,
        "answer": answer,
        "llm_model": get_llm_client().chat_model,
        "latency_ms": llm_latency_ms,
        "total_latency_ms": total_latency_ms,
        "metrics": metrics,
        "versions": versions,
    }


def _assign_criticality_to_answer(
    answer: dict[str, Any],
    evidence_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assign the critical flag to each claim in the answer."""
    claims = answer.get("claims", [])
    claims = assign_critical(claims, evidence_lookup)
    answer["claims"] = claims
    return answer


def build_evidence_lookup(evidence_package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a lookup dict from evidence_id → evidence item for citation rendering."""
    lookup: dict[str, dict[str, Any]] = {}

    for drug_name, drug_data in evidence_package.get("drugs", {}).items():
        for e in drug_data.get("nbn_evidence", []):
            lookup[e["evidence_id"]] = e
        for e in drug_data.get("stahl_evidence", []):
            lookup[e["evidence_id"]] = e

    for e in evidence_package.get("kaplan_passages", []):
        lookup[e["evidence_id"]] = e

    return lookup
