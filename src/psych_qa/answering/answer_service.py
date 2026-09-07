"""Answer service — orchestrates the full answer pipeline.

parse question → drug lookup → Kaplan retrieval → evidence package → LLM answer → validate citations → compute metrics → save trace
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..llm.client import get_llm_client
from ..llm.schemas import ANSWER_SCHEMA
from ..repositories import conversations as conv_repo
from ..retrieval.question_parser import parse_question
from .citations import strip_invalid_citations, validate_citations
from .evidence_builder import build_evidence_package
from .metrics import compute_run_metrics
from .prompts import build_answer_prompt, build_evidence_prompt
from .sufficiency import check_sufficiency

logger = logging.getLogger(__name__)


def answer_question(question: str, clinician_id: str | None = None) -> dict[str, Any]:
    """Answer a psychopharmacology question with cited evidence.

    This is the main entry point for the answering pipeline.

    Returns:
        Dict with: question, understanding, evidence_package, answer, trace_id, llm_model, latency_ms, metrics
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
    is_sufficient, reason = check_sufficiency(evidence_package)
    invalid_citation_ids: list[str] = []
    llm_latency_ms: int | None = None

    if not is_sufficient:
        logger.info(f"Insufficient evidence: {reason}")
        answer = {
            "direct_answer": "",
            "explanation": reason or "Insufficient evidence to answer this question.",
            "citations": [],
            "uncertainty": reason,
            "status": "abstained",
            "clarification_question": None,
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

        # 5. Validate citations
        is_valid, invalid_citation_ids = validate_citations(answer, evidence_package.get("required_citations", []))
        if not is_valid:
            logger.warning(f"Stripping invalid citations: {invalid_citation_ids}")
            answer = strip_invalid_citations(answer, evidence_package.get("required_citations", []))

    total_latency_ms = int((time.time() - t0) * 1000)

    # 6. Compute metrics
    metrics = compute_run_metrics(
        question=question,
        understanding=understanding,
        evidence_package=evidence_package,
        answer=answer,
        invalid_citation_ids=invalid_citation_ids,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    # 7. Save trace
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
    }


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
