"""Pydantic schemas for evaluation cases and trace versioning.

Gold cases use stable evidence identifiers (claim_hash / locator based),
not database-generated IDs, so they survive DB recreation and re-ingestion.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    NEEDS_SASSON_APPROVAL = "needs_sasson_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExpectedStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NEEDS_CLARIFICATION = "needs_clarification"


class AcceptableEvidence(BaseModel):
    """A stable reference to evidence that may support an answer point.

    Uses claim_hash for Stahl/NbN claims and locator for Kaplan passages.
    Multiple AcceptableEvidence entries per point allow different passages
    to legitimately support the same answer.
    """
    source: str  # "stahl", "nbn", "kaplan"
    drug_slug: str | None = None  # for stahl/nbn
    category: str | None = None  # for stahl/nbn
    claim_hash: str | None = None  # stable hash from ingestion
    locator: dict[str, Any] | None = None  # for kaplan: {physical_pdf_page, section_number}
    text_contains: str | None = None  # fallback: match by text content


class RequiredAnswerPoint(BaseModel):
    """A point that the answer must cover."""
    text: str
    acceptable_evidence: list[AcceptableEvidence] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    """A previous turn in a conversation (for multi-turn gold cases)."""
    turn: int = 1
    question: str
    resolved_question: str | None = None
    understanding: dict[str, Any] = Field(default_factory=dict)
    answer_summary: str = ""


class GoldCase(BaseModel):
    """A gold evaluation case with required/forbidden points."""
    question: str
    expected_status: ExpectedStatus = ExpectedStatus.ANSWERED
    required_answer_points: list[RequiredAnswerPoint] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    acceptable_evidence: list[AcceptableEvidence] = Field(default_factory=list)
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    reviewer_notes: str = ""
    review_status: ReviewStatus = ReviewStatus.NEEDS_SASSON_APPROVAL
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AbstentionCase(BaseModel):
    """A case where the system should abstain or request clarification."""
    question: str
    expected_status: ExpectedStatus  # abstained or needs_clarification
    reason: str = ""  # why the sources can't answer this
    reviewer_notes: str = ""
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.NEEDS_SASSON_APPROVAL
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionCase(BaseModel):
    """A regression case converted from clinician feedback."""
    question: str
    expected_status: ExpectedStatus = ExpectedStatus.ANSWERED
    required_answer_points: list[RequiredAnswerPoint] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    acceptable_evidence: list[AcceptableEvidence] = Field(default_factory=list)
    reviewer_notes: str = ""
    review_status: ReviewStatus = ReviewStatus.NEEDS_SASSON_APPROVAL
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    source_trace_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceVersions(BaseModel):
    """Version metadata stamped on every answer trace for reproducibility."""
    answer_schema_version: int = 2
    prompt_version: str = "claims-v1"
    entailment_prompt_version: str | None = None
    judge_model: str | None = None
    retriever_version: str = "hybrid-v1"
    corpus_versions: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hard challenge cases — designed to expose reasoning and trust boundary failures
# ---------------------------------------------------------------------------

class ExpectedOutcome(str, Enum):
    """Expected outcome category for hard challenge cases."""
    SUPPORTED_ANSWER = "supported_answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFICATION_REQUIRED = "clarification_required"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNSUPPORTED_SPECIFICITY = "unsupported_specificity"
    OUT_OF_CORPUS = "out_of_corpus"
    CLINICAL_BOUNDARY = "clinical_boundary"


class HardGoldCase(BaseModel):
    """A challenging conversation case designed to expose where reasoning
    and trust boundaries break.

    Unlike regular gold cases, these are scored for required and forbidden
    behaviors — not only retrieval recall. Performance is expected to fall
    below 100%, which is healthy: the set should reveal the next real
    engineering problems.
    """
    question: str
    expected_status: ExpectedStatus = ExpectedStatus.ANSWERED
    expected_outcome: ExpectedOutcome = ExpectedOutcome.PARTIAL_ANSWER
    expected_resolution: str | None = None
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    required_behaviors: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    required_answer_points: list[RequiredAnswerPoint] = Field(default_factory=list)
    reviewer_notes: str = ""
    review_status: ReviewStatus = ReviewStatus.NEEDS_SASSON_APPROVAL
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def parse_case(raw: dict[str, Any], dataset: str) -> GoldCase | AbstentionCase | RegressionCase | HardGoldCase:
    """Parse a raw JSONL dict into the appropriate case type."""
    if dataset.startswith("abstention"):
        return AbstentionCase(**raw)
    elif dataset == "regression":
        return RegressionCase(**raw)
    elif dataset == "golden_hard":
        return HardGoldCase(**raw)
    else:
        return GoldCase(**raw)
