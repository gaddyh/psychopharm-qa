"""Core domain models for psychopharm-qa.

These are the in-memory data structures used across the pipeline.
They mirror the database tables in db/tables.py but are framework-agnostic
(pydantic models, not SQLAlchemy ORM objects).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    AnswerStatus,
    ClaimType,
    ChunkType,
    FeedbackType,
    SourceType,
    SupportStatus,
)


# ---------------------------------------------------------------------------
# Drug entities
# ---------------------------------------------------------------------------


class Drug(BaseModel):
    """Canonical drug entity.

    Stahl and NbN merge under this entity. The integer id is internal;
    slug is the normalized name for lookups.
    """

    id: int | None = None
    canonical_name: str
    slug: str
    gtopdb_ligand_id: int | None = None
    constituents: str | None = None


class DrugVariant(BaseModel):
    """Canonical semantic variant of a drug (e.g. base / low / upper dose)."""

    id: int | None = None
    drug_id: int
    variant_key: str  # "base", "low", "upper"
    label: str


class DrugAlias(BaseModel):
    """Brand name or spelling variant pointing to a canonical drug.

    Classifications like "antipsychotic" are NOT aliases — they are SourceClaims.
    """

    id: int | None = None
    drug_id: int
    alias: str
    normalized_alias: str
    alias_type: str  # AliasType value
    source_document_id: int | None = None


class SourceVariantMapping(BaseModel):
    """Maps a source-specific variant ID to a canonical DrugVariant.

    Keeps nbn_id out of the canonical variant so NbN version changes
    don't break historical claims.
    """

    id: int | None = None
    source_document_id: int
    external_variant_id: str
    drug_variant_id: int
    source_label: str


# ---------------------------------------------------------------------------
# Source documents & ingestion
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """A specific version of a source (e.g. NbN fetched 2026-08-22, Stahl 6th ed).

    Unique on (source_type, content_hash). Re-running the same file reuses
    this row; a new ingestion_run is created instead.
    """

    id: int | None = None
    source_type: SourceType
    scope_key: str | None = None  # e.g. "chapter-33" for Kaplan
    content_hash: str  # sha256 of input file bytes
    version_label: str  # human-readable edition
    is_active: bool = False


class IngestionRun(BaseModel):
    """One execution of an ingestion pipeline for a source document."""

    id: int | None = None
    source_document_id: int
    source: SourceType
    status: str  # IngestionStatus value
    records_seen: int = 0
    records_written: int = 0
    warnings: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


# ---------------------------------------------------------------------------
# Claims (deterministic knowledge from NbN + Stahl)
# ---------------------------------------------------------------------------


class SourceClaim(BaseModel):
    """One atomic assertion from NbN or Stahl.

    Uniqueness: (source_document_id, drug_id, drug_variant_id, category, claim_hash).
    Each Stahl bullet and each NbN field value becomes a separate claim.

    attributes holds typed, filterable fields (target, action, fda_approved, etc.)
    locator holds precise source location (pages, section, bbox) for citation.
    """

    id: int | None = None
    source_document_id: int
    drug_id: int
    drug_variant_id: int | None = None
    category: str  # "indication", "mechanism", "side_effects", "dosing", ...
    text: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    claim_hash: str
    locator: dict[str, Any] = Field(default_factory=dict)
    claim_type: ClaimType = ClaimType.SOURCE_RAW
    derived_from_claim_id: int | None = None


# ---------------------------------------------------------------------------
# Kaplan narrative chunks
# ---------------------------------------------------------------------------


class DocumentSection(BaseModel):
    """A node in the Kaplan section tree (e.g. 33.13a Antipsychotic Agents)."""

    id: int | None = None
    source_document_id: int
    section_number: str  # "33.13a"
    title: str
    parent_section_id: int | None = None
    physical_page_start: int
    physical_page_end: int


class DocumentChunk(BaseModel):
    """A searchable Kaplan passage.

    Parent-child structure: parent_chunk_id points to a section-level chunk
    for context expansion. Child chunks (400-700 model tokens) are the primary
    search targets. Entity metadata enables filtered retrieval.
    """

    id: int | None = None
    source_document_id: int
    section_id: int | None = None
    parent_chunk_id: int | None = None
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_index: int
    text: str
    table_data: dict[str, Any] | None = None
    entity_metadata: dict[str, Any] = Field(default_factory=dict)
    physical_pdf_page: int
    input_hash: str


class DocumentEmbedding(BaseModel):
    """Embedding vector for a chunk, stored separately from chunk text.

    Model name and dimensions are explicit so changing embedding models
    doesn't require rewriting chunks.
    """

    id: int | None = None
    chunk_id: int
    model: str
    dimensions: int
    input_hash: str
    embedding: list[float]
    embedded_at: str | None = None


class ParseArtifact(BaseModel):
    """A failed/unparsed page — NOT usable for answering.

    Stored separately from source_claims so garbled pages never enter
    production retrieval. Surfaced in the review queue.
    """

    id: int | None = None
    source_document_id: int
    artifact_type: str  # "unparsed_page"
    physical_pdf_page: int
    raw_text: str
    usable_for_answering: bool = False
    warning: str


# ---------------------------------------------------------------------------
# Question understanding & retrieval
# ---------------------------------------------------------------------------


class QuestionUnderstanding(BaseModel):
    """Parsed question: which drugs, what type, key concepts."""

    raw_question: str
    drug_names: list[str] = Field(default_factory=list)
    drug_ids: list[int] = Field(default_factory=list)
    question_type: str = "general"  # "mechanism", "indication", "dosing", ...
    concepts: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """One piece of evidence in an evidence package.

    evidence_id is a stable, citable identifier (e.g. "stahl_claim_123")
    that the LLM references in its answer. The citation validator checks
    these IDs exist before rendering.
    """

    evidence_id: str  # "stahl_claim_123", "nbn_claim_456", "kaplan_chunk_789"
    source: SourceType
    drug_id: int | None = None
    category: str | None = None
    text: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    locator: dict[str, Any] = Field(default_factory=dict)
    support: SupportStatus = SupportStatus.NEUTRAL


class EvidencePackage(BaseModel):
    """The prepared evidence package sent to the LLM.

    The LLM receives this rather than searching blindly. It contains
    deterministic claims from Stahl/NbN plus Kaplan passages, with
    any conflicts or uncertainties flagged.
    """

    question: str
    drugs: dict[str, dict[str, list[EvidenceItem]]] = Field(default_factory=dict)
    kaplan_passages: list[EvidenceItem] = Field(default_factory=list)
    conflicts_or_uncertainties: list[str] = Field(default_factory=list)
    required_citations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Answer & feedback
# ---------------------------------------------------------------------------


class AnswerCitation(BaseModel):
    """A citation in a generated answer, referencing evidence IDs."""

    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """The LLM's structured output.

    The LLM cites internal evidence_ids. The renderer resolves these
    into book names + pages. The citation validator rejects nonexistent IDs.
    """

    direct_answer: str
    explanation: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    uncertainty: str | None = None
    status: AnswerStatus = AnswerStatus.ANSWERED
    clarification_question: str | None = None


class AnswerTrace(BaseModel):
    """Complete trace of one answer: question, evidence, LLM output, timing."""

    id: int | None = None
    conversation_id: int | None = None
    question: str
    question_understanding: QuestionUnderstanding
    evidence_package: EvidencePackage
    answer: GeneratedAnswer
    llm_model: str
    llm_latency_ms: int | None = None
    created_at: str | None = None


class DoctorFeedback(BaseModel):
    """Clinician feedback on an answer.

    Negative or corrected answers enter the review queue and can
    later become regression cases.
    """

    id: int | None = None
    answer_trace_id: int
    feedback_type: FeedbackType
    reason: str | None = None
    correction: str | None = None
    clinician_id: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationCase(BaseModel):
    """One test case for offline evaluation."""

    id: int | None = None
    dataset: str  # "golden_v1", "abstention_v1", "regression"
    question: str
    expected_answer: str | None = None
    expected_status: AnswerStatus | None = None
    expected_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
