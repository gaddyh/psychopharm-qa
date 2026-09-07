"""SQLAlchemy ORM table definitions.

This is the single source of truth for the database schema.
Alembic migrations can be generated from this via `alembic revision --autogenerate`.

Key design decisions (see plan):
- drugs.id is an internal integer PK; slug is a unique column, not the PK.
- source_documents unique on (source_type, content_hash) — re-running same file
  reuses the document, creates a new ingestion_run.
- source_claims uniqueness uses COALESCE(drug_variant_id, 0) for NULLS NOT DISTINCT.
- Claims have typed attributes (jsonb) + locator (jsonb).
- Embeddings live in document_embeddings, separate from document_chunks.
- parse_artifacts store failed pages — NOT in source_claims.
- Partial unique index enforces one active source_document per (source_type, scope_key).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sqltext
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Drug entities
# ---------------------------------------------------------------------------


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    gtopdb_ligand_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    constituents: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )

    variants: Mapped[list[DrugVariant]] = relationship(back_populates="drug")
    aliases: Mapped[list[DrugAlias]] = relationship(back_populates="drug")


class DrugVariant(Base):
    __tablename__ = "drug_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), nullable=False)
    variant_key: Mapped[str] = mapped_column(Text, nullable=False)  # base/low/upper
    label: Mapped[str] = mapped_column(Text, nullable=False)

    drug: Mapped[Drug] = relationship(back_populates="variants")

    __table_args__ = (UniqueConstraint("drug_id", "variant_key", name="uq_drug_variant_key"),)


class DrugAlias(Base):
    """Brand names and spelling variants → canonical drug.

    normalized_alias is NOT globally unique — brands can be ambiguous.
    Resolution returns candidates; a manual override table handles collisions.
    """

    __tablename__ = "drug_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), nullable=False)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(Text, nullable=False)
    alias_type: Mapped[str] = mapped_column(Text, nullable=False)  # brand / spelling_variant
    source_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_documents.id"), nullable=True
    )

    drug: Mapped[Drug] = relationship(back_populates="aliases")

    __table_args__ = (Index("ix_drug_aliases_normalized", "normalized_alias"),)


class SourceVariantMapping(Base):
    """Maps source-specific variant IDs to canonical DrugVariants.

    Keeps nbn_id out of the canonical variant so NbN version changes
    don't break historical claims.
    """

    __tablename__ = "source_variant_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id"), nullable=False
    )
    external_variant_id: Mapped[str] = mapped_column(Text, nullable=False)
    drug_variant_id: Mapped[int] = mapped_column(ForeignKey("drug_variants.id"), nullable=False)
    source_label: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "source_document_id", "external_variant_id", name="uq_source_variant_mapping"
        ),
    )


# ---------------------------------------------------------------------------
# Source documents & ingestion
# ---------------------------------------------------------------------------


class SourceDocument(Base):
    """A specific version of a source.

    Unique on (source_type, content_hash). Re-running same file reuses this row.
    is_active with partial unique index enforces one active doc per scope.
    """

    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)  # nbn/stahl/kaplan
    scope_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # e.g. chapter-33
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    version_label: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("source_type", "content_hash", name="uq_source_doc_hash"),
        # One active document per (source_type, scope_key)
        # Using a raw SQL index for the partial unique index
        Index(
            "uq_source_doc_active",
            "source_type",
            "scope_key",
            unique=True,
            postgresql_where=sqltext("is_active = true"),
        ),
    )


class IngestionRun(Base):
    """One execution of an ingestion pipeline."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id"), nullable=False
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    records_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warnings: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Claims (deterministic knowledge from NbN + Stahl)
# ---------------------------------------------------------------------------


class SourceClaim(Base):
    """One atomic assertion from NbN or Stahl.

    Uniqueness: (source_document_id, drug_id, COALESCE(drug_variant_id, 0), category, claim_hash)
    Each Stahl bullet and each NbN field value becomes a separate claim.
    """

    __tablename__ = "source_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id"), nullable=False
    )
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), nullable=False)
    drug_variant_id: Mapped[int | None] = mapped_column(
        ForeignKey("drug_variants.id"), nullable=True
    )
    category: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    claim_hash: Mapped[str] = mapped_column(Text, nullable=False)
    locator: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    claim_type: Mapped[str] = mapped_column(Text, nullable=False, default="source_raw")
    derived_from_claim_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_claims.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )

    # NULLS NOT DISTINCT semantics via COALESCE expression index
    __table_args__ = (
        Index(
            "uq_source_claim",
            "source_document_id",
            "drug_id",
            sqltext("COALESCE(drug_variant_id, 0)"),
            "category",
            "claim_hash",
            unique=True,
        ),
        Index("ix_claim_drug", "drug_id"),
        Index("ix_claim_category", "category"),
    )


# ---------------------------------------------------------------------------
# Kaplan narrative chunks
# ---------------------------------------------------------------------------


class DocumentSection(Base):
    """A node in the Kaplan section tree."""

    __tablename__ = "document_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id"), nullable=False
    )
    section_number: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    parent_section_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_sections.id"), nullable=True
    )
    physical_page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    physical_page_end: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("ix_section_doc", "source_document_id"),)


class DocumentChunk(Base):
    """A searchable Kaplan passage (parent-child structure)."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id"), nullable=False
    )
    section_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_sections.id"), nullable=True
    )
    parent_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_chunks.id"), nullable=True
    )
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False, default="text")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    table_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    entity_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    physical_pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # search_vector is a tsvector column for PostgreSQL full-text search
    # It's populated via a trigger or explicit UPDATE using to_tsvector()
    # We don't map it as a Python-side column; it's managed at the SQL level.

    __table_args__ = (
        Index("ix_chunk_doc", "source_document_id"),
        Index("ix_chunk_section", "section_id"),
        Index("ix_chunk_parent", "parent_chunk_id"),
    )


class DocumentEmbedding(Base):
    """Embedding vector for a chunk, stored separately.

    Model name and dimensions are explicit so changing embedding models
    doesn't require rewriting chunks.
    """

    __tablename__ = "document_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("document_chunks.id"), nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # embedding is a pgvector column — added via raw SQL in init_db()
    # because pgvector type needs special registration
    embedded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("chunk_id", "model", name="uq_embedding_chunk_model"),)


class ParseArtifact(Base):
    """A failed/unparsed page — NOT usable for answering.

    Stored separately from source_claims so garbled pages never enter
    production retrieval. Surfaced in the review queue.
    """

    __tablename__ = "parse_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(Text, nullable=False)
    physical_pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    usable_for_answering: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    warning: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Conversations, answers, feedback
# ---------------------------------------------------------------------------


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinician_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )


class AnswerTrace(Base):
    """Complete trace of one answer."""

    __tablename__ = "answer_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_understanding: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    evidence_package: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    answer: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    llm_model: Mapped[str] = mapped_column(Text, nullable=False)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )


class DoctorFeedback(Base):
    """Clinician feedback on an answer."""

    __tablename__ = "doctor_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_trace_id: Mapped[int] = mapped_column(
        ForeignKey("answer_traces.id"), nullable=False
    )
    feedback_type: Mapped[str] = mapped_column(Text, nullable=False)  # positive/negative/correction
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinician_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationCase(Base):
    """One test case for offline evaluation.

    V2 schema: uses stable acceptable_evidence (claim_hash/locator based, not
    DB IDs) and required/forbidden points instead of reference paragraphs.
    review_status tracks clinician sign-off.
    """
    __tablename__ = "evaluation_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)  # golden_v1, abstention_v1, regression
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    required_answer_points: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    forbidden_claims: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    acceptable_evidence: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(Text, nullable=False, default="needs_sasson_approval")
    reviewed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    __table_args__ = (Index("ix_eval_case_dataset", "dataset"),)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)
    results: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ClinicianReview(Base):
    """A psychiatrist's review of an answer trace.

    Used to compute the 'psychiatrist-approved >=90%' and 'critical clinical
    errors = 0' release gates. This is separate from doctor_feedback, which
    is the online failure-discovery mechanism.
    """
    __tablename__ = "clinician_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_trace_id: Mapped[int] = mapped_column(
        ForeignKey("answer_traces.id"), nullable=False
    )
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    # correct / correct_with_minor_issue / clinically_significant_error / unsafe_critical_error
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=sqltext("now()"), nullable=False
    )

    __table_args__ = (Index("ix_clinician_review_trace", "answer_trace_id"),)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


def init_db(drop_first: bool = False) -> None:
    """Create all tables and pgvector extension.

    Args:
        drop_first: If True, drop all tables first (destructive — dev only).
    """
    from .connection import get_engine

    engine = get_engine()

    if drop_first:
        Base.metadata.drop_all(engine)

    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(sqltext("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)

    # Add the embedding column with pgvector type (can't be done in ORM easily)
    from sqlalchemy import inspect

    insp = inspect(engine)
    if "document_embeddings" in insp.get_table_names():
        columns = [c["name"] for c in insp.get_columns("document_embeddings")]
        if "embedding" not in columns:
            with engine.connect() as conn:
                conn.execute(
                    sqltext(
                        "ALTER TABLE document_embeddings "
                        "ADD COLUMN embedding vector(1536)"
                    )
                )
                conn.commit()

    # Add search_vector tsvector column to document_chunks
    if "document_chunks" in insp.get_table_names():
        columns = [c["name"] for c in insp.get_columns("document_chunks")]
        if "search_vector" not in columns:
            with engine.connect() as conn:
                conn.execute(
                    sqltext("ALTER TABLE document_chunks ADD COLUMN search_vector tsvector")
                )
                # Index for full-text search
                conn.execute(
                    sqltext(
                        "CREATE INDEX ix_chunk_search_vector "
                        "ON document_chunks USING gin(search_vector)"
                    )
                )
                conn.commit()
