"""Enumerations for the psychopharm-qa domain."""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    """Which knowledge source a piece of data came from."""

    NBN = "nbn"
    STAHL = "stahl"
    KAPLAN = "kaplan"


class AliasType(str, Enum):
    """Type of drug alias.

    Brands and spelling variants are aliases.
    Classifications (e.g. "antipsychotic") are NOT aliases — they are source_claims.
    """

    BRAND = "brand"
    SPELLING_VARIANT = "spelling_variant"


class ClaimType(str, Enum):
    """Whether a claim is raw source text or derived by a parser."""

    SOURCE_RAW = "source_raw"
    DERIVED_STRUCTURED = "derived_structured"


class ChunkType(str, Enum):
    """Type of Kaplan document chunk."""

    TEXT = "text"
    TABLE = "table"
    FIGURE_REF = "figure_ref"


class IngestionStatus(str, Enum):
    """Status of an ingestion run."""

    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class FeedbackType(str, Enum):
    """Type of clinician feedback on an answer."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"


class SupportStatus(str, Enum):
    """Whether evidence supports, contradicts, or is neutral toward a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class AnswerStatus(str, Enum):
    """Outcome of the answering pipeline."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NEEDS_CLARIFICATION = "needs_clarification"
