"""YAML gold set loader with Pydantic validation.

Loads evals/kaplan_ocd_gold_set_v0.1.yaml into typed objects.
Fail-fast: duplicate evidence IDs, missing references, empty critical,
negative pages, evidence in both critical+supporting, duplicate case IDs
all raise immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CorpusInfo(BaseModel):
    file_name: str
    chapter: int
    page_numbering: str
    scope_note: str = ""


class RetrievalPolicy(BaseModel):
    primary_metric: str
    report_metrics: list[str] = Field(default_factory=list)
    match_unit: str = "evidence_id"


class PassPolicy(BaseModel):
    retrieval: str = ""
    answer: str = ""


class EvaluationPolicy(BaseModel):
    release_gate_eligible: bool = False
    reason: str = ""
    retrieval: RetrievalPolicy
    pass_policy: PassPolicy = Field(default_factory=PassPolicy)


class GoldSetMeta(BaseModel):
    id: str
    title: str
    language: str = "en"
    status: str = "draft"
    needs_approval: bool = True
    approver: str = ""
    purpose: str = ""
    corpus: CorpusInfo
    evaluation_policy: EvaluationPolicy


class EvidenceItem(BaseModel):
    evidence_id: str
    pdf_pages: list[int]
    section: str = ""
    evidence_type: str = ""
    certainty: str | None = None
    anchor_text: list[str] = Field(default_factory=list)
    excerpt: str = ""

    @field_validator("pdf_pages")
    @classmethod
    def pages_must_be_non_negative(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("pdf_pages must not be empty")
        for p in v:
            if p < 0:
                raise ValueError(f"pdf_pages contains negative value: {p}")
        return v

    @field_validator("anchor_text")
    @classmethod
    def anchors_required(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("anchor_text must not be empty — matcher requires page+anchor")
        return v


class RequiredClaim(BaseModel):
    claim_id: str
    critical: bool = False
    text: str
    supported_by: list[str] = Field(default_factory=list)


class GoldCase(BaseModel):
    case_id: str
    route: str
    question: str
    answerability: str = "partial"
    answerability_reason: str = ""
    critical_evidence: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    required_claims: list[RequiredClaim] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    ideal_answer_outline: list[str] = Field(default_factory=list)

    @field_validator("critical_evidence")
    @classmethod
    def critical_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("critical_evidence must not be empty")
        return v

    @model_validator(mode="after")
    def no_evidence_overlap(self) -> GoldCase:
        crit_set = set(self.critical_evidence)
        supp_set = set(self.supporting_evidence)
        overlap = crit_set & supp_set
        if overlap:
            raise ValueError(
                f"Evidence appears in both critical and supporting: {overlap}"
            )
        return self


class GoldSet(BaseModel):
    """Complete loaded gold set with cross-references validated."""

    meta: GoldSetMeta
    evidence_catalog: list[EvidenceItem]
    cases: list[GoldCase]

    @model_validator(mode="after")
    def validate_cross_references(self) -> GoldSet:
        # Build evidence catalog lookup
        evidence_ids: dict[str, EvidenceItem] = {}
        for item in self.evidence_catalog:
            if item.evidence_id in evidence_ids:
                raise ValueError(
                    f"Duplicate evidence_id: {item.evidence_id}"
                )
            evidence_ids[item.evidence_id] = item

        # Check case IDs are unique
        case_ids: set[str] = set()
        for case in self.cases:
            if case.case_id in case_ids:
                raise ValueError(f"Duplicate case_id: {case.case_id}")
            case_ids.add(case.case_id)

        # Check all evidence references resolve
        for case in self.cases:
            for eid in case.critical_evidence:
                if eid not in evidence_ids:
                    raise ValueError(
                        f"{case.case_id}: critical_evidence references unknown ID: {eid}"
                    )
            for eid in case.supporting_evidence:
                if eid not in evidence_ids:
                    raise ValueError(
                        f"{case.case_id}: supporting_evidence references unknown ID: {eid}"
                    )
            for claim in case.required_claims:
                for eid in claim.supported_by:
                    if eid not in evidence_ids:
                        raise ValueError(
                            f"{case.case_id} / {claim.claim_id}: "
                            f"supported_by references unknown ID: {eid}"
                        )

        return self

    @property
    def evidence_by_id(self) -> dict[str, EvidenceItem]:
        return {item.evidence_id: item for item in self.evidence_catalog}

    def all_evidence_for_case(self, case: GoldCase) -> list[str]:
        """Return critical ∪ supporting evidence IDs for a case."""
        return list(case.critical_evidence) + list(case.supporting_evidence)


def load_gold_set(yaml_path: Path | str | None = None) -> GoldSet:
    """Load and validate a gold set YAML file.

    Args:
        yaml_path: Path to the YAML file. Defaults to the standard gold set.

    Returns:
        Validated GoldSet.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValidationError: If any validation fails.
    """
    if yaml_path is None:
        from ..config import get_settings
        settings = get_settings()
        yaml_path = settings.evals_dir / "kaplan_ocd_gold_set_v0.1.yaml"

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Gold set not found: {yaml_path}")

    with open(yaml_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    meta = GoldSetMeta(**raw["gold_set"])
    evidence_catalog = [EvidenceItem(**item) for item in raw["evidence_catalog"]]
    cases = [GoldCase(**case) for case in raw["cases"]]

    return GoldSet(
        meta=meta,
        evidence_catalog=evidence_catalog,
        cases=cases,
    )
