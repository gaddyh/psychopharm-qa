"""Regression-case promotion — convert clinician feedback into regression cases.

When a clinician marks an answer as incorrect or provides a correction,
that failure becomes a permanent regression case in evals/datasets/regression.jsonl.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ..config import get_settings
from ..db.connection import get_engine
from .loader import get_dataset_path
from .schemas import RegressionCase, ReviewStatus

logger = logging.getLogger(__name__)


def promote_feedback_to_regression(
    feedback_id: int,
    reviewer_notes: str = "",
    reviewed_by: str | None = None,
) -> RegressionCase | None:
    """Promote a clinician feedback record to a regression case.

    Args:
        feedback_id: The ID of the DoctorFeedback record.
        reviewer_notes: Optional notes from the reviewer.
        reviewed_by: Optional reviewer identifier.

    Returns:
        A RegressionCase if promotion succeeded, None if feedback not found.
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Get feedback + trace
        row = conn.execute(
            text("""
                SELECT df.feedback_type, df.reason, df.correction,
                       at.question, at.answer, at.evidence_package
                FROM doctor_feedback df
                JOIN answer_traces at ON df.answer_trace_id = at.id
                WHERE df.id = :fid
            """),
            {"fid": feedback_id},
        ).fetchone()

    if not row:
        logger.warning(f"Feedback {feedback_id} not found")
        return None

    feedback_type, reason, correction, question, answer_json, evidence_json = row

    # Parse the answer to extract claims for forbidden/required points
    answer = answer_json if isinstance(answer_json, dict) else json.loads(answer_json or "{}")
    evidence = evidence_json if isinstance(evidence_json, dict) else json.loads(evidence_json or "{}")

    # Build regression case from the feedback
    forbidden_claims: list[str] = []
    required_answer_points = []

    if feedback_type == "negative":
        # The claims that were wrong become forbidden
        for claim in answer.get("claims", []):
            forbidden_claims.append(claim.get("text", ""))

    if feedback_type == "correction" and correction:
        # The correction text becomes a required answer point
        from .schemas import RequiredAnswerPoint
        required_answer_points.append(RequiredAnswerPoint(
            text=correction,
            acceptable_evidence=[],  # To be filled by clinician review
        ))

    # Combine reason + reviewer notes
    notes = reviewer_notes
    if reason:
        notes = f"Original feedback reason: {reason}\n{notes}" if notes else f"Original feedback reason: {reason}"

    case = RegressionCase(
        question=question,
        expected_status="answered",  # Default; clinician can change
        required_answer_points=required_answer_points,
        forbidden_claims=[c for c in forbidden_claims if c],
        acceptable_evidence=[],
        reviewer_notes=notes,
        review_status=ReviewStatus.NEEDS_SASSON_APPROVAL,
        reviewed_by=reviewed_by,
        reviewed_at=datetime.now(timezone.utc) if reviewed_by else None,
        source_trace_id=feedback_id,
        metadata={
            "source_feedback_id": feedback_id,
            "feedback_type": feedback_type,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Append to regression dataset
    _append_regression_case(case)
    return case


def _append_regression_case(case: RegressionCase) -> Path:
    """Append a regression case to the regression.jsonl file."""
    path = get_dataset_path("regression")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSONL
    case_dict = case.model_dump(mode="json")

    with open(path, "a") as f:
        f.write(json.dumps(case_dict, default=str) + "\n")

    logger.info(f"Appended regression case to {path}")
    return path


def list_promotable_feedback() -> list[dict[str, Any]]:
    """List feedback records that can be promoted to regression cases."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT df.id, df.feedback_type, df.reason, df.correction,
                       df.answer_trace_id, at.question, df.created_at
                FROM doctor_feedback df
                JOIN answer_traces at ON df.answer_trace_id = at.id
                WHERE df.feedback_type IN ('negative', 'correction')
                ORDER BY df.id DESC
            """)
        ).fetchall()

    return [
        {
            "id": r[0],
            "feedback_type": r[1],
            "reason": r[2],
            "correction": r[3],
            "answer_trace_id": r[4],
            "question": r[5],
            "created_at": str(r[6]) if r[6] else None,
        }
        for r in rows
    ]
