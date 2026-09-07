"""Feedback service — save and retrieve clinician feedback."""

from __future__ import annotations

from ..repositories import feedback as feedback_repo


def save_feedback(
    answer_trace_id: int,
    feedback_type: str,
    reason: str | None = None,
    correction: str | None = None,
    clinician_id: str | None = None,
) -> int:
    """Save doctor feedback for an answer.

    feedback_type: 'positive', 'negative', or 'correction'
    """
    return feedback_repo.save_feedback(
        answer_trace_id=answer_trace_id,
        feedback_type=feedback_type,
        reason=reason,
        correction=correction,
        clinician_id=clinician_id,
    )
