"""Review queue — list negative/corrected answers for clinician review."""

from __future__ import annotations

from typing import Any

from ..repositories import feedback as feedback_repo
from ..repositories import conversations as conv_repo


def get_review_items(limit: int = 50) -> list[dict[str, Any]]:
    """Get negative/correction feedback items for review."""
    return feedback_repo.get_negative_feedback(limit=limit)


def get_trace_with_feedback(trace_id: int) -> dict[str, Any] | None:
    """Get an answer trace with its feedback for detailed review."""
    trace = conv_repo.get_answer_trace(trace_id)
    if not trace:
        return None
    return trace
