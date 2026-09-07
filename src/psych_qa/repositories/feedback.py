"""Feedback repository."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session


def save_feedback(
    answer_trace_id: int,
    feedback_type: str,
    reason: str | None = None,
    correction: str | None = None,
    clinician_id: str | None = None,
) -> int:
    session = get_session()
    try:
        row = session.execute(
            sqltext("""
                INSERT INTO doctor_feedback
                    (answer_trace_id, feedback_type, reason, correction, clinician_id)
                VALUES (:aid, :ft, :r, :c, :cid)
                RETURNING id
            """),
            {"aid": answer_trace_id, "ft": feedback_type, "r": reason, "c": correction, "cid": clinician_id},
        ).fetchone()
        session.commit()
        return row[0]
    finally:
        session.close()


def get_negative_feedback(limit: int = 50) -> list[dict[str, Any]]:
    """Get negative/correction feedback for the review queue."""
    session = get_session()
    try:
        rows = session.execute(
            sqltext("""
                SELECT df.id, df.answer_trace_id, df.feedback_type, df.reason,
                       df.correction, df.created_at, at.question
                FROM doctor_feedback df
                JOIN answer_traces at ON at.id = df.answer_trace_id
                WHERE df.feedback_type IN ('negative', 'correction')
                ORDER BY df.created_at DESC LIMIT :l
            """),
            {"l": limit},
        ).fetchall()
        return [
            {
                "id": r[0],
                "answer_trace_id": r[1],
                "feedback_type": r[2],
                "reason": r[3],
                "correction": r[4],
                "created_at": r[5],
                "question": r[6],
            }
            for r in rows
        ]
    finally:
        session.close()
