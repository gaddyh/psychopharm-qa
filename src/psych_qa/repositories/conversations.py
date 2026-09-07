"""Conversations and answer traces repository."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text as sqltext

from ..db.connection import get_session


def create_conversation(title: str | None = None, clinician_id: str | None = None) -> int:
    session = get_session()
    try:
        row = session.execute(
            sqltext(
                "INSERT INTO conversations (title, clinician_id) VALUES (:t, :c) RETURNING id"
            ),
            {"t": title, "c": clinician_id},
        ).fetchone()
        session.commit()
        return row[0]
    finally:
        session.close()


def save_answer_trace(
    conversation_id: int | None,
    question: str,
    question_understanding: dict,
    evidence_package: dict,
    answer: dict,
    llm_model: str,
    llm_latency_ms: int | None = None,
    metrics: dict | None = None,
    versions: dict | None = None,
) -> int:
    session = get_session()
    try:
        row = session.execute(
            sqltext("""
                INSERT INTO answer_traces
                    (conversation_id, question, question_understanding, evidence_package,
                     answer, metrics, versions, llm_model, llm_latency_ms)
                VALUES (:cid, :q, :qu, :ep, :a, :met, :ver, :m, :lat)
                RETURNING id
            """),
            {
                "cid": conversation_id,
                "q": question,
                "qu": json.dumps(question_understanding),
                "ep": json.dumps(evidence_package),
                "a": json.dumps(answer),
                "met": json.dumps(metrics) if metrics else None,
                "ver": json.dumps(versions) if versions else None,
                "m": llm_model,
                "lat": llm_latency_ms,
            },
        ).fetchone()
        session.commit()
        return row[0]
    finally:
        session.close()


def get_answer_trace(trace_id: int) -> dict[str, Any] | None:
    session = get_session()
    try:
        row = session.execute(
            sqltext("SELECT * FROM answer_traces WHERE id = :id"),
            {"id": trace_id},
        ).fetchone()
        if not row:
            return None
        cols = row._mapping.keys()
        return dict(zip(cols, row))
    finally:
        session.close()


def list_answer_traces(limit: int = 50) -> list[dict[str, Any]]:
    session = get_session()
    try:
        rows = session.execute(
            sqltext("SELECT id, question, created_at FROM answer_traces ORDER BY id DESC LIMIT :l"),
            {"l": limit},
        ).fetchall()
        return [{"id": r[0], "question": r[1], "created_at": r[2]} for r in rows]
    finally:
        session.close()
