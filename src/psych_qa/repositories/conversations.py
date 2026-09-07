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
    turn_number: int | None = None,
    resolved_question: str | None = None,
) -> int:
    session = get_session()
    try:
        row = session.execute(
            sqltext("""
                INSERT INTO answer_traces
                    (conversation_id, turn_number, question, resolved_question,
                     question_understanding, evidence_package,
                     answer, metrics, versions, llm_model, llm_latency_ms)
                VALUES (:cid, :tn, :q, :rq, :qu, :ep, :a, :met, :ver, :m, :lat)
                RETURNING id
            """),
            {
                "cid": conversation_id,
                "tn": turn_number,
                "q": question,
                "rq": resolved_question,
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


def list_traces(conversation_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    """List all traces in a conversation, ordered by turn_number (oldest first)."""
    session = get_session()
    try:
        sql = (
            "SELECT id, turn_number, question, resolved_question, "
            "question_understanding, answer, metrics, created_at "
            "FROM answer_traces "
            "WHERE conversation_id = :cid "
            "ORDER BY COALESCE(turn_number, id) ASC"
        )
        params: dict[str, Any] = {"cid": conversation_id}
        if limit:
            sql += " LIMIT :l"
            params["l"] = limit
        rows = session.execute(sqltext(sql), params).fetchall()
        results = []
        for r in rows:
            cols = list(r._mapping.keys())
            d = dict(zip(cols, r))
            # Parse JSON fields
            for k in ("question_understanding", "answer", "metrics"):
                if d.get(k) and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            results.append(d)
        return results
    finally:
        session.close()


def get_recent_context(conversation_id: int, max_turns: int = 3) -> list[dict[str, Any]]:
    """Get recent conversation turns for context resolution.

    Returns the last max_turns turns (most recent last), each containing:
    - question, resolved_question, understanding, answer_summary, drug_names
    """
    session = get_session()
    try:
        rows = session.execute(
            sqltext(
                "SELECT id, turn_number, question, resolved_question, "
                "question_understanding, answer "
                "FROM answer_traces "
                "WHERE conversation_id = :cid "
                "ORDER BY COALESCE(turn_number, id) DESC "
                "LIMIT :l"
            ),
            {"cid": conversation_id, "l": max_turns},
        ).fetchall()
        # Reverse to chronological order (oldest first)
        turns = []
        for r in reversed(rows):
            understanding = r[4] if isinstance(r[4], dict) else json.loads(r[4]) if r[4] else {}
            answer = r[5] if isinstance(r[5], dict) else json.loads(r[5]) if r[5] else {}
            # Build a brief answer summary from direct_answer or first claim
            summary = answer.get("direct_answer", "")
            if not summary and answer.get("claims"):
                summary = answer["claims"][0].get("text", "")
            turns.append({
                "question": r[2],
                "resolved_question": r[3] or r[2],
                "understanding": understanding,
                "answer_summary": summary[:300],
                "drug_names": understanding.get("drug_names", []),
            })
        return turns
    finally:
        session.close()


def get_next_turn_number(conversation_id: int) -> int:
    """Get the next turn_number for a conversation (MAX + 1, or 1 if no turns)."""
    session = get_session()
    try:
        row = session.execute(
            sqltext(
                "SELECT COALESCE(MAX(turn_number), 0) + 1 "
                "FROM answer_traces WHERE conversation_id = :cid"
            ),
            {"cid": conversation_id},
        ).fetchone()
        return row[0]
    finally:
        session.close()
