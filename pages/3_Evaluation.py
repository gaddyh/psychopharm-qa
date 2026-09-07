"""Evaluation page — aggregate metrics across all answer traces."""

import json

import pandas as pd
import streamlit as st
from sqlalchemy import text

from psych_qa.answering.metrics import compute_aggregate_metrics
from psych_qa.db.connection import get_engine

st.set_page_config(page_title="Evaluation", page_icon="📊")
st.title("📊 Evaluation")
st.markdown("Aggregate metrics across all answer traces.")

# Load all traces with metrics
try:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, question, answer, llm_model, llm_latency_ms, metrics, created_at
                FROM answer_traces ORDER BY id DESC LIMIT 200
            """)
        ).fetchall()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if not rows:
    st.info("No answer traces yet. Ask some questions in the Chat page first.")
    st.stop()

# Parse metrics
traces = []
for r in rows:
    answer = r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {})
    m = r[5] if isinstance(r[5], dict) else (json.loads(r[5]) if r[5] else {})
    traces.append({
        "id": r[0],
        "question": r[1],
        "status": m.get("status", answer.get("status", "unknown")),
        "llm_model": r[3],
        "llm_latency_ms": r[4],
        "metrics": m,
        "created_at": r[6],
    })

st.markdown(f"**{len(traces)} traces** loaded.")

# --- Aggregate metrics ---
all_metrics = [t["metrics"] for t in traces if t["metrics"]]
if all_metrics:
    agg = compute_aggregate_metrics(all_metrics)

    st.markdown("### Summary")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Total runs", agg.get("total_runs", 0))
        st.metric("Abstention rate", f"{agg.get('abstention_rate', 0):.1%}")
    with sc2:
        st.metric("Avg citation validity", f"{agg.get('avg_citation_validity', 0):.1%}")
        st.metric("Avg citation coverage", f"{agg.get('citation_coverage_mean', 0):.1%}")
    with sc3:
        st.metric("Avg total latency", f"{agg.get('total_latency_ms_mean', 0):.0f}ms")
        st.metric("Avg LLM latency", f"{agg.get('llm_latency_ms_mean', 0):.0f}ms")
    with sc4:
        st.metric("Avg sources used", f"{agg.get('sources_used_mean', 0):.1f}/3")
        st.metric("Avg evidence utilization", f"{agg.get('evidence_utilization_mean', 0):.1%}")

    # Status distribution
    st.markdown("### Status Distribution")
    status_dist = agg.get("status_distribution", {})
    if status_dist:
        st.bar_chart(pd.DataFrame({
            "status": list(status_dist.keys()),
            "count": list(status_dist.values()),
        }).set_index("status"))

    # Source usage distribution
    st.markdown("### Source Usage Distribution")
    source_dist = agg.get("source_usage_distribution", {})
    if source_dist:
        col_labels = {0: "0 sources", 1: "1 source", 2: "2 sources", 3: "3 sources"}
        st.bar_chart(pd.DataFrame({
            "sources": [col_labels.get(k, f"{k} sources") for k in source_dist.keys()],
            "count": list(source_dist.values()),
        }).set_index("sources"))

# --- Per-trace table ---
st.markdown("### Per-Trace Metrics")
table_data = []
for t in traces:
    m = t["metrics"]
    table_data.append({
        "ID": t["id"],
        "Question": t["question"][:60],
        "Status": m.get("status", t["status"]),
        "Sources": f"{m.get('sources_used', 0)}/3",
        "NbN": m.get("nbn_cited", 0),
        "Stahl": m.get("stahl_cited", 0),
        "Kaplan": m.get("kaplan_cited", 0),
        "Cov%": f"{m.get('citation_coverage', 0):.0%}",
        "Valid%": f"{m.get('citation_validity', 0):.0%}",
        "Util%": f"{m.get('evidence_utilization', 0):.0%}",
        "Latency": f"{m.get('total_latency_ms', 0)}ms",
    })
df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# --- Detailed metrics for selected trace ---
st.markdown("### Detailed Trace Inspection")
trace_ids = [t["id"] for t in traces]
selected_id = st.selectbox("Select trace:", trace_ids, format_func=lambda x: f"#{x}: {next(t['question'] for t in traces if t['id'] == x)[:50]}")
selected = next(t for t in traces if t["id"] == selected_id)

if selected["metrics"]:
    m = selected["metrics"]
    st.json(m)
