"""Evaluation page — aggregate metrics across all answer traces."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from psych_qa.answering.metrics import compute_aggregate_metrics
from psych_qa.db.connection import get_engine

st.set_page_config(page_title="Evaluation", page_icon="📊")
st.title("📊 Evaluation")
st.markdown("Aggregate metrics across all answer traces.")

# ---------------------------------------------------------------------------
# Golden run — retrieval validation against gold dataset
# ---------------------------------------------------------------------------
GOLDEN_RUN_PATH = Path(__file__).resolve().parent.parent / "evals" / "results" / "latest_golden_run.json"


def _load_golden_run():
    if not GOLDEN_RUN_PATH.exists():
        return None
    try:
        with open(GOLDEN_RUN_PATH) as f:
            return json.load(f)
    except Exception:
        return None


_gr = _load_golden_run()
if _gr is not None:
    st.markdown("---")
    st.markdown("### 🏆 Golden Run — Retrieval Validation")

    s = _gr.get("summary", {})
    total = _gr.get("total_cases", 0)
    avg_recall = s.get("avg_recall_at_k", 0)
    full_recall = s.get("full_recall_cases", 0)
    cross_total = s.get("cross_source_cases", 0)
    cross_pass = s.get("cross_source_pass", 0)
    pts_total = s.get("total_required_points", 0)
    pts_covered = s.get("total_required_points_covered", 0)

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        st.metric("Gold cases", total)
        st.metric("Cross-source pass", f"{cross_pass}/{cross_total}")
    with gc2:
        recall_pct = avg_recall * 100 if avg_recall <= 1 else avg_recall
        st.metric("Avg Recall@K", f"{recall_pct:.0f}%")
        st.metric("100% recall cases", f"{full_recall}/{total}")
    with gc3:
        st.metric("Required points", f"{pts_covered}/{pts_total}")
        st.metric("Review status", "Sasson pending")
    with gc4:
        cov = _gr.get("coverage", {}).get("by_source", {})
        st.metric("NbN evidence", cov.get("nbn", 0))
        st.metric("Stahl evidence", cov.get("stahl", 0))

    # Per-case table
    st.markdown("#### Per-Case Results")
    gr_rows = []
    for r in _gr.get("results", []):
        ev = r.get("evidence_retrieved", {})
        src_req = r.get("sources_required", [])
        gr_rows.append({
            "#": r["case_id"],
            "Question": r["question"][:55],
            "Type": r["question_type"],
            "Recall@K": f"{r['recall_at_k']:.0%}",
            "Points": f"{r['required_points_covered']}/{r['required_points_total']}",
            "Forb": r["forbidden_claims_count"],
            "NbN": ev.get("nbn", 0),
            "Stahl": ev.get("stahl", 0),
            "Kaplan": ev.get("kaplan", 0),
            "3-SRC": "✅" if src_req else "",
        })
    gr_df = pd.DataFrame(gr_rows)
    st.dataframe(gr_df, use_container_width=True, hide_index=True)

    # Coverage by question type
    by_type = _gr.get("coverage", {}).get("by_question_type", {})
    if by_type:
        st.markdown("#### Coverage by Question Type")
        st.bar_chart(pd.DataFrame(
            {"count": list(by_type.values())},
            index=list(by_type.keys()),
        ))

    # Detailed case inspection
    st.markdown("#### Detailed Case Inspection")
    gr_results = _gr.get("results", [])
    if gr_results:
        case_ids = [r["case_id"] for r in gr_results]
        sel_id = st.selectbox(
            "Select gold case:",
            case_ids,
            format_func=lambda x: f"#{x}: {next(r['question'] for r in gr_results if r['case_id'] == x)[:50]}",
            key="gr_case_sel",
        )
        sel = next(r for r in gr_results if r["case_id"] == sel_id)
        st.markdown(f"**Question:** {sel['question']}")
        st.markdown(f"**Type:** {sel['question_type']} · **Recall@K:** {sel['recall_at_k']:.0%} · **Sources required:** {sel.get('sources_required', [])}")
        st.markdown("**Per-point coverage:**")
        for pp in sel.get("per_point", []):
            icon = "✅" if pp["found"] else "❌"
            st.markdown(f"- {icon} {pp['point_text'][:90]}")
            if pp.get("matched_evidence_ids"):
                st.caption(f"  Matched: {', '.join(pp['matched_evidence_ids'][:5])}")

    ts = _gr.get("timestamp", "")
    if ts:
        st.caption(f"Golden run timestamp: {ts[:19].replace('T', ' ')} UTC")

st.markdown("---")

# Load all traces with metrics
try:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, question, answer, llm_model, llm_latency_ms, metrics, versions, created_at
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
    v = r[6] if isinstance(r[6], dict) else (json.loads(r[6]) if r[6] else {})
    traces.append({
        "id": r[0],
        "question": r[1],
        "status": m.get("status", answer.get("status", "unknown")),
        "llm_model": r[3],
        "llm_latency_ms": r[4],
        "metrics": m,
        "versions": v,
        "created_at": r[7],
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
        st.metric("Avg citation completeness", f"{agg.get('avg_citation_completeness', 0):.1%}")
    with sc3:
        st.metric("Avg total latency", f"{agg.get('total_latency_ms_mean', 0):.0f}ms")
        st.metric("Avg LLM latency", f"{agg.get('llm_latency_ms_mean', 0):.0f}ms")
    with sc4:
        st.metric("Avg sources used", f"{agg.get('sources_used_mean', 0):.1f}/3")
        st.metric("Avg evidence utilization", f"{agg.get('evidence_utilization_mean', 0):.1%}")

    if "avg_citation_entailment" in agg:
        st.metric("Avg citation entailment", f"{agg.get('avg_citation_entailment', 0):.1%}")

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
        "Claims": m.get("total_claims", 0),
        "Cmpl%": f"{m.get('citation_completeness', 0):.0%}",
        "Valid%": f"{m.get('citation_validity', 0):.0%}",
        "Crit": m.get("critical_claims", 0),
        "UnsupCrit": m.get("unsupported_critical_claims", 0),
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

if selected.get("versions"):
    st.markdown("#### Version Metadata")
    st.json(selected["versions"])
