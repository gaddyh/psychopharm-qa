"""Streamlit app — psychopharm-qa main entry point."""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Psychopharm QA",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — latest golden run metrics
# ---------------------------------------------------------------------------
GOLDEN_RUN_PATH = Path(__file__).parent / "evals" / "results" / "latest_golden_run.json"


def _load_golden_run():
    if not GOLDEN_RUN_PATH.exists():
        return None
    try:
        with open(GOLDEN_RUN_PATH) as f:
            return json.load(f)
    except Exception:
        return None


_gr = _load_golden_run()

with st.sidebar:
    st.markdown("## 📊 Golden Run")
    if _gr is None:
        st.caption("No golden run yet. Run `python scripts/run_evals.py` to generate metrics.")
    else:
        s = _gr.get("summary", {})
        total = _gr.get("total_cases", 0)
        avg_recall = s.get("avg_recall_at_k", 0)
        full_recall = s.get("full_recall_cases", 0)
        cross_total = s.get("cross_source_cases", 0)
        cross_pass = s.get("cross_source_pass", 0)
        pts_total = s.get("total_required_points", 0)
        pts_covered = s.get("total_required_points_covered", 0)

        # Top-line metric
        recall_pct = avg_recall * 100 if avg_recall <= 1 else avg_recall
        st.metric("Recall@K", f"{recall_pct:.0f}%")
        st.metric("Cases passing", f"{full_recall}/{total}")

        # Compact summary
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Gold cases", total)
            st.metric("Cross-source", f"{cross_pass}/{cross_total}")
        with col_b:
            st.metric("Points covered", f"{pts_covered}/{pts_total}")
            st.metric("Review status", "Sasson pending", help="All gold cases require psychiatrist approval before release.")

        # Coverage by question type
        cov = _gr.get("coverage", {})
        by_type = cov.get("by_question_type", {})
        if by_type:
            st.markdown("---")
            st.markdown("**Coverage by type**")
            type_lines = []
            for qt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
                type_lines.append(f"`{qt}`: {cnt}")
            st.markdown(" · ".join(type_lines))

        # Source evidence totals
        by_src = cov.get("by_source", {})
        if by_src:
            st.markdown("---")
            st.markdown("**Evidence retrieved**")
            src_lines = []
            for src_name in ("nbn", "stahl", "kaplan"):
                val = by_src.get(src_name, 0)
                src_lines.append(f"{src_name.upper()}: {val}")
            st.markdown(" · ".join(src_lines))

        # Timestamp
        ts = _gr.get("timestamp", "")
        if ts:
            st.caption(f"Run: {ts[:19].replace('T', ' ')} UTC")

    st.markdown("---")
    st.markdown("### Trust Pipeline")
    st.markdown(
        """
1. **Source fidelity** — Stahl, NbN, Kaplan extracted
2. **Retrieval** — required evidence retrieved per question
3. **Grounding** — every claim cited to evidence
4. **Clinical correctness** — pending psychiatrist review
        """
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Psychopharm QA")
st.markdown("""
**Evidence-grounded psychopharmacology问答 for clinicians.**

This system answers questions about psychiatric drugs using three knowledge sources:
- **NbN** — pharmacological target, mode of action, dose variants
- **Stahl** — drug monographs (indications, mechanisms, dosing, side effects)
- **Kaplan Ch.33** — deeper clinical context and treatment rationale

👉 Go to **1_Chat** in the sidebar to ask a question.
👉 Go to **2_Feedback_Review** to review negative/corrected answers.
👉 Go to **3_Evaluation** to view offline evaluation results.
""")

st.divider()
st.markdown("### System Status")

try:
    from sqlalchemy import text
    from psych_qa.db.connection import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        # Check source documents
        docs = conn.execute(
            text("SELECT source_type, version_label, is_active FROM source_documents ORDER BY id")
        ).fetchall()
        if docs:
            st.success("Database connected")
            for d in docs:
                status = "✅ Active" if d[2] else "⬜ Inactive"
                st.markdown(f"- **{d[0]}**: {status} — {d[1]}")
        else:
            st.warning("Database connected but no source documents ingested yet.")
except Exception as e:
    st.error(f"Database connection failed: {e}")
