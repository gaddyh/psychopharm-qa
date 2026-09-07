"""Streamlit app — psychopharm-qa main entry point."""

import streamlit as st

st.set_page_config(
    page_title="Psychopharm QA",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
