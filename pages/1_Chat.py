"""Chat page — ask questions and get cited answers with feedback."""

import json

import streamlit as st

from psych_qa.answering.answer_service import answer_question, build_evidence_lookup
from psych_qa.answering.citations import resolve_citation
from psych_qa.feedback.service import save_feedback

st.set_page_config(page_title="Chat", page_icon="💬")
st.title("💬 Ask a Question")
st.markdown("Ask a psychopharmacology question. The system will search Stahl, NbN, and Kaplan Chapter 33.")

# Initialize session state
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Example questions
with st.expander("Example questions"):
    examples = [
        "What is the mechanism of action of amisulpride?",
        "What are the FDA-approved indications for amisulpride?",
        "What is the usual dosage range for amisulpride in schizophrenia?",
        "What are the notable side effects of amisulpride?",
        "What is the evidence for amisulpride's effectiveness as an antipsychotic, and how does its benzamide structure relate to its mechanism?",
        "How does amisulpride compare to other atypical antipsychotics in clinical trials?",
        "What did the EUFEST trial find about amisulpride compared to other antipsychotics in first-episode schizophrenia?",
        "What is the evidence from clinical trials comparing amisulpride to haloperidol?",
        "How does the NbN reclassification affect how amisulpride is categorized?",
        "What are the treatment guidelines for amisulpride in first-episode schizophrenia?",
    ]
    for q in examples:
        if st.button(q, key=f"example_{q}"):
            st.session_state.question_input = q
            st.rerun()

# Question input
question = st.text_area(
    "Your question:",
    value=st.session_state.get("question_input", ""),
    height=100,
    placeholder="e.g. What is the mechanism of action of amisulpride?",
)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Searching evidence sources and generating answer..."):
            try:
                result = answer_question(question.strip())
                st.session_state.current_answer = result
                st.session_state.feedback_submitted = False
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
with col2:
    if st.button("Clear"):
        st.session_state.current_answer = None
        st.session_state.feedback_submitted = False
        st.rerun()

# Display answer
result = st.session_state.current_answer
if result:
    st.divider()
    answer = result["answer"]

    # Status badge
    status = answer.get("status", "answered")
    if status == "answered":
        st.success(f"✅ Answered (trace #{result['trace_id']})")
    elif status == "abstained":
        st.warning(f"⚠️ Abstained (trace #{result['trace_id']})")
    elif status == "needs_clarification":
        st.info(f"❓ Needs clarification (trace #{result['trace_id']})")

    # Direct answer (assembled from claims)
    st.markdown("### Answer")
    st.markdown(answer.get("direct_answer", "_(no answer)_"))

    # Explanation (assembled from claims)
    if answer.get("explanation"):
        st.markdown("### Explanation")
        st.markdown(answer["explanation"])

    # Uncertainties
    uncertainties = answer.get("uncertainties", [])
    if uncertainties:
        st.markdown("### ⚠️ Uncertainties / Disagreements")
        for u in uncertainties:
            st.warning(u)

    # Clarification question
    if answer.get("clarification_question"):
        st.markdown("### Clarification needed")
        st.info(answer["clarification_question"])

    # Claims detail
    claims = answer.get("claims", [])
    if claims:
        st.markdown("### Claims")
        evidence_lookup = build_evidence_lookup(result["evidence_package"])

        for i, claim in enumerate(claims):
            role = claim.get("claim_role", "direct")
            critical = claim.get("critical", False)
            critical_badge = " 🔴 **CRITICAL**" if critical else ""
            role_badge = {
                "direct": "🔵 Direct",
                "explanatory": "🟢 Explanatory",
                "caveat": "🟡 Caveat",
                "comparison": "🟣 Comparison",
            }.get(role, role)

            st.markdown(f"**{i+1}.** [{role_badge}{critical_badge}] {claim.get('text', '')}")
            for eid in claim.get("evidence_ids", []):
                resolved = resolve_citation(eid, evidence_lookup)
                if resolved:
                    page_info = f", p.{resolved['page']}" if resolved.get("page") else ""
                    cat_info = f" ({resolved['category']})" if resolved.get("category") else ""
                    st.markdown(
                        f"  - `{eid}` — **{resolved['source']}**{cat_info}{page_info}: {resolved['text'][:150]}..."
                    )
                else:
                    st.markdown(f"  - `{eid}` — *(unresolved)*")

    # Evidence summary
    with st.expander("📋 Evidence Package Details"):
        ep = result["evidence_package"]
        for drug_name, drug_data in ep.get("drugs", {}).items():
            nbn_count = len(drug_data.get("nbn_evidence", []))
            stahl_count = len(drug_data.get("stahl_evidence", []))
            st.markdown(f"**{drug_name}**: {nbn_count} NbN claims, {stahl_count} Stahl claims")

        kaplan_count = len(ep.get("kaplan_passages", []))
        st.markdown(f"**Kaplan passages**: {kaplan_count}")

        conflicts = ep.get("conflicts_or_uncertainties", [])
        if conflicts:
            st.markdown("**Conflicts:**")
            for c in conflicts:
                st.markdown(f"- {c}")

        understanding = result.get("understanding", {})
        st.markdown(f"**Question type**: {understanding.get('question_type', 'unknown')}")
        st.markdown(f"**Drugs detected**: {understanding.get('drug_names', [])}")
        premises = understanding.get("premises", [])
        if premises:
            st.markdown(f"**Premises to verify**: {premises}")

    # Metrics
    with st.expander("📊 Run Metrics"):
        m = result.get("metrics", {})
        if m:
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Total latency", f"{m.get('total_latency_ms', 0)}ms")
                st.metric("LLM latency", f"{m.get('llm_latency_ms', 0)}ms")
                st.metric("Retrieval latency", f"{m.get('retrieval_latency_ms', 0)}ms")
            with mc2:
                st.metric("Evidence provided", m.get("total_evidence_provided", 0))
                st.metric("Evidence cited", m.get("unique_evidence_ids_cited", 0))
                st.metric("Utilization", f"{m.get('evidence_utilization', 0):.1%}")
            with mc3:
                st.metric("Citation completeness", f"{m.get('citation_completeness', 0):.1%}")
                st.metric("Citation validity", f"{m.get('citation_validity', 0):.1%}")
                st.metric("Invalid citations", m.get("invalid_citation_count", 0))
            with mc4:
                st.metric("Sources used", f"{m.get('sources_used', 0)}/3")
                st.metric("Critical claims", m.get("critical_claims", 0))
                st.metric("Unsupported critical", m.get("unsupported_critical_claims", 0))

            if m.get("citation_entailment") is not None:
                st.metric("Citation entailment", f"{m.get('citation_entailment', 0):.1%}")

            st.markdown("**Source breakdown:**")
            st.markdown(f"- NbN: {m.get('nbn_cited', 0)} cited / {m.get('nbn_evidence_provided', 0)} provided")
            st.markdown(f"- Stahl: {m.get('stahl_cited', 0)} cited / {m.get('stahl_evidence_provided', 0)} provided")
            st.markdown(f"- Kaplan: {m.get('kaplan_cited', 0)} cited / {m.get('kaplan_evidence_provided', 0)} provided")

    # Version metadata
    with st.expander("🔢 Version Metadata"):
        versions = result.get("versions", {})
        st.json(versions)

    # Feedback section
    st.divider()
    st.markdown("### Feedback")

    if st.session_state.feedback_submitted:
        st.success("Thank you! Feedback saved.")
    else:
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        with fb_col1:
            thumbs_up = st.button("👍 Correct", key="fb_positive")
        with fb_col2:
            thumbs_down = st.button("👎 Incorrect", key="fb_negative")
        with fb_col3:
            correction = st.button("✏️ Correction", key="fb_correction")

        if thumbs_up:
            save_feedback(result["trace_id"], "positive")
            st.session_state.feedback_submitted = True
            st.rerun()

        # Use st.form for negative feedback to prevent disappearing form
        if thumbs_down:
            st.session_state.fb_mode = "negative"

        if st.session_state.get("fb_mode") == "negative":
            with st.form("fb_negative_form"):
                reason = st.text_input("What's wrong?", key="fb_neg_reason")
                submitted = st.form_submit_button("Submit feedback")
                if submitted:
                    save_feedback(result["trace_id"], "negative", reason=reason)
                    st.session_state.feedback_submitted = True
                    st.session_state.fb_mode = None
                    st.rerun()

        # Use st.form for correction feedback too
        if correction:
            st.session_state.fb_mode = "correction"

        if st.session_state.get("fb_mode") == "correction":
            with st.form("fb_correction_form"):
                correction_text = st.text_area("Corrected answer:", key="fb_corr_text", height=100)
                submitted = st.form_submit_button("Submit correction")
                if submitted:
                    save_feedback(result["trace_id"], "correction", correction=correction_text)
                    st.session_state.feedback_submitted = True
                    st.session_state.fb_mode = None
                    st.rerun()
