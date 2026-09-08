"""Feedback Review page — review negative/corrected answers and promote to regression cases."""

import json

import streamlit as st

from psych_qa.evaluation.regression import list_promotable_feedback, promote_feedback_to_regression
from psych_qa.feedback.review_queue import get_review_items, get_trace_with_feedback

st.set_page_config(page_title="Feedback Review", page_icon="📋")
st.title("📋 Feedback Review")
st.markdown("Review answers that received negative feedback or corrections. Promote failures to regression test cases.")

# --- Tabs ---
tab1, tab2 = st.tabs(["Review Queue", "Promote to Regression"])

# --- Tab 1: Review Queue ---
with tab1:
    items = get_review_items(limit=50)

    if not items:
        st.info("No negative feedback or corrections yet.")
    else:
        st.markdown(f"**{len(items)} items** in the review queue.")

        for item in items:
            with st.expander(f"#{item['id']} — {item['question'][:80]}... ({item['feedback_type']})"):
                st.markdown(f"**Feedback type**: {item['feedback_type']}")
                st.markdown(f"**Question**: {item['question']}")
                if item.get("reason"):
                    st.markdown(f"**Reason**: {item['reason']}")
                if item.get("correction"):
                    st.markdown(f"**Correction**: {item['correction']}")
                st.markdown(f"**Date**: {item['created_at']}")

                trace = get_trace_with_feedback(item["answer_trace_id"])
                if trace:
                    st.markdown("---")
                    st.markdown("**Original Answer:**")

                    answer = trace.get("answer", {})
                    if isinstance(answer, str):
                        answer = json.loads(answer)

                    # V2: claims-first format
                    claims = answer.get("claims", [])
                    if claims:
                        st.markdown(f"**Status**: {answer.get('status', 'unknown')}")
                        for i, claim in enumerate(claims):
                            role = claim.get("claim_role", "direct")
                            critical = claim.get("critical", False)
                            critical_badge = " 🔴" if critical else ""
                            st.markdown(f"{i+1}. [{role}{critical_badge}] {claim.get('text', '')}")
                            for eid in claim.get("evidence_ids", []):
                                st.markdown(f"   - `{eid}`")

                        if answer.get("uncertainties"):
                            st.markdown("**Uncertainties:**")
                            for u in answer["uncertainties"]:
                                st.warning(u)
                    else:
                        # Fallback for old-format traces
                        st.markdown(answer.get("direct_answer", "_(no answer)_"))
                        if answer.get("explanation"):
                            st.markdown(f"**Explanation**: {answer['explanation']}")
                        citations = answer.get("citations", [])
                        if citations:
                            st.markdown("**Citations:**")
                            for cite in citations:
                                st.markdown(f"- {cite.get('text', '')} ({', '.join(cite.get('evidence_ids', []))})")

                    # Evidence package — use .items() correctly
                    ep = trace.get("evidence_package", {})
                    if isinstance(ep, str):
                        ep = json.loads(ep)
                    with st.expander("Evidence Package"):
                        drugs = ep.get("drugs", {})
                        if isinstance(drugs, dict):
                            for drug_name, drug_data in drugs.items():
                                nbn = drug_data.get("nbn_evidence", [])
                                stahl = drug_data.get("stahl_evidence", [])
                                st.markdown(f"**{drug_name}**: {len(nbn)} NbN, {len(stahl)} Stahl")
                        else:
                            st.warning("Malformed drugs data in evidence package")
                        st.markdown(f"**Kaplan**: {len(ep.get('kaplan_passages', []))} passages")

# --- Tab 2: Promote to Regression ---
with tab2:
    st.markdown("### Promote Feedback to Regression Cases")
    st.markdown("Select a feedback record to convert it into a permanent regression test case.")

    promotable = list_promotable_feedback()

    if not promotable:
        st.info("No promotable feedback records found. Negative or correction feedback will appear here.")
    else:
        st.markdown(f"**{len(promotable)} promotable records**")

        for item in promotable:
            with st.container():
                col1, col2, col3 = st.columns([5, 3, 2])
                with col1:
                    st.markdown(f"**#{item['id']}** ({item['feedback_type']})")
                    st.markdown(f"Q: {item['question'][:80]}...")
                    if item.get("reason"):
                        st.markdown(f"Reason: {item['reason'][:80]}")
                    if item.get("correction"):
                        st.markdown(f"Correction: {item['correction'][:80]}")
                with col2:
                    reviewer = st.text_input(
                        "Reviewer", key=f"reviewer_{item['id']}", placeholder="e.g. Sasson"
                    )
                    notes = st.text_input(
                        "Notes", key=f"notes_{item['id']}", placeholder="Why this becomes a regression case"
                    )
                with col3:
                    if st.button("Promote", key=f"promote_{item['id']}"):
                        if not reviewer:
                            st.error("Reviewer name required.")
                        else:
                            try:
                                case = promote_feedback_to_regression(
                                    feedback_id=item["id"],
                                    reviewer_notes=notes,
                                    reviewed_by=reviewer,
                                )
                                if case:
                                    st.success(f"Promoted to regression case: {case.question[:60]}...")
                                else:
                                    st.error("Promotion failed — feedback not found.")
                            except Exception as e:
                                st.error(f"Error: {e}")

                st.divider()
