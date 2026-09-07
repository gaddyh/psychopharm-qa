"""Feedback Review page — review negative/corrected answers."""

import json

import streamlit as st

from psych_qa.feedback.review_queue import get_review_items, get_trace_with_feedback

st.set_page_config(page_title="Feedback Review", page_icon="📋")
st.title("📋 Feedback Review")
st.markdown("Review answers that received negative feedback or corrections. These can later become regression test cases.")

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

                answer = trace.get("answer", {}) if isinstance(trace.get("answer"), dict) else json.loads(trace.get("answer", "{}"))
                st.markdown(answer.get("direct_answer", "_(no answer)_"))

                if answer.get("explanation"):
                    st.markdown(f"**Explanation**: {answer['explanation']}")

                if answer.get("uncertainty"):
                    st.warning(f"**Uncertainty**: {answer['uncertainty']}")

                citations = answer.get("citations", [])
                if citations:
                    st.markdown("**Citations:**")
                    for cite in citations:
                        st.markdown(f"- {cite.get('text', '')} ({', '.join(cite.get('evidence_ids', []))})")

                # Evidence package
                ep = trace.get("evidence_package", {})
                if isinstance(ep, str):
                    ep = json.loads(ep)
                with st.expander("Evidence Package"):
                    for drug_name, drug_data in ep.get("drugs", {}):
                        nbn = drug_data.get("nbn_evidence", [])
                        stahl = drug_data.get("stahl_evidence", [])
                        st.markdown(f"**{drug_name}**: {len(nbn)} NbN, {len(stahl)} Stahl")
                    st.markdown(f"**Kaplan**: {len(ep.get('kaplan_passages', []))} passages")
