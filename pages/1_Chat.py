"""Chat page — ask questions and get cited answers with feedback.

Supports multi-turn conversations: follow-up questions are resolved using
conversation context, but each turn retrieves fresh evidence and generates
a standalone answer.
"""

import json

import streamlit as st

from psych_qa.answering.answer_service import answer_question, build_evidence_lookup
from psych_qa.answering.citations import resolve_citation
from psych_qa.feedback.service import save_feedback
from psych_qa.repositories import conversations as conv_repo

st.set_page_config(page_title="Chat", page_icon="💬")
st.title("💬 Ask a Question")
st.markdown("Ask a psychopharmacology question. The system will search Stahl, NbN, and Kaplan Chapter 33.")

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}


def _load_thread():
    """Load all traces for the current conversation from the database."""
    cid = st.session_state.conversation_id
    if cid is None:
        return []
    return conv_repo.list_traces(cid)


def _render_turn(turn: dict, is_latest: bool = False):
    """Render a single conversation turn (question + answer + details)."""
    trace_id = turn["id"]
    question = turn["question"]
    resolved = turn.get("resolved_question")
    answer = turn.get("answer", {})
    if isinstance(answer, str):
        answer = json.loads(answer)
    understanding = turn.get("question_understanding", {})
    if isinstance(understanding, str):
        understanding = json.loads(understanding)

    # Question
    st.markdown(f"**🧑 Q:** {question}")
    if resolved and resolved != question:
        context_status = understanding.get("context_status", "resolved")
        st.caption(f"↳ Resolved: {resolved} ({context_status})")

    # Answer
    status = answer.get("status", "answered")
    status_emoji = {"answered": "✅", "abstained": "⚠️", "needs_clarification": "❓"}.get(status, "📝")
    st.markdown(f"**💊 A:** {status_emoji} (trace #{trace_id})")

    if answer.get("direct_answer"):
        st.markdown(answer["direct_answer"])

    if answer.get("explanation"):
        with st.expander("Explanation", expanded=False):
            st.markdown(answer["explanation"])

    uncertainties = answer.get("uncertainties", [])
    if uncertainties:
        for u in uncertainties:
            st.warning(u)

    if answer.get("clarification_question"):
        st.info(f"❓ {answer['clarification_question']}")

    # Claims
    claims = answer.get("claims", [])
    if claims:
        with st.expander(f"Claims ({len(claims)})", expanded=is_latest):
            # Build evidence lookup from the trace's evidence package
            ep = turn.get("evidence_package", {})
            if isinstance(ep, str):
                ep = json.loads(ep)
            evidence_lookup = build_evidence_lookup(ep)

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
                    resolved_cite = resolve_citation(eid, evidence_lookup)
                    if resolved_cite:
                        page_info = f", p.{resolved_cite['page']}" if resolved_cite.get("page") else ""
                        cat_info = f" ({resolved_cite['category']})" if resolved_cite.get("category") else ""
                        st.markdown(
                            f"  - `{eid}` — **{resolved_cite['source']}**{cat_info}{page_info}: {resolved_cite['text'][:120]}..."
                        )
                    else:
                        st.markdown(f"  - `{eid}` — *(unresolved)*")

    # Feedback (only for latest turn)
    if is_latest:
        st.markdown("---")
        st.markdown("#### Feedback")
        fb_key = f"fb_{trace_id}"
        if st.session_state.feedback_submitted.get(fb_key):
            st.success("Thank you! Feedback saved.")
        else:
            fb_col1, fb_col2, fb_col3 = st.columns(3)
            with fb_col1:
                thumbs_up = st.button("👍 Correct", key=f"up_{trace_id}")
            with fb_col2:
                thumbs_down = st.button("👎 Incorrect", key=f"down_{trace_id}")
            with fb_col3:
                correction = st.text_input("Correction (optional)", key=f"corr_{trace_id}")

            if thumbs_up or thumbs_down:
                fb_type = "positive" if thumbs_up else "negative"
                try:
                    save_feedback(
                        trace_id=trace_id,
                        feedback_type=fb_type,
                        reason="" if thumbs_up else "Marked as incorrect",
                        correction=correction if correction else None,
                    )
                    st.session_state.feedback_submitted[fb_key] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")

    st.divider()


# --- Conversation thread ---
thread = _load_thread()

if thread:
    st.markdown(f"### Conversation ({len(thread)} turns)")
    for i, turn in enumerate(thread):
        _render_turn(turn, is_latest=(i == len(thread) - 1))
else:
    # Example questions for first turn
    with st.expander("Example questions"):
        examples = [
            "What is the mechanism of action of amisulpride?",
            "What are the FDA-approved indications for amisulpride?",
            "What is the usual dosage range for amisulpride in schizophrenia?",
            "What are the notable side effects of amisulpride?",
            "What did the EUFEST trial find about amisulpride compared to other antipsychotics in first-episode schizophrenia?",
            "Why does amisulpride help with negative symptoms at low doses?",
        ]
        for q in examples:
            if st.button(q, key=f"example_{q}"):
                st.session_state.question_input = q
                st.rerun()

# --- Question input ---
question = st.text_area(
    "Your question:",
    value=st.session_state.get("question_input", ""),
    height=100,
    placeholder="Ask a question, or follow up on the conversation above...",
    key="question_text",
)

col1, col2, col3 = st.columns([1, 1, 5])
with col1:
    ask_label = "Ask follow-up" if thread else "Ask"
    if st.button(ask_label, type="primary", disabled=not question.strip()):
        with st.spinner("Searching evidence sources and generating answer..."):
            try:
                result = answer_question(
                    question.strip(),
                    conversation_id=st.session_state.conversation_id,
                )
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.pop("question_input", None)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
with col2:
    if st.button("🔄 New conversation"):
        st.session_state.conversation_id = None
        st.session_state.feedback_submitted = {}
        st.session_state.pop("question_input", None)
        st.rerun()
