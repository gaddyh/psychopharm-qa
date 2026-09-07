"""Tests for conversation follow-up question resolution.

Tests the 5 key behaviors:
1. Pronoun resolution: "What about its side effects?" inherits amisulpride
2. Topic switch: "What about aripiprazole?" switches active drug
3. Ambiguity: After two drugs, "Which one causes more weight gain?" → clarification
4. Fresh retrieval: Side-effects turn retrieves side-effect evidence, not mechanism evidence
5. No contamination: Incorrect statement in previous answer is NOT repeated
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: Pronoun resolution
# ---------------------------------------------------------------------------

def test_pronoun_resolution_inherits_drug():
    """'What about its side effects?' should inherit amisulpride from context."""
    from psych_qa.retrieval.question_parser import parse_question_with_context

    recent_context = [
        {
            "question": "What is the mechanism of action of amisulpride?",
            "resolved_question": "What is the mechanism of action of amisulpride?",
            "understanding": {
                "drug_names": ["amisulpride"],
                "question_type": "mechanism",
                "concepts": ["mechanism", "amisulpride"],
                "premises": [],
                "is_patient_specific": False,
            },
            "answer_summary": "Amisulpride blocks presynaptic D2 at low doses, postsynaptic D2 at higher doses.",
            "drug_names": ["amisulpride"],
        }
    ]

    mock_result = {
        "resolved_question": "What are the side effects of amisulpride?",
        "context_status": "resolved",
        "is_follow_up": True,
        "clarification_question": None,
        "drug_names": ["amisulpride"],
        "question_type": "side_effects",
        "concepts": ["side effects", "amisulpride"],
        "premises": [],
        "is_patient_specific": False,
    }

    with patch("psych_qa.retrieval.question_parser.get_llm_client") as mock_client:
        mock_client.return_value.chat_structured.return_value = (mock_result, 0, 0, 0)
        result = parse_question_with_context(
            "What about its side effects?",
            recent_context,
        )

    assert result["context_status"] == "resolved"
    assert result["is_follow_up"] is True
    assert "amisulpride" in result["drug_names"]
    assert result["resolved_question"] == "What are the side effects of amisulpride?"
    assert result["question_type"] == "side_effects"


# ---------------------------------------------------------------------------
# Test 2: Topic switch
# ---------------------------------------------------------------------------

def test_topic_switch_changes_active_drug():
    """'What about aripiprazole?' should switch to aripiprazole, not retain amisulpride."""
    from psych_qa.retrieval.question_parser import parse_question_with_context

    recent_context = [
        {
            "question": "What is the mechanism of action of amisulpride?",
            "resolved_question": "What is the mechanism of action of amisulpride?",
            "understanding": {
                "drug_names": ["amisulpride"],
                "question_type": "mechanism",
                "concepts": [],
                "premises": [],
                "is_patient_specific": False,
            },
            "answer_summary": "Amisulpride blocks D2 receptors.",
            "drug_names": ["amisulpride"],
        }
    ]

    mock_result = {
        "resolved_question": "What is the mechanism of action of aripiprazole?",
        "context_status": "resolved",
        "is_follow_up": True,
        "clarification_question": None,
        "drug_names": ["aripiprazole"],
        "question_type": "mechanism",
        "concepts": ["mechanism", "aripiprazole"],
        "premises": [],
        "is_patient_specific": False,
    }

    with patch("psych_qa.retrieval.question_parser.get_llm_client") as mock_client:
        mock_client.return_value.chat_structured.return_value = (mock_result, 0, 0, 0)
        result = parse_question_with_context(
            "What about aripiprazole?",
            recent_context,
        )

    assert result["context_status"] == "resolved"
    assert "aripiprazole" in result["drug_names"]
    assert "amisulpride" not in result["drug_names"]
    assert "aripiprazole" in result["resolved_question"]


# ---------------------------------------------------------------------------
# Test 3: Ambiguity → clarification
# ---------------------------------------------------------------------------

def test_ambiguity_returns_clarification():
    """After discussing two drugs, 'Which one causes more weight gain?' → ambiguous."""
    from psych_qa.retrieval.question_parser import parse_question_with_context

    recent_context = [
        {
            "question": "What is the mechanism of action of amisulpride?",
            "resolved_question": "What is the mechanism of action of amisulpride?",
            "understanding": {
                "drug_names": ["amisulpride"],
                "question_type": "mechanism",
                "concepts": [],
                "premises": [],
                "is_patient_specific": False,
            },
            "answer_summary": "Amisulpride blocks D2.",
            "drug_names": ["amisulpride"],
        },
        {
            "question": "What about olanzapine?",
            "resolved_question": "What is the mechanism of action of olanzapine?",
            "understanding": {
                "drug_names": ["olanzapine"],
                "question_type": "mechanism",
                "concepts": [],
                "premises": [],
                "is_patient_specific": False,
            },
            "answer_summary": "Olanzapine blocks D2 and 5-HT2A.",
            "drug_names": ["olanzapine"],
        },
    ]

    mock_result = {
        "resolved_question": "Which drug causes more weight gain, amisulpride or olanzapine?",
        "context_status": "ambiguous",
        "is_follow_up": True,
        "clarification_question": "Do you mean amisulpride or olanzapine?",
        "drug_names": ["amisulpride", "olanzapine"],
        "question_type": "comparison",
        "concepts": ["weight gain"],
        "premises": [],
        "is_patient_specific": False,
    }

    with patch("psych_qa.retrieval.question_parser.get_llm_client") as mock_client:
        mock_client.return_value.chat_structured.return_value = (mock_result, 0, 0, 0)
        result = parse_question_with_context(
            "Which one causes more weight gain?",
            recent_context,
        )

    assert result["context_status"] == "ambiguous"
    assert result["clarification_question"] is not None
    assert result["is_follow_up"] is True


# ---------------------------------------------------------------------------
# Test 4: Standalone (no context)
# ---------------------------------------------------------------------------

def test_no_context_returns_standalone():
    """Without conversation context, the question is parsed as standalone."""
    from psych_qa.retrieval.question_parser import parse_question_with_context

    result = parse_question_with_context("What are the side effects of amisulpride?", None)

    assert result["context_status"] == "standalone"
    assert result["is_follow_up"] is False
    assert result["resolved_question"] == "What are the side effects of amisulpride?"
    assert result["clarification_question"] is None


# ---------------------------------------------------------------------------
# Test 5: Ambiguous answer service returns needs_clarification
# ---------------------------------------------------------------------------

def test_ambiguous_followup_returns_clarification_without_retrieval():
    """When context_status is 'ambiguous', the service returns needs_clarification
    without building an evidence package or calling the answer LLM."""
    from psych_qa.answering.answer_service import answer_question

    mock_understanding = {
        "resolved_question": "Which drug causes more weight gain?",
        "context_status": "ambiguous",
        "is_follow_up": True,
        "clarification_question": "Do you mean amisulpride or olanzapine?",
        "drug_names": ["amisulpride", "olanzapine"],
        "question_type": "comparison",
        "concepts": ["weight gain"],
        "premises": [],
        "is_patient_specific": False,
    }

    with patch("psych_qa.answering.answer_service.parse_question_with_context") as mock_parse, \
         patch("psych_qa.answering.answer_service.conv_repo") as mock_repo, \
         patch("psych_qa.answering.answer_service.get_llm_client") as mock_llm:
        mock_parse.return_value = mock_understanding
        mock_repo.get_recent_context.return_value = []
        mock_repo.get_next_turn_number.return_value = 1
        mock_repo.save_answer_trace.return_value = 999
        mock_llm.return_value.chat_model = "test-model"

        result = answer_question(
            "Which one causes more weight gain?",
            conversation_id=42,
        )

    assert result["answer"]["status"] == "needs_clarification"
    assert result["answer"]["clarification_question"] == "Do you mean amisulpride or olanzapine?"
    assert result["answer"]["claims"] == []
    assert result["conversation_id"] == 42
    assert result["turn_number"] == 1


# ---------------------------------------------------------------------------
# Test 6: Fresh retrieval — follow-up uses resolved question for retrieval
# ---------------------------------------------------------------------------

def test_followup_uses_resolved_question_for_retrieval():
    """The evidence builder should receive the resolved question, not the raw question."""
    from psych_qa.answering.answer_service import answer_question

    mock_understanding = {
        "resolved_question": "What are the side effects of amisulpride?",
        "context_status": "resolved",
        "is_follow_up": True,
        "clarification_question": None,
        "drug_names": ["amisulpride"],
        "question_type": "side_effects",
        "concepts": ["side effects"],
        "premises": [],
        "is_patient_specific": False,
    }

    with patch("psych_qa.answering.answer_service.parse_question_with_context") as mock_parse, \
         patch("psych_qa.answering.answer_service.build_evidence_package") as mock_build, \
         patch("psych_qa.answering.answer_service.check_sufficiency") as mock_suff, \
         patch("psych_qa.answering.answer_service.conv_repo") as mock_repo, \
         patch("psych_qa.answering.answer_service.get_llm_client") as mock_llm, \
         patch("psych_qa.answering.answer_service.compute_run_metrics") as mock_metrics:
        mock_parse.return_value = mock_understanding
        mock_repo.get_recent_context.return_value = []
        mock_repo.get_next_turn_number.return_value = 2
        mock_repo.save_answer_trace.return_value = 100

        # Mock evidence package
        from psych_qa.domain.models import EvidencePackage
        mock_ep = EvidencePackage(question="What are the side effects of amisulpride?", drugs={}, kaplan_passages=[])
        mock_build.return_value = (mock_ep, {"budget_diagnostics": {}})
        mock_suff.return_value = (False, "No evidence found")
        mock_llm.return_value.chat_model = "test-model"
        mock_metrics.return_value = {}

        result = answer_question(
            "What about its side effects?",
            conversation_id=42,
        )

    # build_evidence_package should have been called with the RESOLVED question
    call_args = mock_build.call_args
    assert call_args[0][0] == "What are the side effects of amisulpride?"


# ---------------------------------------------------------------------------
# Test 7: No contamination — previous answer prose is NOT in the answer prompt
# ---------------------------------------------------------------------------

def test_no_previous_prose_in_answer_prompt():
    """The answer prompt should contain only the resolved question + evidence,
    not any previous answer text."""
    from psych_qa.answering.prompts import build_answer_prompt

    # The prompt builder should NOT accept or include previous answers
    messages = build_answer_prompt("What are the side effects of amisulpride?", "EVIDENCE HERE")

    # Check that the prompt has exactly 2 messages (system + user)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    assert "What are the side effects of amisulpride?" in user_content
    assert "EVIDENCE HERE" in user_content
    # There should be no "Previous conversation" or "As discussed" section
    assert "Previous conversation" not in user_content
    assert "As discussed" not in user_content


# ---------------------------------------------------------------------------
# Test 8: Repository — get_recent_context builds correct structure
# ---------------------------------------------------------------------------

def test_get_recent_context_structure(monkeypatch):
    """get_recent_context should return turns with the right fields."""
    from psych_qa.repositories import conversations as conv_repo

    # SQL returns DESC (most recent first), code reverses to chronological
    mock_rows = [
        # (id, turn_number, question, resolved_question, understanding, answer)
        (2, 2, "What about its side effects?",
         "What are the side effects of amisulpride?",
         '{"drug_names": ["amisulpride"], "question_type": "side_effects"}',
         '{"direct_answer": "EPS and prolactin elevation.", "claims": []}'),
        (1, 1, "What is the mechanism of amisulpride?",
         "What is the mechanism of amisulpride?",
         '{"drug_names": ["amisulpride"], "question_type": "mechanism"}',
         '{"direct_answer": "It blocks D2 receptors.", "claims": []}'),
    ]

    class MockSession:
        def execute(self, sql, params):
            class MockResult:
                def __init__(self, rows):
                    self._rows = rows
                def fetchall(self):
                    return self._rows
            return MockResult(mock_rows)

        def close(self):
            pass

    monkeypatch.setattr("psych_qa.repositories.conversations.get_session", lambda: MockSession())

    context = conv_repo.get_recent_context(conversation_id=1, max_turns=3)

    assert len(context) == 2
    # Should be in chronological order (oldest first)
    assert context[0]["question"] == "What is the mechanism of amisulpride?"
    assert context[1]["question"] == "What about its side effects?"
    assert context[0]["resolved_question"] == "What is the mechanism of amisulpride?"
    assert context[1]["resolved_question"] == "What are the side effects of amisulpride?"
    assert "amisulpride" in context[0]["drug_names"]
    assert "It blocks D2 receptors." in context[0]["answer_summary"]
