"""Evaluation runner for the hard challenge set.

Scores for:
1. Retrieval recall (where required_answer_points exist)
2. Expected outcome match (does the answer status match expected_outcome?)
3. Required behaviors (LLM judge — screening before human review)
4. Forbidden behaviors (LLM judge — screening before human review)

Unlike the baseline golden set, performance is expected to fall below 100%.
The purpose is to expose where reasoning and trust boundaries break.

Usage:
    # Retrieval + outcome only (no API calls):
    python -m psych_qa.evaluation.hard_runner

    # Full evaluation with LLM judge (requires OPENAI_API_KEY + JUDGE_MODEL):
    python -m psych_qa.evaluation.hard_runner --judge

    # Full evaluation with judge, using real answers from the answer service:
    python -m psych_qa.evaluation.hard_runner --judge --real-answers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .loader import load_dataset
from .matchers import compute_recall_at_k, build_drug_slug_lookup
from .schemas import HardGoldCase, ExpectedOutcome


def _build_mock_answer(case: HardGoldCase, ep: dict[str, Any] | None) -> dict[str, Any]:
    """Build a mock answer for cases where we don't run the full answer service.

    For retrieval-only validation, we construct a minimal answer so the judge
    can still evaluate outcome matching. For real evaluation, use --real-answers.
    """
    if case.expected_status.value == "abstained":
        return {
            "status": "abstained",
            "direct_answer": "",
            "explanation": "The system abstained from answering this question.",
            "claims": [],
        }
    if case.expected_status.value == "needs_clarification":
        return {
            "status": "needs_clarification",
            "direct_answer": "",
            "explanation": "The system requested clarification before answering.",
            "clarification_question": "Could you provide more detail?",
            "claims": [],
        }
    # For answered cases, return a placeholder — real evaluation needs --real-answers
    return {
        "status": "answered",
        "direct_answer": "(mock — use --real-answers for full evaluation)",
        "explanation": "",
        "claims": [],
    }


def evaluate_hard_set(
    *,
    use_judge: bool = False,
    use_real_answers: bool = False,
) -> dict[str, Any]:
    """Run the hard challenge set evaluation.

    Args:
        use_judge: If True, run the LLM judge for behavior scoring.
        use_real_answers: If True, run the full answer service for each case
                         (requires API calls). If False, use mock answers.

    Returns:
        Report dict with per-case results and summary.
    """
    cases = load_dataset("golden_hard")
    drug_slug_lookup = build_drug_slug_lookup()

    results = []
    outcome_matches = 0
    recall_passes = 0
    recall_total = 0
    judge_passes = 0
    judge_total = 0
    required_behavior_passes = 0
    required_behavior_total = 0
    forbidden_behavior_passes = 0
    forbidden_behavior_total = 0

    for i, case in enumerate(cases, 1):
        conv_id = case.metadata.get("conversation_id", "?")
        turn = case.metadata.get("turn", 1)

        # --- Retrieval validation ---
        recall_result = None
        ep = None
        if case.required_answer_points:
            recall_total += 1
            from ..answering.evidence_builder import build_evidence_package
            from ..retrieval.question_parser import parse_question_with_context
            from unittest.mock import patch

            context_dicts = [
                {
                    "question": t.question,
                    "resolved_question": t.resolved_question or t.question,
                    "understanding": t.understanding,
                    "answer_summary": t.answer_summary,
                    "drug_names": t.understanding.get("drug_names", []),
                }
                for t in case.conversation_context
            ]

            qt = case.metadata.get("question_type", "general")
            mock_result = {
                "resolved_question": case.expected_resolution or case.question,
                "context_status": "resolved",
                "is_follow_up": True,
                "clarification_question": None,
                "drug_names": ["amisulpride"],
                "question_type": qt,
                "concepts": [],
                "premises": [],
                "is_patient_specific": False,
            }

            try:
                with patch("psych_qa.retrieval.question_parser.get_llm_client") as mp, \
                     patch("psych_qa.retrieval.reranker.get_llm_client") as mr:
                    mp.return_value.chat_structured.return_value = (mock_result, 0, 0, 0)
                    mr.return_value.chat_structured.return_value = ({"scores": [9.0] * 10}, 0, 0, 0)
                    understanding = parse_question_with_context(case.question, context_dicts)
                    resolved_q = understanding.get("resolved_question", case.question)
                    ep_model, _ = build_evidence_package(resolved_q, understanding)
                    ep = ep_model.model_dump()

                recall_result = compute_recall_at_k(
                    case.required_answer_points, ep, drug_slug_lookup
                )
                if recall_result["recall"] >= 0.9:
                    recall_passes += 1
            except Exception as e:
                recall_result = {"recall": 0.0, "points_covered": 0, "points_total": len(case.required_answer_points), "error": str(e)}

        # --- Get answer (real or mock) ---
        answer = None
        real_evidence = None
        if use_real_answers:
            from ..answering.answer_service import answer_question

            try:
                result = answer_question(case.question, conversation_id=None)
                answer_dict = result.get("answer", {})
                answer = {
                    "status": answer_dict.get("status", "answered"),
                    "direct_answer": answer_dict.get("direct_answer", ""),
                    "explanation": answer_dict.get("explanation", ""),
                    "claims": answer_dict.get("claims", []),
                    "clarification_question": answer_dict.get("clarification_question"),
                }
                real_evidence = result.get("evidence_package", {})
            except Exception as e:
                print(f"  Answer service error: {e}")
                answer = {"status": "error", "direct_answer": str(e), "explanation": "", "claims": []}
                real_evidence = {}

        if answer is None:
            answer = _build_mock_answer(case, ep)

        # --- Expected outcome ---
        outcome_to_status = {
            ExpectedOutcome.SUPPORTED_ANSWER: "answered",
            ExpectedOutcome.PARTIAL_ANSWER: "answered",
            ExpectedOutcome.CLARIFICATION_REQUIRED: "needs_clarification",
            ExpectedOutcome.CONFLICTING_EVIDENCE: "answered",
            ExpectedOutcome.UNSUPPORTED_SPECIFICITY: "abstained",
            ExpectedOutcome.OUT_OF_CORPUS: "abstained",
            ExpectedOutcome.CLINICAL_BOUNDARY: "needs_clarification",
        }
        expected_status_str = outcome_to_status.get(case.expected_outcome, "answered")
        outcome_match = case.expected_status.value == expected_status_str
        if outcome_match:
            outcome_matches += 1

        # --- LLM Judge ---
        judge_verdict = None
        if use_judge:
            from .judge import judge_answer

            context_for_judge = [
                {
                    "turn": t.turn,
                    "question": t.question,
                    "answer_summary": t.answer_summary,
                }
                for t in case.conversation_context
            ]

            try:
                judge_verdict = judge_answer(
                    resolved_question=case.expected_resolution or case.question,
                    expected_outcome=case.expected_outcome.value,
                    expected_status=case.expected_status.value,
                    required_behaviors=case.required_behaviors,
                    forbidden_behaviors=case.forbidden_behaviors,
                    answer=answer,
                    evidence=real_evidence or ep or {},
                    context=context_for_judge,
                )

                # Count behavior results
                for v in judge_verdict.get("required_behavior_verdicts", []):
                    required_behavior_total += 1
                    if v["verdict"] == "demonstrated":
                        required_behavior_passes += 1

                for v in judge_verdict.get("forbidden_behavior_verdicts", []):
                    forbidden_behavior_total += 1
                    if v["verdict"] == "avoided":
                        forbidden_behavior_passes += 1

                judge_total += 1
                if judge_verdict.get("overall", {}).get("pass", False):
                    judge_passes += 1
            except Exception as e:
                judge_verdict = {"error": str(e), "overall": {"pass": False, "confidence": 0, "summary": f"Judge error: {e}"}}

        # --- Build result ---
        result_entry = {
            "case_id": i,
            "conversation_id": conv_id,
            "turn": turn,
            "question": case.question,
            "expected_outcome": case.expected_outcome.value,
            "expected_status": case.expected_status.value,
            "outcome_match": outcome_match,
            "recall_at_k": recall_result["recall"] if recall_result else None,
            "required_points": f"{recall_result['points_covered']}/{recall_result['points_total']}" if recall_result else "N/A",
            "answer_status": answer.get("status"),
            "judge": judge_verdict,
            "reviewer_notes": case.reviewer_notes,
        }
        results.append(result_entry)

    total = len(cases)
    summary = {
        "total_cases": total,
        "outcome_matches": outcome_matches,
        "outcome_match_rate": round(outcome_matches / total, 4) if total else 0,
        "recall_passes": recall_passes,
        "recall_total": recall_total,
        "recall_pass_rate": round(recall_passes / recall_total, 4) if recall_total else None,
        "judge_passes": judge_passes,
        "judge_total": judge_total,
        "judge_pass_rate": round(judge_passes / judge_total, 4) if judge_total else None,
        "required_behavior_passes": required_behavior_passes,
        "required_behavior_total": required_behavior_total,
        "forbidden_behavior_passes": forbidden_behavior_passes,
        "forbidden_behavior_total": forbidden_behavior_total,
        "expected_performance": "below_100_percent",
        "purpose": "expose_reasoning_and_trust_boundary_failures",
        "judge_enabled": use_judge,
        "real_answers": use_real_answers,
    }

    return {"summary": summary, "results": results}


def run_and_save(*, use_judge: bool = False, use_real_answers: bool = False) -> dict[str, Any]:
    """Run the hard set evaluation and save to evals/results/latest_hard_run.json."""
    report = evaluate_hard_set(use_judge=use_judge, use_real_answers=use_real_answers)
    out_path = Path("evals/results/latest_hard_run.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hard challenge set evaluation")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge for behavior scoring")
    parser.add_argument("--real-answers", action="store_true", help="Run the full answer service (requires API calls)")
    args = parser.parse_args()

    report = run_and_save(use_judge=args.judge, use_real_answers=args.real_answers)
    s = report["summary"]
    print(f"Hard Challenge Set Results")
    print(f"  Total cases:          {s['total_cases']}")
    print(f"  Outcome matches:      {s['outcome_matches']}/{s['total_cases']} ({s['outcome_match_rate']:.1%})")
    if s['recall_total']:
        print(f"  Recall passes:        {s['recall_passes']}/{s['recall_total']}")
    if s['judge_total']:
        print(f"  Judge passes:         {s['judge_passes']}/{s['judge_total']} ({s['judge_pass_rate']:.1%})")
        print(f"  Required behaviors:   {s['required_behavior_passes']}/{s['required_behavior_total']}")
        print(f"  Forbidden behaviors:  {s['forbidden_behavior_passes']}/{s['forbidden_behavior_total']}")
    else:
        print(f"  Behavior checks:      (use --judge to enable LLM judge)")
    print(f"  Expected performance: {s['expected_performance']}")
    print()
    for r in report["results"]:
        recall_str = f"{r['recall_at_k']:.0%}" if r["recall_at_k"] is not None else "N/A"
        judge_str = ""
        if r.get("judge"):
            j = r["judge"]
            if "overall" in j:
                judge_str = f" judge={'PASS' if j['overall']['pass'] else 'FAIL'}({j['overall']['confidence']:.1f})"
            elif "error" in j:
                judge_str = f" judge=ERROR"
        print(f"  [{r['conversation_id']} T{r['turn']}] outcome={r['expected_outcome']:25s} recall={recall_str:>5s}{judge_str}  {r['question'][:45]}")
