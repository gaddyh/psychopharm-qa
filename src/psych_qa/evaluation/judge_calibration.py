"""Calibrate the LLM judge against known-good golden_v1 answers.

Golden_v1 cases have validated answers at 100% recall. If the judge
fails these, the judge prompt/schema needs tuning before we trust it
on the hard set.

Usage:
    python -m psych_qa.evaluation.judge_calibration
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .loader import load_dataset
from .judge import judge_answer


def _derive_behaviors_from_golden(case: Any) -> tuple[list[str], list[str]]:
    """Derive required and forbidden behaviors from a golden_v1 case.

    Golden cases don't have explicit behavior lists, so we derive them
    from required_answer_points and forbidden_claims.
    """
    required = []
    for pt in case.required_answer_points:
        required.append(f"Answer addresses: {pt.text}")

    forbidden = []
    for fc in case.forbidden_claims:
        # forbidden_claims are strings in golden_v1
        fc_text = fc.text if hasattr(fc, "text") else str(fc)
        forbidden.append(f"Do not claim: {fc_text}")

    # Add generic grounding behaviors
    required.append("Answer is grounded in cited evidence")
    required.append("Answer does not fabricate information not in sources")

    return required, forbidden


def run_calibration(num_cases: int = 5) -> dict[str, Any]:
    """Run judge calibration on golden_v1 cases.

    Args:
        num_cases: Number of diverse standalone cases to calibrate on.

    Returns:
        Calibration report.
    """
    from ..answering.answer_service import answer_question

    cases = load_dataset("golden_v1")

    # Pick diverse standalone cases
    picks = []
    seen_types = set()
    for i, c in enumerate(cases, 1):
        qt = c.metadata.get("question_type", "general")
        is_conv = c.metadata.get("is_conversation_case", False)
        if not is_conv and qt not in seen_types and len(picks) < num_cases:
            picks.append((i, c))
            seen_types.add(qt)

    results = []
    judge_passes = 0
    judge_total = 0
    required_passes = 0
    required_total = 0
    forbidden_passes = 0
    forbidden_total = 0

    for case_idx, case in picks:
        print(f"\n[{case_idx}] {case.question[:70]}")

        # Run the answer service
        t0 = time.time()
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
            evidence = result.get("evidence_package", {})
        except Exception as e:
            print(f"  Answer service error: {e}")
            answer = {"status": "error", "direct_answer": str(e), "explanation": "", "claims": []}
            evidence = {}
        answer_time = time.time() - t0
        print(f"  Answer: {answer['status']} ({answer_time:.1f}s)")
        print(f"  Claims: {len(answer['claims'])}")

        # Derive behaviors
        required_behaviors, forbidden_behaviors = _derive_behaviors_from_golden(case)

        # Run the judge
        t0 = time.time()
        try:
            verdict = judge_answer(
                resolved_question=case.question,
                expected_outcome="supported_answer",
                expected_status="answered",
                required_behaviors=required_behaviors,
                forbidden_behaviors=forbidden_behaviors,
                answer=answer,
                evidence=evidence,
                context=[],
            )
            judge_time = time.time() - t0
            judge_total += 1

            overall = verdict.get("overall", {})
            passed = overall.get("pass", False)
            confidence = overall.get("confidence", 0)
            if passed:
                judge_passes += 1

            print(f"  Judge: {'PASS' if passed else 'FAIL'} (confidence={confidence:.2f}, {judge_time:.1f}s)")
            print(f"  Summary: {overall.get('summary', '')[:120]}")

            # Count behaviors
            for v in verdict.get("required_behavior_verdicts", []):
                required_total += 1
                if v["verdict"] == "demonstrated":
                    required_passes += 1
                else:
                    print(f"    REQUIRED NOT DEMONSTRATED: {v['behavior'][:60]}")
                    print(f"      → {v['reasoning'][:100]}")

            for v in verdict.get("forbidden_behavior_verdicts", []):
                forbidden_total += 1
                if v["verdict"] == "avoided":
                    forbidden_passes += 1
                else:
                    print(f"    FORBIDDEN VIOLATED: {v['behavior'][:60]}")
                    print(f"      → {v['reasoning'][:100]}")

        except Exception as e:
            judge_time = time.time() - t0
            print(f"  Judge error: {e}")
            verdict = {"error": str(e), "overall": {"pass": False, "confidence": 0, "summary": str(e)}}

        results.append({
            "case_id": case_idx,
            "question": case.question,
            "question_type": case.metadata.get("question_type"),
            "answer_status": answer.get("status"),
            "answer_time_s": round(answer_time, 2),
            "judge_time_s": round(judge_time, 2),
            "judge_verdict": verdict,
            "required_behaviors_derived": required_behaviors,
            "forbidden_behaviors_derived": forbidden_behaviors,
        })

    summary = {
        "total_cases": len(picks),
        "judge_passes": judge_passes,
        "judge_total": judge_total,
        "judge_pass_rate": round(judge_passes / judge_total, 4) if judge_total else 0,
        "required_behavior_passes": required_passes,
        "required_behavior_total": required_total,
        "required_behavior_pass_rate": round(required_passes / required_total, 4) if required_total else 0,
        "forbidden_behavior_passes": forbidden_passes,
        "forbidden_behavior_total": forbidden_total,
        "forbidden_behavior_pass_rate": round(forbidden_passes / forbidden_total, 4) if forbidden_total else 0,
        "calibration_note": "Golden_v1 cases have known-good answers. Judge should mostly pass these.",
    }

    report = {"summary": summary, "results": results}

    # Save
    out_path = Path("evals/results/judge_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    report = run_calibration(num_cases=5)
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"Judge Calibration Results (golden_v1)")
    print(f"  Cases:                {s['total_cases']}")
    print(f"  Judge pass rate:      {s['judge_passes']}/{s['judge_total']} ({s['judge_pass_rate']:.1%})")
    print(f"  Required behaviors:   {s['required_behavior_passes']}/{s['required_behavior_total']} ({s['required_behavior_pass_rate']:.1%})")
    print(f"  Forbidden behaviors:  {s['forbidden_behavior_passes']}/{s['forbidden_behavior_total']} ({s['forbidden_behavior_pass_rate']:.1%})")
    print(f"  Note: {s['calibration_note']}")
