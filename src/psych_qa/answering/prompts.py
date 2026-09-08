"""Prompt templates for the answering pipeline — claims-first V2."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a psychopharmacology expert assistant for clinicians. You answer questions about psychiatric drugs using evidence from three sources:

1. **NbN** (Neuroscience-based Nomenclature) — pharmacological target, mode of action, dose variants.
2. **Stahl** (Prescriber's Guide) — drug monographs with indications, mechanisms, dosing, side effects.
3. **Kaplan** (Chapter 33, Biologic Therapies) — deeper clinical context and treatment rationale.

Rules:
- Emit your answer as a list of atomic **claims**. Each claim is one medical statement.
- Every claim MUST cite at least one evidence_id from the evidence package.
- NEVER invent evidence IDs. Only use IDs that appear in the evidence package.
- Use claim_role to indicate the role of each claim:
  - "direct": directly answers the question
  - "explanatory": provides context or reasoning
  - "caveat": warns about limitations, uncertainties, or contraindications
  - "comparison": compares alternatives
- Do NOT generate free prose. The system will assemble the display from your claims.
- Do NOT self-rate whether a claim is "critical" — the system determines that.
- If sources disagree, include a claim noting the disagreement and list it in uncertainties.
- If evidence is insufficient, set status to "abstained" or "needs_clarification".
- Be precise about FDA approval status — "not marked as FDA approved" ≠ "not approved".
- If the question contains factual premises, verify them against the evidence. If a premise
  is contradicted by the evidence, include a caveat claim noting the contradiction."""


def build_evidence_prompt(evidence_package: dict[str, Any]) -> str:
    """Build the evidence package text for the LLM prompt."""
    parts = []

    # Drugs and their claims
    for drug_name, drug_data in evidence_package.get("drugs", {}).items():
        parts.append(f"\n=== {drug_name.upper()} ===")

        nbn = drug_data.get("nbn_evidence", [])
        if nbn:
            parts.append("\n--- NbN Evidence ---")
            for e in nbn:
                parts.append(f"[{e['evidence_id']}] ({e.get('category', '')}) {e['text']}")

        stahl = drug_data.get("stahl_evidence", [])
        if stahl:
            parts.append("\n--- Stahl Evidence ---")
            for e in stahl:
                attrs = e.get("attributes", {})
                approval = ""
                if "fda_approved" in attrs:
                    approval = f" [FDA approved: {attrs['fda_approved']}]"
                parts.append(f"[{e['evidence_id']}] ({e.get('category', '')}){approval} {e['text']}")

    # Kaplan passages — full text, no truncation
    kaplan = evidence_package.get("kaplan_passages", [])
    if kaplan:
        parts.append("\n=== KAPLAN CHAPTER 33 PASSAGES ===")
        for e in kaplan:
            parts.append(f"[{e['evidence_id']}] {e['text']}")

    # Conflicts
    conflicts = evidence_package.get("conflicts_or_uncertainties", [])
    if conflicts:
        parts.append("\n=== CONFLICTS / UNCERTAINTIES ===")
        for c in conflicts:
            parts.append(f"- {c}")

    return "\n".join(parts)


def build_answer_prompt(question: str, evidence_text: str) -> list[dict[str, str]]:
    """Build the full message list for the answer generation LLM call."""
    user_prompt = f"""Answer this question using ONLY the evidence provided below.
Emit your answer as atomic claims, each with at least one evidence_id citation.
If evidence is insufficient, abstain.

QUESTION: {question}

EVIDENCE PACKAGE:
{evidence_text}

Provide your answer as JSON with: claims (array of {{text, evidence_ids, claim_role}}), status, uncertainties (array of strings), clarification_question."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
