"""Question parser — extract drugs, question type, concepts from a question.

Supports context-aware parsing for follow-up questions in a conversation.
When conversation context is provided, the resolver:
1. Resolves pronouns and implicit references (e.g. "its" → "amisulpride")
2. Produces a standalone resolved_question for retrieval
3. Returns context_status: standalone | resolved | ambiguous | missing_context
If ambiguous, the caller should ask a clarification question instead of answering.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..llm.client import get_llm_client
from ..llm.schemas import CONTEXT_PARSE_SCHEMA, QUESTION_PARSE_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical question parser. Extract drug names, question type, and key concepts from psychopharmacology questions.

Rules:
- Extract ONLY specific drug generic names (e.g. "amisulpride", "haloperidol", "olanzapine").
- Do NOT extract drug classes, categories, or classifications (e.g. "antipsychotics", "atypical antipsychotics", "antidepressants", "benzodiazepines", "mood stabilizers").
- Do NOT extract condition names or symptoms as drug names.
- Use generic names, not brand names, when possible."""

# Terms that are classifications, not drug names — filtered out post-extraction
CLASSIFICATION_TERMS = {
    "antipsychotics", "antipsychotic", "atypical antipsychotics", "atypical antipsychotic",
    "typical antipsychotics", "typical antipsychotic", "first-generation antipsychotics",
    "second-generation antipsychotics", "fga", "sga", "dopamine antagonists",
    "dopamine receptor antagonists", "dopamine-serotonin antagonists", "dsa",
    "antidepressants", "antidepressant", "ssri", "ssris", "snri", "snris", "tca", "tcas",
    "maoi", "maois", "benzodiazepines", "benzodiazepine", "mood stabilizers",
    "mood stabilizer", "stimulants", "stimulant", "anticonvulsants", "anticonvulsant",
    "anticholinergics", "anticholinergic", "cholinesterase inhibitors",
    "opioid receptor modulators", "cation function modulators",
    "histamine modulators", "histamine blockers", "calcium channel inhibitors",
    "dopamine d2 blockers", "d-ran", "d2 antagonists",
}


def _filter_drug_names(names: list[str]) -> list[str]:
    """Filter out classification terms and non-drug names."""
    filtered = []
    for name in names:
        name_lower = name.lower().strip()
        if name_lower in CLASSIFICATION_TERMS:
            logger.debug(f"Filtered out classification term: {name}")
            continue
        # Also filter generic patterns like "X antagonists", "X blockers"
        if re.match(r"^(dopamine|serotonin|histamine|norepinephrine|gaba|glutamate)\s+(antagonist|blocker|modulator|agonist)", name_lower):
            logger.debug(f"Filtered out receptor-class term: {name}")
            continue
        if re.match(r"^(atypical|typical|first.generation|second.generation)\s+", name_lower):
            logger.debug(f"Filtered out generation-class term: {name}")
            continue
        filtered.append(name)
    return filtered


def parse_question(question: str) -> dict:
    """Parse a question using the LLM to extract drugs, type, concepts, premises."""
    client = get_llm_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Parse this question:\n\n{question}"},
    ]
    try:
        result, _, _, _ = client.chat_structured(messages, schema=QUESTION_PARSE_SCHEMA)
        # Post-filter: remove any classification terms the LLM still extracted
        result["drug_names"] = _filter_drug_names(result.get("drug_names", []))
        # Ensure new fields have defaults
        result.setdefault("premises", [])
        result.setdefault("is_patient_specific", False)
        return result
    except Exception as e:
        logger.warning(f"LLM question parsing failed ({e}), using fallback")
        # Fallback: simple regex
        return _fallback_parse(question)


# ---------------------------------------------------------------------------
# Context-aware parsing for follow-up questions
# ---------------------------------------------------------------------------

CONTEXT_SYSTEM_PROMPT = """You are a medical question resolver for a psychopharmacology conversation system.

Your job is to take a potentially ambiguous follow-up question and resolve it using the conversation context.

You must:
1. Determine if the question is a follow-up (references prior turns) or standalone.
2. If it is a follow-up, resolve pronouns and implicit references using the context.
   - "its side effects" → the drug discussed in the previous turn
   - "what about aripiprazole?" → switches the active drug to aripiprazole
   - "the other one" → ambiguous if multiple drugs were discussed
3. Produce a resolved_question that can be answered standalone (without seeing prior turns).
4. Extract drug_names, question_type, concepts from the RESOLVED question.
5. If the reference cannot be resolved (e.g., "the other one" with two drugs), set context_status to "ambiguous" and provide a targeted clarification_question.

Rules:
- Extract ONLY specific drug generic names (e.g. "amisulpride", "haloperidol").
- Do NOT extract drug classes or categories as drug names.
- The resolved_question should be a complete, self-contained question.
- If the question is standalone (no references to prior turns), set context_status to "standalone" and resolved_question equal to the raw question."""


def parse_question_with_context(
    question: str,
    recent_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse a question with optional conversation context for follow-up resolution.

    Args:
        question: The raw question text.
        recent_context: List of recent conversation turns, each containing:
            - question: the previous question
            - resolved_question: the resolved question
            - understanding: the parsed understanding (drug_names, question_type, etc.)
            - answer_summary: brief summary of the answer
            If None or empty, behaves like parse_question (standalone).

    Returns:
        Dict with all QUESTION_PARSE_SCHEMA fields plus:
        - resolved_question: str
        - context_status: "standalone" | "resolved" | "ambiguous" | "missing_context"
        - is_follow_up: bool
        - clarification_question: str | None
    """
    if not recent_context:
        # No context — parse as standalone
        result = parse_question(question)
        result["resolved_question"] = question
        result["context_status"] = "standalone"
        result["is_follow_up"] = False
        result["clarification_question"] = None
        return result

    # Build context description for the LLM
    context_lines = []
    active_drugs: set[str] = set()
    for i, turn in enumerate(recent_context, 1):
        q = turn.get("resolved_question") or turn.get("question", "")
        understanding = turn.get("understanding", {})
        drugs = understanding.get("drug_names", [])
        qtype = understanding.get("question_type", "unknown")
        summary = turn.get("answer_summary", "")
        active_drugs.update(drugs)
        context_lines.append(
            f"Turn {i}:\n"
            f"  Question: {q}\n"
            f"  Drugs discussed: {', '.join(drugs) if drugs else 'none'}\n"
            f"  Question type: {qtype}\n"
            f"  Answer summary: {summary[:200] if summary else 'N/A'}"
        )

    context_text = "\n\n".join(context_lines)
    active_drugs_str = ", ".join(sorted(active_drugs)) if active_drugs else "none"

    user_prompt = (
        f"Conversation context (last {len(recent_context)} turns):\n\n"
        f"{context_text}\n\n"
        f"Active drugs in conversation: {active_drugs_str}\n\n"
        f"New question: {question}\n\n"
        f"Resolve this question using the conversation context."
    )

    client = get_llm_client()
    messages = [
        {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result, _, _, _ = client.chat_structured(messages, schema=CONTEXT_PARSE_SCHEMA)
        # Post-filter drug names
        result["drug_names"] = _filter_drug_names(result.get("drug_names", []))
        result.setdefault("premises", [])
        result.setdefault("is_patient_specific", False)
        return result
    except Exception as e:
        logger.warning(f"LLM context-aware parsing failed ({e}), using fallback")
        # Fallback: parse as standalone
        fallback = _fallback_parse(question)
        fallback["resolved_question"] = question
        fallback["context_status"] = "standalone"
        fallback["is_follow_up"] = False
        fallback["clarification_question"] = None
        return fallback


def _fallback_parse(question: str) -> dict:
    """Simple regex-based fallback."""
    # Common drug names to look for
    drug_patterns = [
        "amisulpride", "aripiprazole", "clozapine", "olanzapine", "quetiapine",
        "risperidone", "haloperidol", "chlorpromazine", "fluoxetine", "sertraline",
        "citalopram", "escitalopram", "paroxetine", "venlafaxine", "duloxetine",
        "bupropion", "mirtazapine", "lithium", "valproate", "lamotrigine",
    ]
    drug_names = []
    q_lower = question.lower()
    for d in drug_patterns:
        if d in q_lower:
            drug_names.append(d)

    # Question type detection
    qtype = "general"
    if re.search(r"mechanism|how.*work|mode.*action|receptor|target", q_lower):
        qtype = "mechanism"
    elif re.search(r"indic|approv|used.*for|prescrib", q_lower):
        qtype = "indication"
    elif re.search(r"dose|dosage|mg", q_lower):
        qtype = "dosing"
    elif re.search(r"side.*effect|adverse|toxic", q_lower):
        qtype = "side_effects"
    elif re.search(r"interact|contraindic", q_lower):
        qtype = "interactions"
    elif re.search(r"pharmacokinetic|half.*life|metabol", q_lower):
        qtype = "pharmacokinetics"
    elif re.search(r"pregnan|breast.*feed|elder|pediatr|renal|hepatic", q_lower):
        qtype = "special_populations"

    return {
        "drug_names": drug_names,
        "question_type": qtype,
        "concepts": [],
        "premises": [],
        "is_patient_specific": bool(re.search(r"my patient|should i give|what if i|patient has|this patient", q_lower)),
    }
