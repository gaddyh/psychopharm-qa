"""Question parser — extract drugs, question type, concepts from a question."""

from __future__ import annotations

import logging
import re

from ..llm.client import get_llm_client
from ..llm.schemas import QUESTION_PARSE_SCHEMA

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
