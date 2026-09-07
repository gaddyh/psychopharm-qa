"""Question-type evidence filtering — select only relevant claim categories.

Instead of sending all 179 Stahl claims to the LLM, filter by question type
so relevant evidence isn't buried. Deterministic claims are filtered by
category; Kaplan passages are always included (they're already ranked).
"""

from __future__ import annotations

from typing import Any

# Always-on categories for context (included regardless of question type)
ALWAYS_ON_CATEGORIES: set[str] = {
    "mechanism",
    "class",
    "mode_of_action",
    "target",
    "classification",
}

# Question type → relevant claim categories
# Maps to the actual category strings in the database (verified)
QUESTION_TYPE_CATEGORIES: dict[str, set[str]] = {
    "mechanism": {
        "mechanism",
        "class",
        "mode_of_action",
        "target",
        "neurobiology",
        "neurotransmitter_effects",
        "side_effect_mechanism",
        "classification",
    },
    "dosing": {
        "dosage_range",
        "dosing",
        "dosing_instructions",
        "dosing_tips",
        "dosage_forms",
        "indication",
        "renal_impairment",
        "hepatic_impairment",
        "cardiac_impairment",
        "pharmacokinetics",
        "elderly",
        "pediatric",
    },
    "side_effects": {
        "side_effects",
        "side_effect_mechanism",
        "side_effect_management",
        "sedation",
        "weight_gain",
        "contraindications",
        "overdose",
        "monitoring",
    },
    "interactions": {
        "interactions",
        "contraindications",
        "pharmacokinetics",
    },
    "indication": {
        "indication",
        "efficacy",
        "therapeutics",
        "treatment_failure",
        "target_symptoms",
        "onset",
    },
    "special_populations": {
        "pregnancy",
        "breast_feeding",
        "renal_impairment",
        "hepatic_impairment",
        "cardiac_impairment",
        "elderly",
        "pediatric",
    },
    "pharmacokinetics": {
        "pharmacokinetics",
        "dosage_forms",
        "habit_forming",
        "long_term_use",
        "discontinuation",
    },
    "comparison": {
        # Comparison questions need broad context
        "mechanism",
        "indication",
        "efficacy",
        "side_effects",
        "side_effect_mechanism",
        "class",
        "mode_of_action",
        "target",
        "neurotransmitter_effects",
        "neurobiology",
        "therapeutics",
        "potential_advantages",
        "potential_disadvantages",
        "classification",
    },
    "general": {
        # General questions get everything (fallback)
    },
}


def get_categories_for_question_type(question_type: str) -> set[str]:
    """Get the set of relevant claim categories for a question type.

    Always includes ALWAYS_ON_CATEGORIES. Returns empty set for "general"
    which signals "all categories" to the caller.
    """
    if question_type not in QUESTION_TYPE_CATEGORIES:
        return set()  # signal: all categories

    categories = QUESTION_TYPE_CATEGORIES[question_type]
    if not categories:
        return set()  # general/fallback: all categories

    return categories | ALWAYS_ON_CATEGORIES


def filter_claims_by_question_type(
    claims: list[dict[str, Any]],
    question_type: str,
) -> list[dict[str, Any]]:
    """Filter claims to only those matching the question type's categories.

    If question_type is "general" or unknown, returns all claims.
    """
    relevant = get_categories_for_question_type(question_type)
    if not relevant:
        return claims  # all categories

    return [c for c in claims if c.get("category") in relevant]
