"""JSON schemas for structured LLM output.

The LLM cites internal evidence_ids. The renderer resolves these into
book names + pages. The citation validator rejects nonexistent IDs.
"""

from __future__ import annotations


# Schema for the generated answer — claims-first V2
# The LLM emits atomic claims with evidence_ids. The application:
# 1. Assigns `critical` deterministically from evidence category (never LLM-decided)
# 2. Assembles direct_answer/explanation prose from validated claims
# 3. Validates that every claim has >=1 valid evidence_id
ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "One atomic medical statement.",
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Internal evidence IDs supporting this claim. At least one required.",
                    },
                    "claim_role": {
                        "type": "string",
                        "enum": ["direct", "explanatory", "caveat", "comparison"],
                        "description": "The role of this claim in the answer.",
                    },
                },
                "required": ["text", "evidence_ids", "claim_role"],
                "additionalProperties": False,
            },
        },
        "status": {
            "type": "string",
            "enum": ["answered", "abstained", "needs_clarification"],
            "description": "Whether the evidence was sufficient to answer.",
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Explicit uncertainties or disagreements between sources.",
        },
        "clarification_question": {
            "type": ["string", "null"],
            "description": "If status is needs_clarification, the question to ask the user.",
        },
    },
    "required": ["claims", "status", "uncertainties", "clarification_question"],
    "additionalProperties": False,
}


# Schema for question parsing (extracting drugs, question type, concepts, premises)
QUESTION_PARSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "drug_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Drug names mentioned in the question.",
        },
        "question_type": {
            "type": "string",
            "enum": [
                "mechanism",
                "indication",
                "dosing",
                "side_effects",
                "interactions",
                "pharmacokinetics",
                "special_populations",
                "comparison",
                "general",
            ],
            "description": "The type of question being asked.",
        },
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key medical concepts in the question.",
        },
        "premises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Factual premises stated in the question that should be verified against evidence. E.g. 'amisulpride is a serotonin 2A antagonist'. Empty if no factual premises.",
        },
        "is_patient_specific": {
            "type": "boolean",
            "description": "True if the question asks about a specific patient scenario (e.g. 'my patient', 'should I give').",
        },
    },
    "required": ["drug_names", "question_type", "concepts", "premises", "is_patient_specific"],
    "additionalProperties": False,
}


# Schema for context-aware question parsing (follow-up resolution)
CONTEXT_PARSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "resolved_question": {
            "type": "string",
            "description": (
                "The question after resolving pronouns and implicit references using "
                "conversation context. If the question is standalone, this equals "
                "the raw question. If ambiguous, provide your best guess but set "
                "context_status to 'ambiguous'."
            ),
        },
        "context_status": {
            "type": "string",
            "enum": ["standalone", "resolved", "ambiguous", "missing_context"],
            "description": (
                "standalone: no context needed. "
                "resolved: context successfully applied. "
                "ambiguous: cannot resolve which entity is referenced. "
                "missing_context: references prior context but none available."
            ),
        },
        "is_follow_up": {
            "type": "boolean",
            "description": "True if the question references or builds on prior conversation turns.",
        },
        "clarification_question": {
            "type": ["string", "null"],
            "description": "If context_status is 'ambiguous', the targeted question to ask the user. Null otherwise.",
        },
        "drug_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Drug names after context resolution (inherited + newly mentioned).",
        },
        "question_type": {
            "type": "string",
            "enum": [
                "mechanism",
                "indication",
                "dosing",
                "side_effects",
                "interactions",
                "pharmacokinetics",
                "special_populations",
                "comparison",
                "general",
            ],
            "description": "The type of question being asked, after resolution.",
        },
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key medical concepts in the resolved question.",
        },
        "premises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Factual premises stated in the question that should be verified against evidence.",
        },
        "is_patient_specific": {
            "type": "boolean",
            "description": "True if the question asks about a specific patient scenario.",
        },
    },
    "required": [
        "resolved_question",
        "context_status",
        "is_follow_up",
        "clarification_question",
        "drug_names",
        "question_type",
        "concepts",
        "premises",
        "is_patient_specific",
    ],
    "additionalProperties": False,
}


# Schema for claim-to-evidence entailment judging
ENTAILMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "UNSUPPORTED", "UNSUPPORTED_SPECIFICITY", "CONTRADICTED", "MISSING_CITATION"],
            "description": "Whether the evidence supports the claim.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the verdict.",
        },
        "unsupported_specificity_detail": {
            "type": ["string", "null"],
            "description": "If UNSUPPORTED_SPECIFICITY, what specific detail was added beyond the evidence.",
        },
    },
    "required": ["verdict", "reasoning", "unsupported_specificity_detail"],
    "additionalProperties": False,
}
