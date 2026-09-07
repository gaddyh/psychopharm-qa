"""JSON schemas for structured LLM output.

The LLM cites internal evidence_ids. The renderer resolves these into
book names + pages. The citation validator rejects nonexistent IDs.
"""

from __future__ import annotations


# Schema for the generated answer with citations
ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "direct_answer": {
            "type": "string",
            "description": "A direct, concise answer to the question.",
        },
        "explanation": {
            "type": "string",
            "description": "Detailed explanation and reasoning, referencing evidence.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The claim or statement being cited.",
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Internal evidence IDs supporting this claim.",
                    },
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "uncertainty": {
            "type": ["string", "null"],
            "description": "Explicit uncertainty or disagreement between sources, if any.",
        },
        "status": {
            "type": "string",
            "enum": ["answered", "abstained", "needs_clarification"],
            "description": "Whether the evidence was sufficient to answer.",
        },
        "clarification_question": {
            "type": ["string", "null"],
            "description": "If status is needs_clarification, the question to ask the user.",
        },
    },
    "required": [
        "direct_answer",
        "explanation",
        "citations",
        "uncertainty",
        "status",
        "clarification_question",
    ],
    "additionalProperties": False,
}


# Schema for question parsing (extracting drugs, question type, concepts)
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
    },
    "required": ["drug_names", "question_type", "concepts"],
    "additionalProperties": False,
}
