"""Dataset loader for evaluation cases.

Loads and validates JSONL files from evals/datasets/ into typed case objects.
Refuses to compute psychiatrist-approved metrics against cases with
review_status = needs_sasson_approval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_settings
from .schemas import (
    AbstentionCase,
    GoldCase,
    RegressionCase,
    ReviewStatus,
    parse_case,
)


def get_dataset_path(dataset: str) -> Path:
    """Get the path to a dataset JSONL file."""
    settings = get_settings()
    return settings.evals_dir / "datasets" / f"{dataset}.jsonl"


def load_dataset(dataset: str) -> list[GoldCase | AbstentionCase | RegressionCase]:
    """Load and validate all cases from a dataset JSONL file.

    Args:
        dataset: Dataset name without extension (e.g. "golden_v1", "abstention_v1").

    Returns:
        List of typed case objects.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        ValidationError: If any case fails schema validation.
    """
    path = get_dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases: list[GoldCase | AbstentionCase | RegressionCase] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                case = parse_case(raw, dataset)
                cases.append(case)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse {dataset} line {line_num}: {e}\n  Raw: {line[:200]}"
                ) from e

    return cases


def filter_approved(
    cases: list[GoldCase | AbstentionCase | RegressionCase],
) -> list[GoldCase | AbstentionCase | RegressionCase]:
    """Return only cases that have been approved by a clinician."""
    return [c for c in cases if c.review_status == ReviewStatus.APPROVED]


def has_unapproved_cases(
    cases: list[GoldCase | AbstentionCase | RegressionCase],
) -> bool:
    """Check if any cases still need clinician approval."""
    return any(c.review_status == ReviewStatus.NEEDS_SASSON_APPROVAL for c in cases)


def export_cases_to_jsonl(
    cases: list[dict[str, Any]],
    dataset: str,
) -> Path:
    """Export cases to a JSONL file (used by the export script, not Streamlit)."""
    path = get_dataset_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for case in cases:
            f.write(json.dumps(case, default=str) + "\n")
    return path
