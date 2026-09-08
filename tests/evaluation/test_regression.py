"""Tests for regression-case promotion (with mocked DB)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from psych_qa.evaluation.regression import (
    _append_regression_case,
    list_promotable_feedback,
    promote_feedback_to_regression,
)
from psych_qa.evaluation.schemas import RegressionCase, ReviewStatus


class TestAppendRegressionCase:
    def test_append_to_file(self, tmp_path, monkeypatch):
        """Test that regression cases are appended to the JSONL file."""
        # Patch the regression module's reference to get_dataset_path
        from psych_qa.evaluation import regression as reg_module
        monkeypatch.setattr(
            reg_module, "get_dataset_path",
            lambda name: tmp_path / f"{name}.jsonl",
        )

        case = RegressionCase(
            question="Test question?",
            expected_status="answered",
            reviewer_notes="Test notes",
            source_trace_id=42,
        )
        path = _append_regression_case(case)
        assert path.exists()
        content = path.read_text().strip()
        assert "Test question?" in content
        assert "42" in content

    def test_append_multiple(self, tmp_path, monkeypatch):
        from psych_qa.evaluation import regression as reg_module
        monkeypatch.setattr(
            reg_module, "get_dataset_path",
            lambda name: tmp_path / f"{name}.jsonl",
        )

        for i in range(3):
            case = RegressionCase(
                question=f"Question {i}?",
                source_trace_id=i,
            )
            _append_regression_case(case)

        # Read the file directly
        path = tmp_path / "regression.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "Question" in data["question"]


class TestPromoteFeedback:
    def test_promote_not_found(self):
        """Promoting a nonexistent feedback ID returns None."""
        with patch("psych_qa.evaluation.regression.get_engine") as mock_engine:
            mock_conn = mock_engine.return_value.connect.return_value.__enter__.return_value
            mock_conn.execute.return_value.fetchone.return_value = None
            result = promote_feedback_to_regression(999)
        assert result is None

    def test_promote_negative_feedback(self, tmp_path, monkeypatch):
        """Test promoting negative feedback to a regression case."""
        from psych_qa.evaluation import regression as reg_module
        monkeypatch.setattr(
            reg_module, "get_dataset_path",
            lambda name: tmp_path / f"{name}.jsonl",
        )

        mock_row = (
            "negative",  # feedback_type
            "Wrong dose mentioned",  # reason
            None,  # correction
            "What is the dose?",  # question
            json.dumps({"claims": [{"text": "Dose is 50mg", "evidence_ids": ["stahl_1"]}]}),  # answer
            json.dumps({"drugs": {}}),  # evidence_package
        )

        with patch("psych_qa.evaluation.regression.get_engine") as mock_engine:
            mock_conn = mock_engine.return_value.connect.return_value.__enter__.return_value
            mock_conn.execute.return_value.fetchone.return_value = mock_row

            case = promote_feedback_to_regression(
                feedback_id=1,
                reviewer_notes="Dose was wrong",
                reviewed_by="Sasson",
            )

        assert case is not None
        assert case.question == "What is the dose?"
        assert case.reviewed_by == "Sasson"
        assert case.review_status == ReviewStatus.NEEDS_SASSON_APPROVAL
        assert case.source_trace_id == 1
        # The wrong claim should be in forbidden_claims
        assert "Dose is 50mg" in case.forbidden_claims

    def test_promote_correction_feedback(self, tmp_path, monkeypatch):
        """Test promoting correction feedback to a regression case."""
        from psych_qa.evaluation import regression as reg_module
        monkeypatch.setattr(
            reg_module, "get_dataset_path",
            lambda name: tmp_path / f"{name}.jsonl",
        )

        mock_row = (
            "correction",
            None,
            "The correct dose is 400-800 mg/day",
            "What is the dose?",
            json.dumps({"claims": [{"text": "Dose is 50mg", "evidence_ids": []}]}),
            json.dumps({"drugs": {}}),
        )

        with patch("psych_qa.evaluation.regression.get_engine") as mock_engine:
            mock_conn = mock_engine.return_value.connect.return_value.__enter__.return_value
            mock_conn.execute.return_value.fetchone.return_value = mock_row

            case = promote_feedback_to_regression(
                feedback_id=2,
                reviewer_notes="Corrected dose",
                reviewed_by="Sasson",
            )

        assert case is not None
        assert len(case.required_answer_points) == 1
        assert "400-800 mg/day" in case.required_answer_points[0].text


class TestListPromotableFeedback:
    def test_list_empty(self):
        with patch("psych_qa.evaluation.regression.get_engine") as mock_engine:
            mock_conn = mock_engine.return_value.connect.return_value.__enter__.return_value
            mock_conn.execute.return_value.fetchall.return_value = []
            result = list_promotable_feedback()
        assert result == []

    def test_list_with_items(self):
        mock_rows = [
            (1, "negative", "Wrong", None, 10, "What is the dose?", None),
            (2, "correction", None, "Corrected", 11, "What is the mechanism?", None),
        ]
        with patch("psych_qa.evaluation.regression.get_engine") as mock_engine:
            mock_conn = mock_engine.return_value.connect.return_value.__enter__.return_value
            mock_conn.execute.return_value.fetchall.return_value = mock_rows
            result = list_promotable_feedback()

        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["feedback_type"] == "negative"
        assert result[1]["feedback_type"] == "correction"
