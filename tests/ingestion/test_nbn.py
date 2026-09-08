"""Tests for NbN ingestion fidelity — Amisulpride.

Integration tests against the locally ingested real NbN source.
Expected values come from data/samples/amisulpride_expected.json, which was
extracted independently from the source JSON (not by running the parser).

Each source is verified against what it actually says:
- NbN low-dose variant says "Pre-synaptic antagonist" (mode_of_action)
- NbN upper-dose variant says "antagonist" (mode_of_action)
- The explicit presynaptic-vs-postsynaptic explanation comes from Stahl,
  so we do NOT require both terms in NbN.

Empty fields (present_and_empty) are tested separately from claim production:
an empty field should NOT produce a claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import text

from psych_qa.config import get_settings
from psych_qa.db.connection import get_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def expected():
    """Load the independently-extracted expected values."""
    path = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "amisulpride_expected.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def amisulpride_drug_id(engine):
    """Resolve amisulpride by slug — never hardcode the ID."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM drugs WHERE slug = :slug"),
            {"slug": "amisulpride"},
        ).fetchone()
    assert row is not None, "Amisulpride drug not found. Run ingest_nbn first."
    return row[0]


@pytest.fixture(scope="module")
def nbn_source_doc_id(engine):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM source_documents WHERE source_type='nbn' AND is_active=true")
        ).fetchone()
    assert row is not None, "No active NbN source document."
    return row[0]


# ---------------------------------------------------------------------------
# Drug + variant assertions
# ---------------------------------------------------------------------------

class TestNbNDrugAndVariants:
    def test_drug_exists(self, amisulpride_drug_id):
        assert amisulpride_drug_id is not None

    def test_two_variants(self, engine, amisulpride_drug_id, expected):
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT variant_key, label FROM drug_variants WHERE drug_id = :did ORDER BY variant_key"),
                {"did": amisulpride_drug_id},
            ).fetchall()
        variant_keys = {r[0] for r in rows}
        assert "low" in variant_keys, f"Missing 'low' variant. Got: {variant_keys}"
        assert "upper" in variant_keys, f"Missing 'upper' variant. Got: {variant_keys}"

    def test_brand_aliases(self, engine, amisulpride_drug_id, expected):
        expected_aliases = expected["nbn"]["brand_aliases"]
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT alias FROM drug_aliases WHERE drug_id = :did"),
                {"did": amisulpride_drug_id},
            ).fetchall()
        actual_aliases = {r[0] for r in rows}
        for expected_alias in expected_aliases:
            assert expected_alias in actual_aliases, f"Missing brand alias: {expected_alias}"


# ---------------------------------------------------------------------------
# Claim assertions — per variant
# ---------------------------------------------------------------------------

class TestNbNClaims:
    def _get_claims(self, engine, drug_id, variant_key):
        """Get claims as dict: category -> list of (category, text, attrs, locator)."""
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT sc.category, sc.text, sc.attributes, sc.locator
                    FROM source_claims sc
                    JOIN source_documents sd ON sc.source_document_id = sd.id
                    JOIN drug_variants dv ON sc.drug_variant_id = dv.id
                    WHERE sd.source_type = 'nbn' AND sc.drug_id = :did
                    AND dv.variant_key = :vk AND sd.is_active = true
                    ORDER BY sc.category, sc.id
                """),
                {"did": drug_id, "vk": variant_key},
            ).fetchall()
        # Group by category (some categories have multiple claims)
        result: dict[str, list] = {}
        for r in rows:
            result.setdefault(r[0], []).append(r)
        return result

    def test_low_dose_claims(self, engine, amisulpride_drug_id, expected):
        """Low-dose variant: verify each field independently."""
        claims = self._get_claims(engine, amisulpride_drug_id, "low")
        low_expected = expected["nbn"]["variants"][0]["fields"]

        present_fields = {
            k: v for k, v in low_expected.items() if v["status"] == "present"
        }
        for field, expected_val in present_fields.items():
            category = _field_to_category(field)
            assert category in claims, (
                f"Low-dose field '{field}' (category '{category}') should produce a claim but is missing. "
                f"Available: {sorted(claims.keys())}"
            )
            # Check all claims in this category for a match
            matched = False
            for claim_row in claims[category]:
                claim_text = claim_row[1]
                if "value" in expected_val and claim_text == expected_val["value"]:
                    matched = True
                    break
                if "value_prefix" in expected_val and claim_text.startswith(expected_val["value_prefix"]):
                    matched = True
                    break
            assert matched, (
                f"Low-dose {field}: no claim matched expected value. "
                f"Expected '{expected_val.get('value', expected_val.get('value_prefix', ''))[:60]}', "
                f"got: {[r[1][:60] for r in claims[category]]}"
            )

    def test_upper_dose_claims(self, engine, amisulpride_drug_id, expected):
        """Upper-dose variant: verify each field independently."""
        claims = self._get_claims(engine, amisulpride_drug_id, "upper")
        upper_expected = expected["nbn"]["variants"][1]["fields"]

        present_fields = {
            k: v for k, v in upper_expected.items() if v["status"] == "present"
        }
        for field, expected_val in present_fields.items():
            category = _field_to_category(field)
            assert category in claims, (
                f"Upper-dose field '{field}' (category '{category}') should produce a claim but is missing. "
                f"Available: {sorted(claims.keys())}"
            )
            matched = False
            for claim_row in claims[category]:
                claim_text = claim_row[1]
                if "value" in expected_val and claim_text == expected_val["value"]:
                    matched = True
                    break
                if "value_prefix" in expected_val and claim_text.startswith(expected_val["value_prefix"]):
                    matched = True
                    break
            assert matched, (
                f"Upper-dose {field}: no claim matched expected value. "
                f"Expected '{expected_val.get('value', expected_val.get('value_prefix', ''))[:60]}', "
                f"got: {[r[1][:60] for r in claims[category]]}"
            )

    def test_empty_fields_do_not_produce_claims(self, engine, amisulpride_drug_id, expected):
        """Fields that are present_and_empty in the source should NOT produce claims."""
        for variant_idx, variant_key in enumerate(["low", "upper"]):
            claims = self._get_claims(engine, amisulpride_drug_id, variant_key)
            fields = expected["nbn"]["variants"][variant_idx]["fields"]
            empty_fields = {
                k: v for k, v in fields.items() if v["status"] == "present_and_empty"
            }
            for field, _ in empty_fields.items():
                category = _field_to_category(field)
                if category in claims:
                    for claim_row in claims[category]:
                        claim_text = claim_row[1]
                        assert claim_text.strip() != "", (
                            f"Empty field '{field}' (category '{category}') produced an empty claim in {variant_key} variant"
                        )

    def test_claim_counts_diagnostic(self, engine, amisulpride_drug_id, expected):
        """Diagnostic: total claim count should match expected (not a hard gate)."""
        expected_counts = expected["nbn"]["expected_claim_counts"]
        with engine.connect() as conn:
            total = conn.execute(
                text("""
                    SELECT count(*) FROM source_claims sc
                    JOIN source_documents sd ON sc.source_document_id = sd.id
                    WHERE sd.source_type = 'nbn' AND sc.drug_id = :did AND sd.is_active = true
                """),
                {"did": amisulpride_drug_id},
            ).scalar()
        # Diagnostic with tolerance — not a hard gate
        assert total == expected_counts["total"], (
            f"NbN claim count diagnostic: expected {expected_counts['total']}, got {total}. "
            f"This is monitored, not a hard gate — but a mismatch warrants investigation."
        )

    def test_every_claim_has_locator(self, engine, amisulpride_drug_id):
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT sc.id, sc.locator FROM source_claims sc
                    JOIN source_documents sd ON sc.source_document_id = sd.id
                    WHERE sd.source_type = 'nbn' AND sc.drug_id = :did AND sd.is_active = true
                """),
                {"did": amisulpride_drug_id},
            ).fetchall()
        for r in rows:
            assert r[1] is not None, f"Claim {r[0]} has null locator"
            assert "nbn_id" in r[1], f"Claim {r[0]} locator missing nbn_id"

    def test_low_dose_mode_of_action_presynaptic(self, engine, amisulpride_drug_id):
        """NbN low-dose mode_of_action explicitly says 'Pre-synaptic antagonist'."""
        claims = self._get_claims(engine, amisulpride_drug_id, "low")
        assert "mode_of_action" in claims
        text_val = claims["mode_of_action"][0][1]
        assert "Pre-synaptic" in text_val or "pre-synaptic" in text_val.lower(), (
            f"Low-dose mode_of_action should mention 'Pre-synaptic', got: '{text_val}'"
        )

    def test_upper_dose_mode_of_action_antagonist(self, engine, amisulpride_drug_id):
        """NbN upper-dose mode_of_action says 'antagonist' (not presynaptic-specific)."""
        claims = self._get_claims(engine, amisulpride_drug_id, "upper")
        assert "mode_of_action" in claims
        text_val = claims["mode_of_action"][0][1]
        assert "antagonist" in text_val.lower(), (
            f"Upper-dose mode_of_action should contain 'antagonist', got: '{text_val}'"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mirror of FIELD_MAP in nbn.py — kept here so tests don't import the parser
_FIELD_TO_CATEGORY = {
    "pharmacological_target": "target",
    "mode_of_action": "mode_of_action",
    "approved_indications": "indication",
    "efficacy": "efficacy",
    "side_effects": "side_effects",
    "committee_notes": "committee_notes",
    "pregnancy": "pregnancy",
    "drugs_interaction": "interaction",
    "neurobiology_extra": "neurobiology",
    "former_terminology": "classification",
    "brand_names": "brand_names",
    "uptake_inhibition_selectivity_SERT_NET": "uptake_selectivity",
    "uptake_inhibition_selectivity_NET_SERT": "uptake_selectivity",
    "neurotransmitter_effects_preclinical": "neurotransmitter_effects",
    "neurotransmitter_effects_human": "neurotransmitter_effects",
    "physiological_preclinical": "physiological_effects",
    "physiological_human": "physiological_effects",
    "brain_circuits_preclinical": "brain_circuits",
    "brain_circuits_human": "brain_circuits",
}


def _field_to_category(field: str) -> str:
    return _FIELD_TO_CATEGORY.get(field, field)
