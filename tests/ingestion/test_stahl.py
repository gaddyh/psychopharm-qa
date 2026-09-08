"""Tests for Stahl ingestion fidelity — Amisulpride.

Integration tests against the locally ingested real Stahl source.
Expected values come from data/samples/amisulpride_expected.json, which was
extracted independently from the source PDF (not by running the parser).

Hard gates:
- Required sections present
- Critical claim fingerprints present (by text content, not DB ID)
- FDA-bold interpretation correct
- No parse artifacts for the monograph pages
- Every claim has printed + physical page numbers

Diagnostic (not a hard gate):
- Total claim count (monitored with tolerance for bullet regrouping)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import text

from psych_qa.db.connection import get_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def expected():
    path = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "amisulpride_expected.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def amisulpride_drug_id(engine):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM drugs WHERE slug = :slug"),
            {"slug": "amisulpride"},
        ).fetchone()
    assert row is not None, "Amisulpride drug not found."
    return row[0]


@pytest.fixture(scope="module")
def stahl_claims(engine, amisulpride_drug_id):
    """All Stahl claims for amisulpride as list of dicts."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT sc.id, sc.category, sc.text, sc.attributes, sc.locator
                FROM source_claims sc
                JOIN source_documents sd ON sc.source_document_id = sd.id
                WHERE sd.source_type = 'stahl' AND sc.drug_id = :did AND sd.is_active = true
                ORDER BY sc.id
            """),
            {"did": amisulpride_drug_id},
        ).fetchall()
    return [
        {"id": r[0], "category": r[1], "text": r[2], "attributes": r[3], "locator": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Section coverage
# ---------------------------------------------------------------------------

class TestStahlSections:
    def test_all_major_sections_present(self, stahl_claims, expected):
        """All expected major sections should have at least one claim."""
        expected_sections = set(expected["stahl"]["expected_sections"])
        actual_categories = {c["category"] for c in stahl_claims}

        # Map major section names to claim categories that should exist
        section_to_categories = {
            "THERAPEUTICS": ["therapeutics", "indication", "efficacy", "treatment_failure", "target_symptoms"],
            "SIDE EFFECTS": ["side_effects", "side_effect_mechanism", "side_effect_management", "sedation", "weight_gain"],
            "DOSING AND USE": ["dosing", "dosage_range", "dosing_instructions", "dosing_tips", "dosage_forms", "pharmacokinetics"],
            "SPECIAL POPULATIONS": ["pregnancy", "breast_feeding", "renal_impairment", "hepatic_impairment", "cardiac_impairment", "elderly", "pediatric"],
            "THE ART OF PSYCHOPHARMACOLOGY": ["pearls", "clinical_pearls", "potential_advantages", "potential_disadvantages"],
        }

        for section, cats in section_to_categories.items():
            found = any(cat in actual_categories for cat in cats)
            assert found, (
                f"Major section '{section}' has no corresponding claims. "
                f"Expected one of categories: {cats}, available: {sorted(actual_categories)}"
            )


# ---------------------------------------------------------------------------
# FDA-bold interpretation
# ---------------------------------------------------------------------------

class TestStahlFDABold:
    def test_fda_bold_claims(self, stahl_claims, expected):
        """FDA-bold detection: verify each expected FDA claim."""
        fda_expected = expected["stahl"]["fda_bold_claims"]
        indication_claims = [c for c in stahl_claims if c["category"] == "indication"]

        for expected_fda in fda_expected:
            # Find the matching claim by text prefix
            matching = [
                c for c in indication_claims
                if c["text"].startswith(expected_fda["text_prefix"])
            ]
            assert len(matching) >= 1, (
                f"No indication claim starting with '{expected_fda['text_prefix']}'"
            )
            for claim in matching:
                actual_fda = claim["attributes"].get("fda_approved")
                assert actual_fda == expected_fda["fda_approved"], (
                    f"FDA-bold mismatch for '{claim['text'][:50]}': "
                    f"expected fda_approved={expected_fda['fda_approved']}, got {actual_fda}"
                )


# ---------------------------------------------------------------------------
# Critical claim fingerprints
# ---------------------------------------------------------------------------

class TestStahlCriticalClaims:
    def test_critical_fingerprints(self, stahl_claims, expected):
        """Every critical claim fingerprint must be present in the DB."""
        fingerprints = expected["stahl"]["critical_claim_fingerprints"]
        for fp in fingerprints:
            matching = [
                c for c in stahl_claims
                if c["category"] == fp["category"]
                and fp["text_contains"].lower() in c["text"].lower()
            ]
            assert len(matching) >= 1, (
                f"Critical fingerprint not found: category='{fp['category']}', "
                f"text contains '{fp['text_contains']}'"
            )


# ---------------------------------------------------------------------------
# Page numbers
# ---------------------------------------------------------------------------

class TestStahlPageNumbers:
    def test_every_claim_has_pages(self, stahl_claims):
        for claim in stahl_claims:
            loc = claim["locator"]
            assert loc is not None, f"Claim {claim['id']} has null locator"
            assert "printed_book_page" in loc, (
                f"Claim {claim['id']} missing printed_book_page in locator"
            )
            assert "physical_pdf_page" in loc, (
                f"Claim {claim['id']} missing physical_pdf_page in locator"
            )

    def test_page_range(self, stahl_claims, expected):
        page_range = expected["stahl"]["page_range"]
        for claim in stahl_claims:
            phys = claim["locator"]["physical_pdf_page"]
            printed = claim["locator"]["printed_book_page"]
            assert page_range["physical_pdf_start"] <= phys <= page_range["physical_pdf_end"], (
                f"Claim {claim['id']} physical page {phys} outside expected range "
                f"{page_range['physical_pdf_start']}-{page_range['physical_pdf_end']}"
            )
            assert page_range["printed_book_start"] <= printed <= page_range["printed_book_end"], (
                f"Claim {claim['id']} printed page {printed} outside expected range "
                f"{page_range['printed_book_start']}-{page_range['printed_book_end']}"
            )


# ---------------------------------------------------------------------------
# Parse artifacts
# ---------------------------------------------------------------------------

class TestStahlParseArtifacts:
    def test_no_parse_artifacts(self, engine, expected):
        """No unparsed pages for the amisulpride monograph."""
        page_range = expected["stahl"]["page_range"]
        with engine.connect() as conn:
            count = conn.execute(
                text("""
                    SELECT count(*) FROM parse_artifacts
                    WHERE physical_pdf_page >= :start AND physical_pdf_page <= :end
                """),
                {
                    "start": page_range["physical_pdf_start"],
                    "end": page_range["physical_pdf_end"],
                },
            ).scalar()
        assert count == 0, f"{count} parse artifacts found for amisulpride pages"


# ---------------------------------------------------------------------------
# Diagnostic: total claim count (monitored, not a hard gate)
# ---------------------------------------------------------------------------

class TestStahlDiagnosticCounts:
    def test_total_claim_count_diagnostic(self, stahl_claims, expected):
        """Diagnostic: total claim count should be within tolerance.

        This is NOT a hard gate — bullet regrouping can legitimately change
        the count. It's monitored to detect unexpected changes.
        """
        expected_total = expected["stahl"]["diagnostic_total_claims"]
        tolerance = expected["stahl"]["total_claims_tolerance"]
        actual = len(stahl_claims)
        # Log the actual count for visibility
        print(f"\n  Stahl total claims: {actual} (expected ~{expected_total}, tolerance ±{tolerance})")
        assert abs(actual - expected_total) <= tolerance, (
            f"Stahl claim count diagnostic: {actual} is outside tolerance "
            f"{expected_total}±{tolerance}. This is monitored, not a hard gate — "
            f"but a large change warrants investigation."
        )

    def test_category_count(self, stahl_claims, expected):
        """Minimum category count — hard gate."""
        min_cats = expected["stahl"]["expected_category_count_min"]
        actual_cats = len({c["category"] for c in stahl_claims})
        assert actual_cats >= min_cats, (
            f"Only {actual_cats} categories, expected >= {min_cats}"
        )
