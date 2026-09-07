"""NbN (Neuroscience-based Nomenclature) loader.

Loads normalized NbN JSON into the database:
- drugs (canonical, slug-keyed)
- drug_variants (base/low/upper)
- source_variant_mappings (nbn_id → canonical variant)
- drug_aliases (brand names only; former_terminology is a classification claim)
- source_claims (one per atomic field value, source-preserving)

Idempotent within a version: upserts keyed by
(source_document_id, drug_id, COALESCE(drug_variant_id, 0), category, claim_hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.connection import get_session
from ..db.tables import (
    Drug,
    DrugAlias,
    DrugVariant,
    IngestionRun,
    SourceClaim,
    SourceDocument,
    SourceVariantMapping,
)

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    """Normalize a drug name to a slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _normalize_alias(alias: str) -> str:
    """Normalize an alias for lookup."""
    return _slug(alias)


def _claim_hash(text_val: str, field_name: str = "") -> str:
    """Hash a claim text (+ field name) for uniqueness.

    Including field_name prevents collisions when two different NbN fields
    (e.g. brain_circuits_preclinical and brain_circuits_human) map to the
    same category with identical text.
    """
    normalized = re.sub(r"\s+", " ", text_val.strip().lower())
    key = f"{field_name}:{normalized}" if field_name else normalized
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _content_hash(file_path: Path) -> str:
    """SHA-256 of file bytes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# NbN fields → claim categories
FIELD_MAP: dict[str, str] = {
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


def _variant_key(dose: str | None) -> str:
    """Map NbN dose to canonical variant key."""
    if dose is None:
        return "base"
    return dose  # "low" or "upper"


def ingest_nbn(file_path: Path | None = None) -> dict[str, Any]:
    """Load NbN JSON into the database.

    Returns a summary dict with counts.
    """
    settings = get_settings()
    if file_path is None:
        file_path = settings.nbn_json_path

    if not file_path.exists():
        raise FileNotFoundError(f"NbN JSON not found: {file_path}")

    # Load JSON
    with open(file_path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    drugs_data = data.get("drugs", {})

    content_hash = _content_hash(file_path)
    version_label = f"NbN fetched {meta.get('fetched_at', 'unknown')} hash={content_hash[:8]}"

    session = get_session()
    try:
        # Upsert source document (unique on source_type + content_hash)
        doc = session.execute(
            sqltext(
                "SELECT id FROM source_documents WHERE source_type = :st AND content_hash = :ch"
            ),
            {"st": "nbn", "ch": content_hash},
        ).fetchone()

        if doc:
            source_doc_id = doc[0]
            logger.info(f"Reusing existing source_document id={source_doc_id}")
        else:
            sd = SourceDocument(
                source_type="nbn",
                scope_key=None,
                content_hash=content_hash,
                version_label=version_label,
                is_active=False,
            )
            session.add(sd)
            session.flush()
            source_doc_id = sd.id

        # Create ingestion run
        run = IngestionRun(
            source_document_id=source_doc_id,
            source="nbn",
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(run)
        session.flush()
        run_id = run.id

        records_seen = 0
        records_written = 0
        warnings: list[str] = []

        for drug_name, drug_info in drugs_data.items():
            records_seen += 1
            slug = _slug(drug_name)

            # Upsert drug
            existing = session.execute(
                sqltext("SELECT id FROM drugs WHERE slug = :slug"),
                {"slug": slug},
            ).fetchone()

            if existing:
                drug_id = existing[0]
            else:
                drug = Drug(
                    canonical_name=drug_name,
                    slug=slug,
                    gtopdb_ligand_id=drug_info.get("gtopdb_ligand_id"),
                    constituents=drug_info.get("constituents"),
                )
                session.add(drug)
                session.flush()
                drug_id = drug.id

            # Process variants
            variants = drug_info.get("variants", [])
            if not variants:
                # Create a base variant if none specified
                variants = [{"nbn_id": None, "label": drug_name, "dose": None, "nbn_raw": {}, "normalized": {}}]

            for variant in variants:
                dose = variant.get("dose")
                vkey = _variant_key(dose)
                vlabel = variant.get("label", drug_name)

                # Upsert drug_variant
                existing_v = session.execute(
                    sqltext(
                        "SELECT id FROM drug_variants WHERE drug_id = :did AND variant_key = :vk"
                    ),
                    {"did": drug_id, "vk": vkey},
                ).fetchone()

                if existing_v:
                    variant_id = existing_v[0]
                else:
                    dv = DrugVariant(drug_id=drug_id, variant_key=vkey, label=vlabel)
                    session.add(dv)
                    session.flush()
                    variant_id = dv.id

                # Source variant mapping
                nbn_id = variant.get("nbn_id")
                if nbn_id is not None:
                    existing_m = session.execute(
                        sqltext(
                            "SELECT id FROM source_variant_mappings "
                            "WHERE source_document_id = :sdid AND external_variant_id = :eid"
                        ),
                        {"sdid": source_doc_id, "eid": str(nbn_id)},
                    ).fetchone()
                    if not existing_m:
                        svm = SourceVariantMapping(
                            source_document_id=source_doc_id,
                            external_variant_id=str(nbn_id),
                            drug_variant_id=variant_id,
                            source_label=vlabel,
                        )
                        session.add(svm)

                # Emit claims from nbn_raw fields
                nbn_raw = variant.get("nbn_raw", {})
                normalized = variant.get("normalized", {})

                # Extract normalized attributes
                domains = normalized.get("domains", [])
                moa = normalized.get("modes_of_action", {})
                norm_target = domains[0]["value"] if domains else None
                norm_actions = moa.get("values", []) if moa else []

                for field_name, category in FIELD_MAP.items():
                    value = nbn_raw.get(field_name, "")
                    if not value or not str(value).strip():
                        continue

                    text_val = str(value).strip()
                    chash = _claim_hash(text_val, field_name)

                    # Build attributes
                    attrs: dict[str, Any] = {}
                    if category == "target" and norm_target:
                        attrs["target"] = norm_target
                    if category == "mode_of_action" and norm_actions:
                        attrs["actions"] = norm_actions
                    if dose:
                        attrs["dose_band"] = dose

                    # Check for existing claim (idempotent)
                    existing_c = session.execute(
                        sqltext(
                            "SELECT id FROM source_claims "
                            "WHERE source_document_id = :sdid AND drug_id = :did "
                            "AND COALESCE(drug_variant_id, 0) = :vid "
                            "AND category = :cat AND claim_hash = :ch"
                        ),
                        {
                            "sdid": source_doc_id,
                            "did": drug_id,
                            "vid": variant_id or 0,
                            "cat": category,
                            "ch": chash,
                        },
                    ).fetchone()

                    if not existing_c:
                        claim = SourceClaim(
                            source_document_id=source_doc_id,
                            drug_id=drug_id,
                            drug_variant_id=variant_id,
                            category=category,
                            text=text_val,
                            attributes=attrs,
                            claim_hash=chash,
                            locator={"source": "nbn", "nbn_id": nbn_id, "variant_label": vlabel},
                            claim_type="source_raw",
                        )
                        session.add(claim)
                        records_written += 1

                # Brand names → drug_aliases (NOT former_terminology)
                brand_names = nbn_raw.get("brand_names", "")
                if brand_names and brand_names.strip():
                    for brand in re.split(r"[,;]", brand_names):
                        brand = brand.strip()
                        if not brand:
                            continue
                        norm = _normalize_alias(brand)
                        if not norm:
                            continue
                        # Check if alias already exists for this drug
                        existing_a = session.execute(
                            sqltext(
                                "SELECT id FROM drug_aliases "
                                "WHERE drug_id = :did AND normalized_alias = :na"
                            ),
                            {"did": drug_id, "na": norm},
                        ).fetchone()
                        if not existing_a:
                            alias = DrugAlias(
                                drug_id=drug_id,
                                alias=brand,
                                normalized_alias=norm,
                                alias_type="brand",
                                source_document_id=source_doc_id,
                            )
                            session.add(alias)

            if records_seen % 50 == 0:
                logger.info(f"Processed {records_seen}/{len(drugs_data)} drugs")
                session.flush()

        # Update run status
        run.status = "completed"
        run.records_seen = records_seen
        run.records_written = records_written
        run.warnings = warnings
        run.completed_at = datetime.utcnow()

        # Atomically activate this version
        session.execute(
            sqltext(
                "UPDATE source_documents SET is_active = false "
                "WHERE source_type = 'nbn' AND scope_key IS NULL"
            )
        )
        session.execute(
            sqltext(
                "UPDATE source_documents SET is_active = true WHERE id = :id"
            ),
            {"id": source_doc_id},
        )

        session.commit()

        logger.info(
            f"NbN ingestion complete: {records_seen} drugs, {records_written} claims, "
            f"{len(warnings)} warnings"
        )
        return {
            "source_document_id": source_doc_id,
            "run_id": run_id,
            "records_seen": records_seen,
            "records_written": records_written,
            "warnings": warnings,
        }

    except Exception as e:
        session.rollback()
        if "run_id" in locals():
            run.status = "failed"
            run.completed_at = datetime.utcnow()
            try:
                session.commit()
            except Exception:
                session.rollback()
        logger.error(f"NbN ingestion failed: {e}")
        raise
    finally:
        session.close()
