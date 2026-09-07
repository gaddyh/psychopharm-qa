"""Stahl PDF parser → structured monographs → atomic claims.

Staged pipeline:
  PDF bytes → pages.layout.jsonl (PyMuPDF dict extraction)
            → monographs.jsonl (boundary detection + section parsing)
            → claims.jsonl (each bullet = one claim)
            → database

Key design (verified against amisulpride monograph, PDF p.33-40):
- PyMuPDF get_text('dict') for span-level font/bold recovery.
- Two-column layout: left col x≈42.8, right col x≈212.9.
- Font roles (VERIFIED):
    HelveticaLTStd-BoldCond  = major section headers (ALL CAPS) AND bold body text
    HelveticaLTStd-BlkCond   = sub-section headers (Title Case)
    HelveticaLTStd-Cond      = regular body text
    HelveticaLTStd-CondObl   = italic notes
- Monograph boundaries: <digits>\n<UPPERCASE DRUG> = new monograph;
  (continued) DRUGNAME = append to current.
- Bullets start with "•"; continuation lines (no "•") merge into previous bullet.
- Bold = FDA approved: bold_ratio >= 0.85 threshold.
- Each bullet = one SourceClaim with claim_hash.
- Both physical_pdf_page and printed_book_page stored.
- Failed pages → parse_artifacts, NOT source_claims.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
from sqlalchemy import text as sqltext

from ..config import get_settings
from ..db.connection import get_session
from ..db.tables import (
    Drug,
    DrugAlias,
    IngestionRun,
    ParseArtifact,
    SourceClaim,
    SourceDocument,
)

logger = logging.getLogger(__name__)

BOLD_THRESHOLD = 0.85

MAJOR_SECTIONS = {
    "THERAPEUTICS",
    "SIDE EFFECTS",
    "DOSING AND USE",
    "SPECIAL POPULATIONS",
    "THE ART OF PSYCHOPHARMACOLOGY",
    "THE ART OF SWITCHING",
    "SUGGESTED READING",
}

# Column boundary (x < this = left column, >= this = right column)
COLUMN_BOUNDARY = 150.0


def _slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _claim_hash(text_val: str, section: str = "", subsection: str = "", bullet_index: int = 0) -> str:
    normalized = re.sub(r"\s+", " ", text_val.strip().lower())
    key = f"{section}:{subsection}:{bullet_index}:{normalized}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _content_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_bold(font_name: str) -> bool:
    return "Bold" in font_name or "Blk" in font_name


# ---------------------------------------------------------------------------
# Stage 1: PDF → page-layout (line-level extraction with column detection)
# ---------------------------------------------------------------------------


def extract_pages_layout(
    pdf_path: Path, page_range: tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    """Extract line-level layout data from PDF pages.

    Returns pages with lines grouped by column (left first, then right).
    Each line has: text, font, bold, x0, y0, bbox, spans.
    """
    doc = pymupdf.open(str(pdf_path))
    pages = []

    if page_range is None:
        page_indices = range(len(doc))
    else:
        page_indices = range(page_range[0], min(page_range[1], len(doc)))

    for i in page_indices:
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]

        all_lines: list[dict[str, Any]] = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                # Determine font (most common by char count)
                font_counts: dict[str, int] = {}
                for s in spans:
                    font_counts[s["font"]] = font_counts.get(s["font"], 0) + len(s["text"])
                font = max(font_counts, key=font_counts.get) if font_counts else "unknown"
                x0 = min(s["bbox"][0] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                all_lines.append(
                    {
                        "text": text,
                        "font": font,
                        "bold": _is_bold(font),
                        "x0": round(x0, 1),
                        "y0": round(y0, 1),
                        "spans": [
                            {
                                "text": s["text"],
                                "font": s["font"],
                                "bold": _is_bold(s["font"]),
                                "bbox": [round(c, 1) for c in s["bbox"]],
                            }
                            for s in spans
                        ],
                    }
                )

        # Sort lines: left column first (by y), then right column (by y)
        left = [l for l in all_lines if l["x0"] < COLUMN_BOUNDARY]
        right = [l for l in all_lines if l["x0"] >= COLUMN_BOUNDARY]
        left.sort(key=lambda l: l["y0"])
        right.sort(key=lambda l: l["y0"])
        ordered = left + right

        pages.append({"page_index": i, "lines": ordered})

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Stage 2: page-layout → monographs
# ---------------------------------------------------------------------------


def _detect_monograph_start(lines: list[dict[str, Any]]) -> str | None:
    """Detect if this page starts a new monograph.

    The drug name heading is at the top of the page (low y), in BoldCond font,
    ALL CAPS, and NOT a continuation. The printed page number may be at the
    bottom of the page. The drug name may be in either column, so we search
    all lines sorted by y for a heading near the top.
    """
    # Sort all lines by y to find the top-most content
    by_y = sorted(lines, key=lambda l: l["y0"])
    for line in by_y[:5]:
        text = line["text"].strip()
        if not text or text.isdigit():
            continue
        # Must be ALL CAPS
        if text.upper() != text:
            continue
        # Must not be a continuation
        if "continued" in text.lower():
            continue
        # Must be bold (BoldCond font = drug name heading)
        if not line["bold"]:
            continue
        # Must not be a known major section header
        if text in MAJOR_SECTIONS:
            continue
        # Must be a plausible drug name (letters, at least 3 chars)
        if len(text) < 3 or not any(c.isalpha() for c in text):
            continue
        return text
    return None


def _detect_continuation(lines: list[dict[str, Any]]) -> str | None:
    """Detect if this page is a continuation of a monograph."""
    by_y = sorted(lines, key=lambda l: l["y0"])
    for line in by_y[:5]:
        text = line["text"].strip()
        if "continued" in text.lower():
            m = re.match(r"^\(continued\)\s+([A-Z][A-Z\s\-]+)", text)
            if m:
                return m.group(1).strip()
            m = re.match(r"^([A-Z][A-Z\s\-]+)\s*\(continued\)", text)
            if m:
                return m.group(1).strip()
            if text == "(continued)":
                return ""
    return None


def _detect_printed_page(lines: list[dict[str, Any]]) -> int | None:
    for l in lines:
        t = l["text"].strip()
        if t.isdigit() and len(t) <= 4:
            return int(t)
    return None


def _is_major_section(line: dict[str, Any]) -> bool:
    """Major section header: BoldCond font, ALL CAPS, no bullet, in MAJOR_SECTIONS."""
    text = line["text"].strip()
    if "•" in text:
        return False
    if text.upper() != text:
        return False
    if "BoldCond" not in line["font"]:
        return False
    # Check against known sections (fuzzy)
    for ms in MAJOR_SECTIONS:
        if ms in text.upper():
            return True
    return False


def _is_subsection_header(line: dict[str, Any]) -> bool:
    """Sub-section header: BlkCond font, Title Case, no bullet, short."""
    text = line["text"].strip()
    if "•" in text:
        return False
    if "BlkCond" not in line["font"]:
        return False
    if len(text) > 80:
        return False
    if text.upper() == text:
        return False  # all caps = major section
    return True


def _is_bullet_start(line: dict[str, Any]) -> bool:
    """Check if a line starts a new bullet (begins with •)."""
    return line["text"].strip().startswith("•")


def _clean_bullet_text(text: str) -> str:
    """Clean bullet text: remove leading bullet, normalize whitespace."""
    t = text.strip()
    if t.startswith("•"):
        t = t[1:].strip()
    # Also remove leading tab after bullet
    t = t.lstrip("\t").strip()
    return re.sub(r"\s+", " ", t)


def parse_monographs(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse page-layout data into monograph structures.

    Each monograph:
    {
        "drug_name": "AMISULPRIDE",
        "start_pdf_page": 33,
        "end_pdf_page": 40,
        "printed_pages": [17, 18, ...],
        "sections": [
            {"name": "THERAPEUTICS", "subsections": [
                {"name": "How the Drug Works", "bullets": [...],
                 "pdf_page": 33, "printed_page": 17}
            ]}
        ]
    }
    """
    monographs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_section: str | None = None
    current_subsection: dict[str, Any] | None = None
    current_bullet_text: str = ""
    current_bullet_spans: list[dict] = []
    current_bullet_bold_chars = 0
    current_bullet_total_chars = 0
    current_bullet_pdf_page: int | None = None
    current_bullet_printed_page: int | None = None

    def _flush_bullet():
        nonlocal current_bullet_text, current_bullet_spans
        nonlocal current_bullet_bold_chars, current_bullet_total_chars
        nonlocal current_bullet_pdf_page, current_bullet_printed_page

        if not current_bullet_text or not current_subsection:
            current_bullet_text = ""
            current_bullet_spans = []
            current_bullet_bold_chars = 0
            current_bullet_total_chars = 0
            return

        bold_ratio = (
            current_bullet_bold_chars / current_bullet_total_chars
            if current_bullet_total_chars > 0
            else 0.0
        )
        bullet = {
            "text": re.sub(r"\s+", " ", current_bullet_text.strip()),
            "bold_ratio": round(bold_ratio, 3),
            "pdf_page": current_bullet_pdf_page,
            "printed_page": current_bullet_printed_page,
            "bboxes": [s["bbox"] for s in current_bullet_spans if s.get("text", "").strip()],
        }
        current_subsection["bullets"].append(bullet)

        current_bullet_text = ""
        current_bullet_spans = []
        current_bullet_bold_chars = 0
        current_bullet_total_chars = 0

    def _ensure_subsection(name: str, pdf_page: int, printed_page: int | None):
        nonlocal current_subsection
        if current and current["sections"]:
            # Check if last subsection has same name
            subs = current["sections"][-1]["subsections"]
            if subs and subs[-1]["name"] == name:
                current_subsection = subs[-1]
                return
        current_subsection = {
            "name": name,
            "bullets": [],
            "pdf_page": pdf_page,
            "printed_page": printed_page,
        }
        current["sections"][-1]["subsections"].append(current_subsection)

    for page in pages:
        lines = page["lines"]
        pdf_page = page["page_index"]
        printed_page = _detect_printed_page(lines)

        # Check for new monograph
        new_name = _detect_monograph_start(lines)
        if new_name:
            _flush_bullet()
            if current:
                monographs.append(current)
            current = {
                "drug_name": new_name,
                "start_pdf_page": pdf_page,
                "end_pdf_page": pdf_page,
                "printed_pages": [],
                "sections": [],
            }
            current_section = None
            current_subsection = None
        else:
            cont_name = _detect_continuation(lines)
            if cont_name and current:
                if cont_name.upper() != current["drug_name"].upper():
                    logger.warning(
                        f"Page {pdf_page}: continuation '{cont_name}' "
                        f"doesn't match current '{current['drug_name']}'"
                    )
            elif not current:
                continue

        if current is None:
            continue

        current["end_pdf_page"] = pdf_page
        if printed_page is not None:
            current["printed_pages"].append(printed_page)

        for line in lines:
            text = line["text"].strip()
            if not text:
                continue

            # Skip page numbers
            if text.isdigit():
                continue
            # Skip continuation markers
            if "(continued)" in text.lower() and len(text) < 40:
                continue
            # Skip DOI lines
            if "doi.org" in text or "Published online" in text:
                continue
            # Skip drug name heading (already captured)
            if text == current["drug_name"] and line["bold"]:
                continue

            # Major section header
            if _is_major_section(line):
                _flush_bullet()
                section_name = text.upper().strip()
                for ms in MAJOR_SECTIONS:
                    if ms in section_name:
                        section_name = ms
                        break
                current_section = section_name
                current_subsection = None
                current["sections"].append({"name": current_section, "subsections": []})
                continue

            # Sub-section header
            if _is_subsection_header(line):
                _flush_bullet()
                sub_name = text.strip()
                _ensure_subsection(sub_name, pdf_page, printed_page)
                continue

            # Bullet or continuation
            if current_section is None:
                continue

            if _is_bullet_start(line):
                # Flush previous bullet
                _flush_bullet()
                # Start new bullet
                clean = _clean_bullet_text(text)
                current_bullet_text = clean
                current_bullet_pdf_page = pdf_page
                current_bullet_printed_page = printed_page
                for s in line["spans"]:
                    if s["text"].strip():
                        current_bullet_spans.append(s)
                        chars = len(s["text"].strip())
                        current_bullet_total_chars += chars
                        if s["bold"]:
                            current_bullet_bold_chars += chars
            else:
                # Continuation of previous bullet
                if current_bullet_text:
                    clean = _clean_bullet_text(text)
                    current_bullet_text += " " + clean
                    for s in line["spans"]:
                        if s["text"].strip():
                            current_bullet_spans.append(s)
                            chars = len(s["text"].strip())
                            current_bullet_total_chars += chars
                            if s["bold"]:
                                current_bullet_bold_chars += chars
                else:
                    # No current bullet — could be a label/value line
                    # (e.g. "Brands • Solian" where "Brands" is sub-section)
                    # Try to detect inline sub-section + value
                    if "BlkCond" in line["font"] and "•" in text:
                        _flush_bullet()
                        parts = text.split("•", 1)
                        sub_name = parts[0].strip()
                        _ensure_subsection(sub_name, pdf_page, printed_page)
                        # The rest after bullet is the first bullet
                        bullet_text = parts[1].strip()
                        if bullet_text:
                            current_bullet_text = bullet_text
                            current_bullet_pdf_page = pdf_page
                            current_bullet_printed_page = printed_page
                            for s in line["spans"]:
                                if s["text"].strip():
                                    current_bullet_spans.append(s)
                                    chars = len(s["text"].strip())
                                    current_bullet_total_chars += chars
                                    if s["bold"]:
                                        current_bullet_bold_chars += chars

    _flush_bullet()
    if current:
        monographs.append(current)

    return monographs


# ---------------------------------------------------------------------------
# Stage 3: monographs → claims
# ---------------------------------------------------------------------------


SECTION_CATEGORY_MAP: dict[str, str] = {
    "THERAPEUTICS": "therapeutics",
    "SIDE EFFECTS": "side_effects",
    "DOSING AND USE": "dosing",
    "SPECIAL POPULATIONS": "special_populations",
    "THE ART OF PSYCHOPHARMACOLOGY": "clinical_pearls",
    "THE ART OF SWITCHING": "switching",
    "SUGGESTED READING": "suggested_reading",
}

SUBSECTION_CATEGORY_MAP: dict[str, str] = {
    "Brands": "brands",
    "Generic?": "generic_status",
    "Class": "class",
    "Commonly Prescribed for": "indication",
    "How the Drug Works": "mechanism",
    "How Long Until It Works": "onset",
    "If It Works": "efficacy",
    "If It Doesn’t Work": "treatment_failure",
    "If It Doesn't Work": "treatment_failure",
    "Best Augmenting Combos for Partial Response or Treatment Resistance": "augmentation",
    "Tests": "monitoring",
    "How Drug Causes Side Effects": "side_effect_mechanism",
    "Notable Side Effects": "side_effects",
    "Life-Threatening or Dangerous Side Effects": "dangerous_side_effects",
    "Weight Gain": "weight_gain",
    "Sedation": "sedation",
    "What to Do About Side Effects": "side_effect_management",
    "Usual Dosage Range": "dosage_range",
    "Dosage Forms": "dosage_forms",
    "How to Dose": "dosing_instructions",
    "Dosing Tips": "dosing_tips",
    "Best Augmenting Agents for Side Effects": "side_effect_augmentation",
    "Overdose": "overdose",
    "Long-Term Use": "long_term_use",
    "Habit Forming?": "habit_forming",
    "How to Stop": "discontinuation",
    "Pharmacokinetics": "pharmacokinetics",
    "Drug Interactions": "interactions",
    "Other Warnings/Precautions": "warnings",
    "Do Not Use": "contraindications",
    "Renal Impairment": "renal_impairment",
    "Hepatic Impairment": "hepatic_impairment",
    "Cardiac Impairment": "cardiac_impairment",
    "Elderly": "elderly",
    "Children and Adolescents": "pediatric",
    "Pregnancy": "pregnancy",
    "Breast Feeding": "breast_feeding",
    "Potential Advantages": "potential_advantages",
    "Potential Disadvantages": "potential_disadvantages",
    "Primary Target Symptoms": "target_symptoms",
    "Pearls": "pearls",
}


def monographs_to_claims(
    monographs: list[dict[str, Any]], source_doc_id: int
) -> list[dict[str, Any]]:
    """Convert monographs to claim dicts ready for DB insertion."""
    claims = []
    for mono in monographs:
        drug_name = mono["drug_name"]
        slug = _slug(drug_name)

        for section in mono["sections"]:
            section_name = section["name"]
            for subsection in section["subsections"]:
                sub_name = subsection["name"]
                category = SUBSECTION_CATEGORY_MAP.get(sub_name) or SECTION_CATEGORY_MAP.get(
                    section_name, section_name.lower().replace(" ", "_")
                )

                for i, bullet in enumerate(subsection["bullets"]):
                    text_val = bullet["text"]
                    if not text_val:
                        continue
                    chash = _claim_hash(text_val, section_name, sub_name, i)

                    attrs: dict[str, Any] = {}
                    if category == "indication":
                        attrs["fda_approved"] = bullet["bold_ratio"] >= BOLD_THRESHOLD
                        attrs["approval_interpretation"] = "inferred_from_stahl_bold_format"
                        attrs["bold_ratio"] = bullet["bold_ratio"]

                    locator = {
                        "physical_pdf_page": bullet["pdf_page"],
                        "printed_book_page": bullet["printed_page"],
                        "section": section_name,
                        "subsection": sub_name,
                        "bullet_index": i,
                        "bounding_boxes": bullet["bboxes"],
                    }

                    claims.append(
                        {
                            "drug_slug": slug,
                            "drug_name": drug_name,
                            "category": category,
                            "text": text_val,
                            "attributes": attrs,
                            "claim_hash": chash,
                            "locator": locator,
                            "claim_type": "source_raw",
                        }
                    )

    return claims


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_monograph(mono: dict[str, Any]) -> list[str]:
    warnings = []
    if not mono["drug_name"]:
        warnings.append("No drug name")
    if not mono["sections"]:
        warnings.append("No sections detected")
    section_names = {s["name"] for s in mono["sections"]}
    expected = {"THERAPEUTICS", "SIDE EFFECTS", "DOSING AND USE"}
    missing = expected - section_names
    if missing:
        warnings.append(f"Missing major sections: {missing}")
    all_subs = set()
    for s in mono["sections"]:
        for ss in s["subsections"]:
            all_subs.add(ss["name"])
    if "How the Drug Works" not in all_subs:
        warnings.append("No mechanism subsection found")
    if "Commonly Prescribed for" not in all_subs:
        warnings.append("No indication subsection found")
    pages = mono["printed_pages"]
    if pages and len(pages) > 1:
        for i in range(1, len(pages)):
            if pages[i] != pages[i - 1] + 1:
                warnings.append(f"Page gap: {pages[i-1]} → {pages[i]}")
                break
    return warnings


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------


def ingest_stahl(
    pdf_path: Path | None = None,
    page_range: tuple[int, int] | None = None,
    drug_filter: str | None = None,
) -> dict[str, Any]:
    """Ingest Stahl PDF into the database.

    Args:
        pdf_path: Path to Stahl PDF. Defaults to config.
        page_range: (start, end) 0-based page indices. None = all pages.
        drug_filter: If set, only ingest monographs for this drug slug.

    Returns:
        Summary dict with counts.
    """
    settings = get_settings()
    if pdf_path is None:
        pdf_path = settings.stahl_pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"Stahl PDF not found: {pdf_path}")

    content_hash = _content_hash(pdf_path)
    version_label = "Stahl Prescriber's Guide (Cambridge 2024)"

    # Stage 1: Extract layout
    logger.info(f"Extracting layout from {pdf_path.name}...")
    if page_range is None:
        page_range = (33, 41)  # amisulpride default for POC
    pages = extract_pages_layout(pdf_path, page_range)
    logger.info(f"Extracted {len(pages)} pages")

    # Stage 2: Parse monographs
    logger.info("Parsing monographs...")
    monographs = parse_monographs(pages)
    logger.info(f"Found {len(monographs)} monographs")

    all_warnings: list[str] = []
    for mono in monographs:
        w = validate_monograph(mono)
        if w:
            all_warnings.append(f"{mono['drug_name']}: {w}")

    # Stage 3: Convert to claims
    logger.info("Converting to claims...")
    claims_data = monographs_to_claims(monographs, 0)

    if drug_filter:
        drug_slug = _slug(drug_filter)
        claims_data = [c for c in claims_data if c["drug_slug"] == drug_slug]
        monographs = [m for m in monographs if _slug(m["drug_name"]) == drug_slug]

    # Database operations
    session = get_session()
    try:
        doc = session.execute(
            sqltext(
                "SELECT id FROM source_documents WHERE source_type = :st AND content_hash = :ch"
            ),
            {"st": "stahl", "ch": content_hash},
        ).fetchone()

        if doc:
            source_doc_id = doc[0]
            logger.info(f"Reusing existing source_document id={source_doc_id}")
        else:
            sd = SourceDocument(
                source_type="stahl",
                scope_key=None,
                content_hash=content_hash,
                version_label=version_label,
                is_active=False,
            )
            session.add(sd)
            session.flush()
            source_doc_id = sd.id

        run = IngestionRun(
            source_document_id=source_doc_id,
            source="stahl",
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(run)
        session.flush()
        run_id = run.id

        records_seen = len(claims_data)
        records_written = 0

        for claim_data in claims_data:
            drug_slug = claim_data["drug_slug"]
            drug_row = session.execute(
                sqltext("SELECT id FROM drugs WHERE slug = :slug"),
                {"slug": drug_slug},
            ).fetchone()

            if not drug_row:
                drug = Drug(
                    canonical_name=claim_data["drug_name"].title(),
                    slug=drug_slug,
                )
                session.add(drug)
                session.flush()
                drug_id = drug.id
                all_warnings.append(f"Created new drug from Stahl: {claim_data['drug_name']}")
            else:
                drug_id = drug_row[0]

            existing_c = session.execute(
                sqltext(
                    "SELECT id FROM source_claims "
                    "WHERE source_document_id = :sdid AND drug_id = :did "
                    "AND COALESCE(drug_variant_id, 0) = 0 "
                    "AND category = :cat AND claim_hash = :ch"
                ),
                {
                    "sdid": source_doc_id,
                    "did": drug_id,
                    "cat": claim_data["category"],
                    "ch": claim_data["claim_hash"],
                },
            ).fetchone()

            if not existing_c:
                claim = SourceClaim(
                    source_document_id=source_doc_id,
                    drug_id=drug_id,
                    drug_variant_id=None,
                    category=claim_data["category"],
                    text=claim_data["text"],
                    attributes=claim_data["attributes"],
                    claim_hash=claim_data["claim_hash"],
                    locator=claim_data["locator"],
                    claim_type=claim_data["claim_type"],
                )
                session.add(claim)
                records_written += 1

        run.status = "completed" if not all_warnings else "partial"
        run.records_seen = records_seen
        run.records_written = records_written
        run.warnings = all_warnings
        run.completed_at = datetime.utcnow()

        # Atomically activate
        session.execute(
            sqltext(
                "UPDATE source_documents SET is_active = false "
                "WHERE source_type = 'stahl' AND scope_key IS NULL"
            )
        )
        session.execute(
            sqltext("UPDATE source_documents SET is_active = true WHERE id = :id"),
            {"id": source_doc_id},
        )

        session.commit()

        logger.info(
            f"Stahl ingestion complete: {len(monographs)} monographs, "
            f"{records_seen} claims seen, {records_written} written, "
            f"{len(all_warnings)} warnings"
        )
        return {
            "source_document_id": source_doc_id,
            "run_id": run_id,
            "monographs": len(monographs),
            "records_seen": records_seen,
            "records_written": records_written,
            "warnings": all_warnings,
        }

    except Exception as e:
        session.rollback()
        logger.error(f"Stahl ingestion failed: {e}")
        raise
    finally:
        session.close()
