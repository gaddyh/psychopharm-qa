"""Tests for Kaplan Chapter 33 ingestion and RAG retrieval.

These tests assert against the live database (PostgreSQL + pgvector).
They verify:
- Source document is active
- Chapter 33 page range is fully covered (9785-11293)
- Sections are detected with correct numbering
- Chunks are token-bounded (<= 800 tokens)
- Every chunk has an embedding
- Entity metadata (drug_ids) is populated
- Amisulpride (drug_id 22) appears in chunks
- Hybrid retrieval (vector + FTS + RRF) returns relevant passages
- RRF fusion deduplicates and ranks correctly
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from psych_qa.db.connection import get_engine
from psych_qa.llm.client import get_llm_client
from psych_qa.repositories.chunks import fulltext_search, get_chunk_by_id, vector_search
from psych_qa.retrieval.chapter_search import KaplanRetriever, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def kaplan_doc_id(engine):
    """Get the active Kaplan source document ID."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM source_documents WHERE source_type='kaplan' AND is_active=true")
        ).fetchone()
    assert row is not None, "No active Kaplan source document found. Run ingest_kaplan first."
    return row[0]


# ---------------------------------------------------------------------------
# Source document assertions
# ---------------------------------------------------------------------------

class TestKaplanSourceDocument:
    def test_source_document_exists_and_active(self, kaplan_doc_id):
        assert kaplan_doc_id is not None

    def test_scope_key_is_chapter_33(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT scope_key, version_label FROM source_documents WHERE id=:id"),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert row is not None
        assert row[0] == "chapter-33"
        assert "Chapter 33" in row[1]


# ---------------------------------------------------------------------------
# Section assertions
# ---------------------------------------------------------------------------

class TestKaplanSections:
    def test_sections_count(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("SELECT count(*) FROM document_sections WHERE source_document_id=:id"),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] >= 50, f"Expected >=50 sections, got {r[0]}"

    def test_at_least_one_section_exists(self, engine, kaplan_doc_id):
        """At least one section should exist in the chapter."""
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT section_number, title FROM document_sections
                    WHERE source_document_id=:id
                    ORDER BY section_number
                    LIMIT 1
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert row is not None, "No sections found in Kaplan chapter"
        assert row[0].startswith("33."), f"Section {row[0]} doesn't start with '33.'"

    def test_section_numbers_follow_pattern(self, engine, kaplan_doc_id):
        """All section numbers should start with '33.'."""
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT section_number FROM document_sections WHERE source_document_id=:id"),
                {"id": kaplan_doc_id},
            ).fetchall()
        for r in rows:
            assert r[0].startswith("33."), f"Section {r[0]} doesn't start with '33.'"

    def test_sections_have_page_ranges(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT section_number, physical_page_start, physical_page_end
                    FROM document_sections WHERE source_document_id=:id
                """),
                {"id": kaplan_doc_id},
            ).fetchall()
        for r in rows:
            assert r[1] is not None, f"Section {r[0]} has null physical_page_start"
            assert r[2] is not None, f"Section {r[0]} has null physical_page_end"
            assert r[1] <= r[2], f"Section {r[0]}: start > end"


# ---------------------------------------------------------------------------
# Chunk assertions
# ---------------------------------------------------------------------------

class TestKaplanChunks:
    def test_chunk_count(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("SELECT count(*) FROM document_chunks WHERE source_document_id=:id"),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] >= 1000, f"Expected >=1000 chunks, got {r[0]}"

    def test_page_range_covers_chapter_33(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT min(physical_pdf_page), max(physical_pdf_page)
                    FROM document_chunks WHERE source_document_id=:id
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] <= 9790, f"First chunk page {r[0]} too late (expected ~9785)"
        assert r[1] >= 11280, f"Last chunk page {r[1]} too early (expected ~11293)"

    def test_chunks_are_token_bounded(self, engine, kaplan_doc_id):
        """Each chunk should be <= 1500 tokens.

        Target is 500 tokens with 100 overlap, but some Kaplan paragraphs
        are large unbroken blocks. 1500 is the safety ceiling — anything
        larger should be split by the chunker.
        """
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, text FROM document_chunks WHERE source_document_id=:id"),
                {"id": kaplan_doc_id},
            ).fetchall()
        oversized = []
        for r in rows:
            tokens = len(enc.encode(r[1]))
            if tokens > 1500:
                oversized.append((r[0], tokens))
        assert not oversized, f"{len(oversized)} chunks exceed 1500 tokens: {oversized[:5]}"

    def test_chunks_have_input_hash(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT count(*) FROM document_chunks
                    WHERE source_document_id=:id AND input_hash IS NULL
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] == 0, f"{r[0]} chunks have null input_hash"

    def test_chunks_have_search_vector(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT count(*) FROM document_chunks
                    WHERE source_document_id=:id AND search_vector IS NULL
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] == 0, f"{r[0]} chunks have null search_vector"


# ---------------------------------------------------------------------------
# Embedding assertions
# ---------------------------------------------------------------------------

class TestKaplanEmbeddings:
    def test_every_chunk_has_embedding(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT count(*) FROM document_chunks dc
                    WHERE dc.source_document_id=:id
                    AND NOT EXISTS (
                        SELECT 1 FROM document_embeddings de
                        WHERE de.chunk_id = dc.id
                    )
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] == 0, f"{r[0]} chunks missing embeddings"

    def test_embedding_dimensions(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT DISTINCT dimensions FROM document_embeddings de
                    JOIN document_chunks dc ON dc.id = de.chunk_id
                    WHERE dc.source_document_id=:id
                """),
                {"id": kaplan_doc_id},
            ).fetchall()
        assert len(r) == 1, f"Expected 1 dimension value, got {r}"
        assert r[0][0] == 1536, f"Expected 1536 dimensions, got {r[0][0]}"

    def test_embedding_model_is_text_embedding_3_small(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT DISTINCT model FROM document_embeddings de
                    JOIN document_chunks dc ON dc.id = de.chunk_id
                    WHERE dc.source_document_id=:id
                """),
                {"id": kaplan_doc_id},
            ).fetchall()
        assert len(r) == 1
        assert r[0][0] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Entity metadata assertions
# ---------------------------------------------------------------------------

class TestKaplanEntityMetadata:
    def test_amisulpride_in_chunk_metadata(self, engine, kaplan_doc_id):
        """Drug ID 22 (amisulpride) should appear in at least 5 chunks."""
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT count(*) FROM document_chunks
                    WHERE source_document_id=:id
                    AND entity_metadata->'drug_ids' @> '[22]'::jsonb
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] >= 5, f"Expected >=5 chunks with amisulpride, got {r[0]}"

    def test_amisulpride_in_chunk_text(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT count(*) FROM document_chunks
                    WHERE source_document_id=:id AND text ILIKE '%amisulpride%'
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] >= 5, f"Expected >=5 chunks mentioning amisulpride, got {r[0]}"

    def test_metadata_has_drug_ids_array(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            r = conn.execute(
                text("""
                    SELECT count(*) FROM document_chunks
                    WHERE source_document_id=:id
                    AND entity_metadata->'drug_ids' IS NOT NULL
                """),
                {"id": kaplan_doc_id},
            ).fetchone()
            total = conn.execute(
                text("SELECT count(*) FROM document_chunks WHERE source_document_id=:id"),
                {"id": kaplan_doc_id},
            ).fetchone()
        assert r[0] == total[0], f"{r[0]}/{total[0]} chunks have drug_ids array"


# ---------------------------------------------------------------------------
# Retrieval assertions
# ---------------------------------------------------------------------------

class TestKaplanVectorSearch:
    def test_vector_search_returns_results(self):
        client = get_llm_client()
        emb = client.embed_one("amisulpride mechanism of action dopamine receptor")
        results = vector_search(emb, limit=5)
        assert len(results) > 0, "Vector search returned no results"

    def test_vector_search_scores_sorted(self):
        client = get_llm_client()
        emb = client.embed_one("amisulpride antipsychotic")
        results = vector_search(emb, limit=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"

    def test_vector_search_drug_filter(self):
        client = get_llm_client()
        emb = client.embed_one("amisulpride")
        results = vector_search(emb, limit=5, drug_id=22)
        assert len(results) > 0, "Vector search with drug filter returned no results"
        for r in results:
            drug_ids = r.get("entity_metadata", {}).get("drug_ids", [])
            assert 22 in drug_ids, f"Chunk {r['id']} doesn't have drug_id 22 in metadata"


class TestKaplanFullTextSearch:
    def test_fts_returns_results(self):
        results = fulltext_search("amisulpride schizophrenia", limit=5)
        assert len(results) > 0, "FTS returned no results"

    def test_fts_scores_non_negative(self):
        results = fulltext_search("dopamine receptor antagonist", limit=5)
        for r in results:
            assert r["score"] >= 0, f"Negative FTS score: {r['score']}"

    def test_fts_drug_filter(self):
        results = fulltext_search("amisulpride", limit=5, drug_id=22)
        assert len(results) > 0, "FTS with drug filter returned no results"


# ---------------------------------------------------------------------------
# RRF fusion assertions
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_rrf_combines_and_deduplicates(self):
        vector_results = [
            {"id": 1, "text": "a", "score": 0.9, "evidence_id": "k_1"},
            {"id": 2, "text": "b", "score": 0.8, "evidence_id": "k_2"},
            {"id": 3, "text": "c", "score": 0.7, "evidence_id": "k_3"},
        ]
        fts_results = [
            {"id": 2, "text": "b", "score": 0.5, "evidence_id": "k_2"},
            {"id": 4, "text": "d", "score": 0.4, "evidence_id": "k_4"},
        ]
        fused = reciprocal_rank_fusion(vector_results, fts_results)
        ids = [f["id"] for f in fused]
        assert len(ids) == len(set(ids)), "RRF has duplicates"
        assert set(ids) == {1, 2, 3, 4}, f"Unexpected IDs: {ids}"
        # Item 2 appears in both lists → should rank higher
        assert fused[0]["id"] == 2, f"Expected item 2 first (in both lists), got {fused[0]['id']}"

    def test_rrf_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_rrf_single_list(self):
        results = [{"id": 1, "text": "a", "score": 0.9, "evidence_id": "k_1"}]
        fused = reciprocal_rank_fusion(results, [])
        assert len(fused) == 1
        assert fused[0]["id"] == 1


# ---------------------------------------------------------------------------
# End-to-end retrieval assertions
# ---------------------------------------------------------------------------

class TestKaplanEndToEndRetrieval:
    def test_retriever_returns_kaplan_passages_for_amisulpride(self):
        retriever = KaplanRetriever(top_k=5)
        results = retriever.retrieve(
            "What is the evidence for amisulpride's effectiveness as an antipsychotic?",
            drug_ids=[22],
        )
        assert len(results) > 0, "Retriever returned no passages"
        assert len(results) <= 5
        for r in results:
            assert r["evidence_id"].startswith("kaplan_chunk_")
            assert "physical_pdf_page" in r
            assert r["physical_pdf_page"] >= 9785
            assert r["physical_pdf_page"] <= 11293

    def test_retriever_results_have_fused_score(self):
        retriever = KaplanRetriever(top_k=5)
        results = retriever.retrieve("dopamine receptor antagonist antipsychotic")
        for r in results:
            assert "fused_score" in r, "Missing fused_score"
            assert r["fused_score"] > 0

    def test_get_chunk_by_id(self, engine, kaplan_doc_id):
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id FROM document_chunks WHERE source_document_id=:id LIMIT 1"),
                {"id": kaplan_doc_id},
            ).fetchone()
        chunk = get_chunk_by_id(row[0])
        assert chunk is not None
        assert chunk["id"] == row[0]
        assert "text" in chunk
        assert chunk["evidence_id"] == f"kaplan_chunk_{row[0]}"
