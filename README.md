# psychopharm-qa

Evidence-grounded psychopharmacology QA for clinicians.

## What it does

A Streamlit app that answers clinician questions about psychiatric drugs using three knowledge sources, with every claim cited to its source:

- **NbN** (Neuroscience-based Nomenclature) — deterministic drug database: pharmacological domain, target, mode of action, dose variants, neurobiology, clinical effects.
- **Stahl** (Prescriber's Guide) — deterministic drug monographs: indications, mechanisms, dosing, safety, interactions, special populations, clinical pearls.
- **Kaplan** Chapter 33 (Biologic Therapies) — hierarchical narrative RAG for deeper clinical context, trial evidence, and treatment rationale.

Stahl and NbN are merged under a canonical drug entity, but each fact preserves which source asserted it. Kaplan remains narrative evidence retrieved by hybrid search. Disagreements between sources are surfaced explicitly.

## Architecture

```
Streamlit → parse question (LLM)
         → deterministic Stahl/NbN drug lookup
         → Kaplan Chapter 33 hybrid retrieval (vector + FTS + RRF + rerank)
         → build unified evidence package
         → LLM generates cited answer (structured JSON)
         → citation validator (rejects hallucinated evidence IDs)
         → compute run metrics
         → save answer trace + metrics
         → clinician feedback (thumbs up/down/correction)
```

### Source separation

Every fact retains its source identity:
- NbN claims → `nbn_claim_<id>`
- Stahl claims → `stahl_claim_<id>`
- Kaplan passages → `kaplan_chunk_<id>`

The LLM cites these internal IDs. The renderer resolves them into source names and page numbers. The citation validator strips any ID that doesn't exist in the evidence package.

### Bold/FDA-approved detection (Stahl)

Stahl uses bold formatting to mark FDA-approved indications. The parser recovers span-level font information via PyMuPDF and infers `fda_approved=True` when `bold_ratio >= 0.85`. This is represented as an inference from the book's formatting convention, not as absolute regulatory truth.

## Current state

### Ingested data

| Source | Records | Notes |
|--------|---------|-------|
| NbN | 154 drugs, 2098 claims | All drugs, 2 dose variants for amisulpride (low/upper) |
| Stahl | 1 monograph, 179 claims | Amisulpride only (PDF p.33-40), 38 categories |
| Kaplan Ch.33 | 68 sections, 1507 chunks, 1507 embeddings | Full chapter (PDF p.9785-11293), `text-embedding-3-small` |

### Verified

- NbN Amisulpride: 2 variants (low/upper dose), 8 brand aliases, claims across all categories
- Stahl Amisulpride: bold/FDA detection verified (PONV=fda_approved, Schizophrenia=not, Dysthymia=not)
- Kaplan: 12 chunks mention amisulpride, hybrid retrieval returns relevant passages
- End-to-end: 10 default questions all answered, 100% citation validity, 0% abstention
- Tests: 29 Kaplan RAG tests passing

### Run metrics (10-question evaluation)

| Metric | Value |
|--------|-------|
| Citation validity | 100% |
| Citation coverage | 100% |
| Abstention rate | 0% |
| Avg sources used | 1.9/3 |
| Avg total latency | 5.7s |
| Avg LLM latency | 3.4s |
| Avg retrieval latency | 2.3s |
| Source usage | 3 questions used 1 source, 5 used 2, 2 used all 3 |

## Quick start

```bash
# 1. Database
docker compose up -d db

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env  # fill in OPENAI_API_KEY, DATABASE_URL

# 4. Ingest
python scripts/ingest_nbn.py
python scripts/ingest_stahl.py --drug amisulpride
python scripts/ingest_kaplan.py
# or: python scripts/ingest_all.py

# 5. Run
streamlit run app.py
```

The app runs at `http://localhost:8501`.

## Project structure

```
app.py                          # Streamlit home (system status)
pages/
  1_Chat.py                     # Ask questions, view cited answers + metrics, give feedback
  2_Feedback_Review.py          # Review queue for negative/corrected answers
  3_Evaluation.py               # Aggregate metrics across all traces
scripts/
  ingest_nbn.py                 # NbN ingestion
  ingest_stahl.py               # Stahl ingestion (--drug filter)
  ingest_kaplan.py              # Kaplan ingestion + embeddings
  ingest_all.py                 # Full pipeline
  run_evals.py                  # Offline evaluation (placeholder)
src/psych_qa/
  config.py                     # pydantic-settings configuration
  domain/
    enums.py                    # Source types, claim categories, feedback types
    models.py                   # Pydantic domain models (EvidenceItem, EvidencePackage, etc.)
  db/
    connection.py               # SQLAlchemy engine + session factory
    tables.py                   # All table definitions + init_db()
  llm/
    client.py                   # OpenAI chat + embeddings client
    schemas.py                  # JSON schemas for structured output
  ingestion/
    nbn.py                      # NbN JSON loader → drugs, variants, claims, aliases
    stahl.py                    # Stahl PDF parser → monographs → atomic claims
    kaplan.py                   # Kaplan PDF → sections → token-bounded chunks
    embeddings.py               # Batch embedding generation
    pipeline.py                 # Orchestrator (--source nbn/stahl/kaplan/all)
  repositories/
    drugs.py                    # Drug lookup by slug/name/alias
    claims.py                   # Source claim queries (active sources only)
    chunks.py                   # Kaplan vector + FTS search
    conversations.py            # Answer trace persistence
    feedback.py                 # Doctor feedback persistence
  retrieval/
    question_parser.py          # LLM question parsing + classification filter
    drug_lookup.py              # Resolve drug names → canonical drugs + claims
    chapter_search.py           # Kaplan hybrid retrieval (vector + FTS + RRF)
    reranker.py                 # LLM-based passage reranking
  answering/
    evidence_builder.py         # Assemble unified evidence package from all sources
    answer_service.py           # Full pipeline orchestrator
    sufficiency.py              # Abstain when evidence insufficient
    citations.py                # Validate + strip hallucinated evidence IDs
    prompts.py                  # System + evidence prompt templates
    metrics.py                  # Per-run + aggregate metrics
  feedback/
    service.py                  # Save clinician feedback
    review_queue.py             # List negative/correction feedback
tests/
  ingestion/
    test_kaplan.py              # 29 tests: sections, chunks, embeddings, metadata, retrieval, RRF
```

## Database

PostgreSQL 16 + pgvector, running in Docker:

```
host port: 5433
container port: 5432
database: psychopharm
user: psych
password: psych
```

Key tables: `drugs`, `drug_variants`, `drug_aliases`, `source_documents`, `source_claims`, `document_sections`, `document_chunks`, `document_embeddings`, `ingestion_runs`, `conversations`, `answer_traces`, `doctor_feedback`.

## Testing

```bash
# Kaplan RAG tests (29 tests)
python -m pytest tests/ingestion/test_kaplan.py -v

# All tests
python -m pytest tests/ -v
```

## Default questions

The Chat page includes 10 example questions:

1. What is the mechanism of action of amisulpride?
2. What are the FDA-approved indications for amisulpride?
3. What is the usual dosage range for amisulpride in schizophrenia?
4. What are the notable side effects of amisulpride?
5. What is the evidence for amisulpride's effectiveness as an antipsychotic, and how does its benzamide structure relate to its mechanism?
6. How does amisulpride compare to other atypical antipsychotics in clinical trials?
7. What did the EUFEST trial find about amisulpride compared to other antipsychotics in first-episode schizophrenia?
8. What is the evidence from clinical trials comparing amisulpride to haloperidol?
9. How does the NbN reclassification affect how amisulpride is categorized?
10. What are the treatment guidelines for amisulpride in first-episode schizophrenia?

Questions 7-9 are Kaplan-forward — their answers meaningfully draw from Kaplan's narrative content (EUFEST trial, NbN reclassification context).

## Metrics

### Per-run metrics (stored in `answer_traces.metrics`)

**Retrieval:** drugs_requested, drugs_found, drug_lookup_success_rate, nbn/stahl/kaplan_evidence_provided, kaplan_top_score, kaplan_mean_score, conflicts_detected

**Citations:** total_claims, claims_with_citations, citation_coverage, unique_evidence_ids_cited, evidence_utilization, invalid_citation_count, citation_validity

**Source coverage:** nbn_cited, stahl_cited, kaplan_cited, sources_used (0-3)

**Answer quality:** status, has_uncertainty, has_clarification, direct_answer_chars, explanation_chars

**Performance:** llm_latency_ms, total_latency_ms, retrieval_latency_ms

### Aggregate metrics (Evaluation page)

Mean/min/max across all traces, plus status distribution, source usage distribution, abstention rate, and avg citation validity.

## Next steps

### Ingestion expansion
- [ ] Ingest full Stahl book (973 pages, ~100+ monographs) — currently only amisulpride
- [ ] Stahl TOC validation: compare detected monographs against book table of contents
- [ ] Kaplan table extraction (structured data from tables, not just text chunks)
- [ ] Kaplan figure flagging (caption-only references, not figure knowledge)
- [ ] Improve chunker for large unbroken paragraphs (4 chunks currently exceed 1200 tokens)

### Retrieval improvements
- [ ] Parent-section context expansion (retrieve child chunk + compact parent summary)
- [ ] Adjacent chunk expansion (neighboring passages for continuity)
- [ ] Metadata-filtered retrieval (filter by condition, target, section_path)
- [ ] Tune RRF k parameter and reranker prompt
- [ ] Add cross-encoder reranker option (in addition to LLM-based reranking)

### Answer quality
- [ ] Question parser: improve drug name extraction for multi-drug comparison questions
- [ ] Evidence builder: smarter claim selection (don't send all 179 Stahl claims — filter by question type)
- [ ] Sufficiency checker: question-type-specific thresholds (dosing needs different evidence than mechanism)
- [ ] Conflict detection: compare NbN vs Stahl indications, dosing, side effects systematically
- [ ] Answer prompt: add few-shot examples for citation format
- [ ] Multi-drug comparison answers (currently optimized for single-drug questions)

### Evaluation
- [ ] Golden answer dataset (`golden_v1.jsonl`) — clinician-verified reference answers
- [ ] Abstention test cases — questions that should trigger abstention
- [ ] Regression dataset — promoted from corrected feedback
- [ ] Retrieval metrics: Recall@k, MRR, NDCG against labeled relevant passages
- [ ] Answer metrics: faithfulness, correctness vs golden answers
- [ ] Automated eval runner (`scripts/run_evals.py`)
- [ ] LLM-as-judge evaluator for answer quality scoring

### Feedback loop
- [ ] Promote corrected answers to evaluation cases
- [ ] Track feedback trends over time
- [ ] Alert when citation validity drops below threshold
- [ ] A/B compare prompt versions using feedback as signal

### Infrastructure
- [ ] CI pipeline (lint + test on push)
- [ ] Alembic migrations (currently using `init_db(drop_first=False)`)
- [ ] Source version diffing (what changed between NbN fetches)
- [ ] Embedding cache invalidation on chunk text change
- [ ] Rate limiting / cost tracking for OpenAI calls
- [ ] Deploy (Docker compose with app + db, or cloud)
