# psychopharm-qa

Evidence-grounded psychopharmacology QA for clinicians.

## What it does

A Streamlit app that answers clinician questions about psychiatric drugs using three knowledge sources, with every medical claim cited to its source and validated by entailment:

- **NbN** (Neuroscience-based Nomenclature) — deterministic drug database: pharmacological domain, target, mode of action, dose variants, neurobiology, clinical effects.
- **Stahl** (Prescriber's Guide) — deterministic drug monographs: indications, mechanisms, dosing, safety, interactions, special populations, clinical pearls.
- **Kaplan** Chapter 33 (Biologic Therapies) — hierarchical narrative RAG for deeper clinical context, trial evidence, and treatment rationale.

Stahl and NbN are merged under a canonical drug entity, but each fact preserves which source asserted it. Kaplan remains narrative evidence retrieved by hybrid search. Disagreements between sources are surfaced explicitly.

### Trust model

We trust an answer only when:

1. **Source fidelity** — Stahl, NbN, and Kaplan are extracted accurately.
2. **Retrieval** — required evidence is retrieved for each question.
3. **Grounding** — every answer claim is supported by cited evidence (validated by entailment).
4. **Clinical correctness** — answers and uncertainty are reviewed by a psychiatrist.

### Conversation support

The system supports multi-turn conversations. Context resolves what the doctor means; fresh evidence determines what the system says. Prior generated answers are used only for resolving the new question, never as medical evidence for answer generation.

## Architecture

```
Streamlit → parse question (LLM)
         → resolve conversational references (if conversation exists)
         → deterministic Stahl/NbN drug lookup
         → Kaplan Chapter 33 hybrid retrieval (vector + FTS + RRF + rerank)
         → question-type evidence filtering
         → sufficiency check (abstain or clarify if insufficient)
         → LLM generates claims with citation IDs (structured JSON)
         → citation validator (rejects hallucinated evidence IDs)
         → claim-to-evidence entailment validation
         → deterministic claim criticality assignment
         → build answer prose from validated claims
         → save answer trace + metrics
         → compute evaluation metrics
         → clinician feedback (thumbs up/down/correction)
```

### Source separation

Every fact retains its source identity:
- NbN claims → `nbn_claim_<id>`
- Stahl claims → `stahl_claim_<id>`
- Kaplan passages → `kaplan_chunk_<id>`

The LLM cites these internal IDs. The renderer resolves them into source names and page numbers. The citation validator strips any ID that doesn't exist in the evidence package. The entailment validator checks that each claim is actually supported by its cited evidence.

### Bold/FDA-approved detection (Stahl)

Stahl uses bold formatting to mark FDA-approved indications. The parser recovers span-level font information via PyMuPDF and infers `fda_approved=True` when `bold_ratio >= 0.85`. This is represented as an inference from the book's formatting convention, not as absolute regulatory truth.

### Conversation model

Answer traces store `conversation_id`, `turn_number`, `resolved_question`, `question_understanding`, `evidence_package`, `answer`, and metrics. Conversation context is reconstructed from prior traces — there is no separate conversation-state table. The resolver receives the last three turns; the UI displays the complete thread.

The context resolver returns statuses: `standalone`, `resolved`, `ambiguous`, `missing_context`. Ambiguous references trigger clarification rather than guessing.

## Current state

### Ingested data

| Source | Records | Notes |
|--------|---------|-------|
| NbN | 154 drugs, 2098 claims | All drugs, 2 dose variants for amisulpride (low/upper) |
| Stahl | 1 monograph, 179 claims | Amisulpride only (PDF p.33-40), 38 categories |
| Kaplan Ch.33 | 68 sections, 1507 chunks, 1507 embeddings | Full chapter (PDF p.9785-11293), `text-embedding-3-small` |

Current clinical focus: **Amisulpride only**. Aripiprazole appears in abstention/behavioral tests to verify correct out-of-corpus handling; it has not been ingested as a supported drug.

### Evaluation datasets

| Dataset | Cases | Purpose | Expected performance |
|---------|-------|---------|---------------------|
| `golden_v1.jsonl` | 49 | Baseline regression — retrieval validation | 100% Recall@K (frozen) |
| `abstention_v1.jsonl` | 21 | Abstention/clarification behavior | 100% correct abstention |
| `golden_hard.jsonl` | 23 | Challenge set — expose reasoning/trust boundary failures | Below 100% (healthy) |

**Golden v1** (49 cases) includes:
- 28 standalone cases (mechanism, indications, dosing, side effects, interactions, renal impairment, pregnancy, pharmacokinetics, QTc, prolactin, monitoring, treatment failure, overdose, dementia, onset, discontinuation, weight/metabolic)
- 7 cross-source cases (Stahl + NbN + Kaplan)
- 14 clinician-designed simulated conversation cases (prolactin, low-dose mechanism, negative symptoms, renal impairment, QT-risk, akathisia, certainty interrogation)

**Hard challenge set** (23 turns across 12 conversations) tests:
- Apparent contradiction (antagonist that increases transmission)
- Mechanism-to-outcome overreach (mechanism ≠ efficacy proof)
- Source disagreement (Stahl "theoretically" vs NbN direct statement)
- Differential diagnosis boundary (drug monographs ≠ diagnostic references)
- Multiple simultaneous risks (renal + QTc + elderly)
- Patient-specific dosing boundary (must not prescribe)
- Association vs causality (timing alone ≠ proof)
- Formulation/indication confusion (PONV vs psychiatric use)
- Unsupported comparator (aripiprazole not in corpus)
- Exact number trap (resist pressure to fabricate percentages)
- Active-entity tracking across conversation turns
- Self-audit and certainty correction when challenged

Each hard case specifies `required_behaviors`, `forbidden_behaviors`, and `expected_outcome` (supported_answer, partial_answer, clarification_required, conflicting_evidence, unsupported_specificity, out_of_corpus, clinical_boundary).

### LLM judge

A dedicated judge model (`JUDGE_MODEL`, defaults to `gpt-5.4`) evaluates answers against required and forbidden behaviors. The judge is a screening layer before human (Sasson) review — it flags likely pass/fail so human review can focus on uncertain cases.

Calibration on golden_v1 (5 cases with known-good answers):
- Forbidden behavior detection: 100% (correctly identifies no forbidden claims)
- Required behavior detection: 67.7% (strict — catches content coverage gaps)
- Judge pass rate: 20% (healthy — not rubber-stamping)
- Found a real system bug: over-aggressive abstention on drug interactions

The judge prompt and structured output schema are designed to be updated after Sasson review.

### Golden run metrics

```
Golden v1:              49 cases
Average Recall@K:       100%
Full-recall cases:      49/49
Required points:        166/166
Cross-source cases:     7/7
Conversation cases:     21/21
Clinical simulation:    14/14
Review status:          Sasson pending
```

### Tests

```
168 passed
```

Test coverage includes: Kaplan RAG (29 tests), conversation resolution (11 tests), answer pipeline, citations, entailment, sufficiency, criticality, evidence filtering, and regression promotion.

## Quick start

```bash
# 1. Database
docker compose up -d db

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env  # fill in OPENAI_API_KEY, DATABASE_URL, JUDGE_MODEL

# 4. Ingest
python scripts/ingest_nbn.py
python scripts/ingest_stahl.py --drug amisulpride
python scripts/ingest_kaplan.py
# or: python scripts/ingest_all.py

# 5. Run
streamlit run app.py
```

The app runs at `http://localhost:8501`.

## Evaluation

### Baseline retrieval validation

```bash
# Validate golden_v1 retrieval (no API calls needed for retrieval check)
python scripts/run_evals.py
```

### Hard challenge set

```bash
# Retrieval + outcome matching only (no API calls)
python -m psych_qa.evaluation.hard_runner

# Full evaluation with LLM judge (requires JUDGE_MODEL)
python -m psych_qa.evaluation.hard_runner --judge --real-answers
```

### Judge calibration

```bash
# Calibrate judge against known-good golden_v1 answers
python -m psych_qa.evaluation.judge_calibration
```

### Reports

- `evals/results/latest_golden_run.json` — baseline golden run report
- `evals/results/latest_hard_run.json` — hard challenge set report
- `evals/results/judge_calibration.json` — judge calibration results

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
  run_evals.py                  # Offline evaluation
  reset_db.py                   # Reset database
src/psych_qa/
  config.py                     # pydantic-settings configuration
  domain/
    enums.py                    # Source types, claim categories, feedback types
    models.py                   # Pydantic domain models (EvidenceItem, EvidencePackage, etc.)
  db/
    connection.py               # SQLAlchemy engine + session factory
    tables.py                   # All table definitions + init_db()
  llm/
    client.py                   # OpenAI chat + embeddings client (supports per-call model override)
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
    conversations.py            # Answer trace persistence + conversation context
    feedback.py                 # Doctor feedback persistence
  retrieval/
    question_parser.py          # LLM question parsing + context resolution
    drug_lookup.py              # Resolve drug names → canonical drugs + claims
    chapter_search.py           # Kaplan hybrid retrieval (vector + FTS + RRF)
    reranker.py                 # LLM-based passage reranking
  answering/
    evidence_builder.py         # Assemble unified evidence package from all sources
    evidence_filter.py          # Question-type-specific evidence category filtering
    answer_service.py           # Full pipeline orchestrator (supports conversations)
    sufficiency.py              # Abstain or clarify when evidence insufficient
    citations.py                # Validate + strip hallucinated evidence IDs
    entailment.py               # Claim-to-evidence entailment validation
    criticality.py              # Deterministic claim criticality assignment
    prompts.py                  # System + evidence prompt templates
    token_budget.py             # Kaplan token budget management
    metrics.py                  # Per-run + aggregate metrics
  evaluation/
    schemas.py                  # GoldCase, AbstentionCase, RegressionCase, HardGoldCase
    loader.py                   # Dataset loading from JSONL
    matchers.py                 # Evidence matching for Recall@K computation
    evaluators.py               # Retrieval evaluation logic
    regression.py               # Regression promotion from feedback
    hard_runner.py              # Hard challenge set evaluation runner
    judge.py                    # LLM judge for behavior scoring
    judge_calibration.py        # Judge calibration on golden_v1
  feedback/
    service.py                  # Save clinician feedback
    review_queue.py             # List negative/correction feedback
evals/
  datasets/
    golden_v1.jsonl             # 49 baseline gold cases (frozen)
    abstention_v1.jsonl         # 21 abstention/clarification cases
    golden_hard.jsonl           # 23 hard challenge turns (12 conversations)
  results/
    latest_golden_run.json      # Latest golden run report
    latest_hard_run.json        # Latest hard set report
    judge_calibration.json      # Judge calibration results
tests/
  ingestion/
    test_kaplan.py              # Kaplan RAG tests
  answering/
    test_conversation.py        # Conversation resolution + context tests
    test_answering.py           # Answer pipeline tests
    test_citations.py           # Citation validation tests
    test_entailment.py          # Entailment validation tests
    test_sufficiency.py         # Sufficiency/abstention tests
    test_evidence_filter.py     # Evidence filtering tests
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

Answer traces store: `conversation_id`, `turn_number`, `question`, `resolved_question`, `question_understanding`, `evidence_package`, `answer`, `metrics`, `versions`. Uniqueness guaranteed for `(conversation_id, turn_number)`.

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Conversation tests
python -m pytest tests/answering/test_conversation.py -v

# Kaplan RAG tests
python -m pytest tests/ingestion/test_kaplan.py -v
```

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg://psych:psych@localhost:5433/psychopharm` | Database connection |
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Model for question parsing + answer generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `JUDGE_MODEL` | `gpt-5.4` | Model for hard-set behavior evaluation |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit port |

## Metrics

### Per-run metrics (stored in `answer_traces.metrics`)

**Retrieval:** drugs_requested, drugs_found, drug_lookup_success_rate, nbn/stahl/kaplan_evidence_provided, kaplan_top_score, kaplan_mean_score, conflicts_detected

**Citations:** total_claims, claims_with_valid_citations, citation_completeness, unique_evidence_ids_cited, evidence_utilization, invalid_citation_count, citation_validity

**Entailment:** citation_entailment, supported_claims, unsupported_specificity_count, contradicted_count

**Source coverage:** nbn_cited, stahl_cited, kaplan_cited, sources_used (0-3)

**Answer quality:** status, has_uncertainties, has_clarification, direct_answer_chars, explanation_chars, critical_claims, unsupported_critical_claims

**Performance:** llm_latency_ms, total_latency_ms, retrieval_latency_ms, input_token_budget, evidence_tokens_used

### Aggregate metrics (Evaluation page)

Mean/min/max across all traces, plus status distribution, source usage distribution, abstention rate, citation validity, and entailment scores.

## Review status

All clinical content is **pending Sasson (psychiatrist) review**. Perfect retrieval recall does not mean the questions or answers are clinically approved. The clinical simulation conversations are labeled as "clinician-designed simulated conversations" — not real-world transcripts — until authentic de-identified conversations are collected from practicing psychiatrists.

## Next steps

### Clinical validation
- [ ] Collect 15 authentic de-identified conversations from Sasson and 2-3 psychiatrists
- [ ] Have Sasson review and correct 5 hard challenge cases
- [ ] Convert important failures into regression tests
- [ ] Update judge prompt and schema after Sasson review

### System improvements
- [ ] Fix over-aggressive abstention on drug interactions (found by judge calibration)
- [ ] Tune judge fabrication check for truncated evidence context
- [ ] Implement dynamic source filtering ("answer using Kaplan only")
- [ ] Add LLM judge for hard-set behavior scoring at scale

### Ingestion expansion
- [ ] Ingest full Stahl book (973 pages, ~100+ monographs) — currently only amisulpride
- [ ] Add a second supported drug (after Amisulpride is clinically validated)
- [ ] Kaplan table extraction (structured data from tables, not just text chunks)

### Infrastructure
- [ ] CI pipeline (lint + test on push)
- [ ] Alembic migrations (currently using `init_db(drop_first=False)`)
- [ ] Rate limiting / cost tracking for OpenAI calls
- [ ] Deploy (Docker compose with app + db, or cloud)
