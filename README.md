# psychopharm-qa

Evidence-grounded psychopharmacology QA for clinicians.

Three knowledge layers feed one orchestrated answer pipeline:

- **NbN** (Neuroscience-based Nomenclature) — deterministic drug database: pharmacological domain, target, mode of action, dose variants, neurobiology, clinical effects.
- **Stahl** (Prescriber's Guide) — deterministic drug monographs: indications, mechanisms, dosing, safety, interactions, special populations, clinical pearls.
- **Kaplan** Chapter 33 (Biologic Therapies) — hierarchical narrative RAG for deeper clinical context and treatment rationale.

Stahl and NbN are merged under a canonical drug entity, but each fact preserves which source asserted it. Kaplan remains narrative evidence retrieved by hybrid search.

## Architecture

```
Streamlit → parse question
         → deterministic Stahl/NbN lookup
         → Kaplan Chapter 33 hybrid retrieval
         → build unified evidence package
         → LLM creates cited answer
         → save trace + doctor feedback
```

## Quick start

```bash
# 1. Database
docker compose up -d db

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env  # fill in OPENAI_API_KEY

# 4. Ingest (amisulpride vertical slice first)
python -m psych_qa.ingestion.pipeline --source nbn
python -m psych_qa.ingestion.pipeline --source stahl --drug amisulpride
python -m psych_qa.ingestion.pipeline --source kaplan --section 33.13

# 5. Run
streamlit run app.py
```

## Project structure

See `/Users/gaddy/.devin/plans/plan-abed6e002986dff1.md` for the full design document.
