# Failure Taxonomy — Complex Hybrid Pipeline (fixed, source_doc=5 only)

**Pipeline:** vector + FTS + RRF, section-aware chunks (1,691 chunks)
**Gold set:** kaplan_ch33_ocd_gold_v0.1 (7 cases, 23 evidence items)
**Result:** 0/7 pass, Macro critical Recall@10 = 47.6%

## Three failure types

### Type 1: SEMANTIC GAP — evidence chunk not in top 20 (12 of 14 misses)

The chunk containing the anchor exists and is on the correct page,
but is not retrieved at all. The question language doesn't match the
chunk's dominant topic under vector+FTS+RRF fusion.

| Evidence ID | Gold page | Chunk ID | Closest retrieved | Distance |
|---|---|---|---|---|
| P040-OCD-AUGMENTATION | 39 | 3722 | rank 2, page 37 | 2 pages |
| P075-OCD-DOPAMINE-AUGMENTERS | 74 | 3781 | rank 5, page 75 | 1 page |
| P195-RISPERIDONE-MOA | 194 | 3905 | rank 16, page 176 | 18 pages |
| P283-PARTIAL-AGONIST-SAFETY | 282 | 4003 | rank 2, page 281 | 1 page |
| P283-ARIPIPRAZOLE-NBN | 282 | 4003 | rank 0, page 281 | 1 page |
| P284-ARIPIPRAZOLE-SAFETY | 283 | 4004 | rank 6, page 285 | 2 pages |
| P077-NBN-DIMENSIONS | 76 | 3784 | rank 17, page 77 | 1 page |

**Root cause per evidence item:**

- **P040-OCD-AUGMENTATION** (missed in 1 case): Chunk 3722 is a small
  chunk (1,022 chars) about combination therapy. The question asks
  about "pharmacologic directions" for "partial response to an SRI".
  The chunk's text starts with omega-3 fatty acids and bipolar
  disorder, with the OCD evidence mid-chunk. FTS doesn't help because
  the question doesn't use the chunk's keywords.

- **P075-OCD-DOPAMINE-AUGMENTERS** (missed in 5 cases): Chunk 3781 is
  in section "33.1c Classification of Psychotropics: NbN" (pages 72-78).
  The chunk's dominant topic is dopamine blocker classification and
  NbN taxonomy, with the OCD augmentation evidence as one sentence.
  The question language about "OCD augmentation" or "pharmacologic
  mechanisms" doesn't match the section's classification language.
  This is the most impactful failure: 5 of 7 cases need this evidence.

- **P195-RISPERIDONE-MOA** (missed in 2 cases): Chunk 3905 is in
  section "33.2b Dopamine-Serotonin Antagonists" (pages 9925-10033).
  The chunk starts with "Mechanism of Action. Risperidone and
  paliperidone share similar receptor affinities." The question asks
  about "D2 partial agonism vs D2 antagonism" — the chunk is about
  D2 antagonism, not partial agonism. The vector similarity is low
  because the question is about the distinction between two mechanisms,
  while the chunk describes only one.

- **P283-PARTIAL-AGONIST-SAFETY** (missed in 1 case): Chunk 4003 is in
  section "33.3a Dopamine Partial Agonists" (pages 10066-10088). The
  chunk's dominant topic is aripiprazole's development history, with
  the safety rationale buried mid-chunk. The question asks about "D2
  partial agonism" — the chunk is about aripiprazole specifically, not
  the general mechanism distinction.

- **P283-ARIPIPRAZOLE-NBN** (missed in 1 case): Same chunk 4003. The
  NbN classification text is at the end of the chunk. The question
  asks about "drugs with dopamine antagonism or partial agonism" —
  the chunk's dominant topic is aripiprazole's clinical development.

- **P284-ARIPIPRAZOLE-SAFETY** (missed in 1 case): The chunk with the
  safety profile text is on page 283, but no retrieved chunk overlaps.
  The closest is rank 6, page 285 (2 pages away). The safety text is
  in a chunk dominated by dosing information.

- **P077-NBN-DIMENSIONS** (missed in 2 cases): Chunk 3784 is in
  section "33.1c Classification of Psychotropics: NbN". The chunk's
  dominant topic is mood stabilizers and cation channel blockers, with
  the NbN dimensions text at the end: "the pharmacology and MoA domains
  represent the main pharmacologic properties." The question asks about
  NbN classification — the chunk's dominant topic is mood stabilizers,
  not NbN dimensions.

### Type 2: RANKING BOUNDARY — in top 20 but outside top 10 (2 of 14 misses)

| Evidence ID | Chunk ID | Rank | Missed in |
|---|---|---|---|
| P076-OCD-SRI-SPECIFICITY | 3783 | 13 | NBN-GOLD-01 |
| P077-NBN-DIMENSIONS | 3784 | 14 | NBN-GOLD-01 |

Both misses are in NBN-GOLD-01. The chunks are on the correct pages
(75 and 76) and contain the anchors, but are ranked just outside the
top 10. A small ranking improvement would fix both.

### Type 3: CHUNK BOUNDARY SPLIT (0 of 14 misses)

The complex pipeline's section-aware chunking does NOT produce chunk
boundary splits. This is because section-aware chunks are larger and
tend to keep a single topic within one chunk, avoiding the split that
the naive 500-token chunking creates.

## Summary by case

| Case | Missed critical @10 | Failure types |
|---|---|---|
| OCD-GOLD-01 | P040, P075 | Type 1, Type 1 |
| OCD-GOLD-02 | P075 | Type 1 |
| OCD-GOLD-03 | P195, P283-SAFETY, P075 | Type 1, Type 1, Type 1 |
| OCD-GOLD-04 | P195, P283-NBN | Type 1, Type 1 |
| OCD-GOLD-05 | P075, P284 | Type 1, Type 1 |
| NBN-GOLD-01 | P076, P077 | Type 2, Type 2 |
| NBN-GOLD-02 | P075, P077 | Type 1, Type 1 |

## Root cause frequency

| Root cause | Occurrences | Evidence items affected |
|---|---|---|
| Semantic gap (not in top 20) | 12 | P040, P075 (×5), P195 (×2), P283-SAFETY, P283-NBN, P284, P077 (×2) |
| Ranking boundary (in top 20, not top 10) | 2 | P076, P077 |
| Chunk boundary split | 0 | — |

## Comparison with baseline

| Metric | Baseline | Complex (fixed) |
|---|---|---|
| Macro critical Recall@10 | 56.0% | 47.6% |
| Total missed @10 | 13 | 14 |
| Type 1 (not in top 20) | 8 | 12 |
| Type 2 (ranking boundary) | 5 | 2 |
| Type 3 (chunk boundary) | 3 | 0 |
| P075 missed in | 3 cases | 5 cases |

## Key observations

1. **The complex pipeline is WORSE than the naive baseline.** Macro
   critical Recall@10 drops from 56.0% to 47.6%.

2. **The complex pipeline has MORE Type 1 failures (12 vs 8).** The
   section-aware chunking makes the semantic gap worse, not better.
   Section boundaries split evidence across chunks that are topically
   distant from the question.

3. **The complex pipeline eliminates Type 3 (chunk boundary splits).**
   Section-aware chunks are larger and keep topics together, avoiding
   the naive chunking's page-split problem.

4. **P075 is missed in ALL 5 cases that need it** (vs 3 in baseline).
   The chunk is in the NbN classification section, and the question
   language about OCD augmentation doesn't match the section's
   classification language. The section-aware chunking makes this worse
   because the chunk is larger and more dominated by classification
   text.

5. **FTS + RRF fusion doesn't help.** The FTS results don't surface
   the evidence chunks because the question language doesn't share
   keywords with the evidence text. RRF fusion averages the two
   rankings, which doesn't improve the evidence chunk's position.

6. **The complex pipeline's advantage (section awareness) is also its
   disadvantage.** Sections keep topics together, but the evidence is
   often a minority topic within a section. The section's dominant
   topic drowns out the evidence in both vector and FTS retrieval.
