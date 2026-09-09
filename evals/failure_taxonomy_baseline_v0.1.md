# Failure Taxonomy — Naive Baseline (baseline-v0)

**Pipeline:** vector-only, 500-token chunks, 400-token step, 100-token overlap
**Gold set:** kaplan_ch33_ocd_gold_v0.1 (7 cases, 23 evidence items)
**Result:** 0/7 pass, Macro critical Recall@10 = 56.0%

## Three failure types

### Type 1: SEMANTIC GAP — evidence chunk not in top 20

The question language doesn't match the evidence language. Vector
similarity is too low for the chunk to appear at all. The chunk exists,
has the right text, is on the right page, but is never retrieved.

| Evidence ID | Gold page | Chunk ID | Chunk pages | Missed in |
|---|---|---|---|---|
| K33-P040-OCD-AUGMENTATION | 39 | 7308 | 39–40 | OCD-GOLD-01 |
| K33-P075-OCD-DOPAMINE-AUGMENTERS | 74 | 7367 | 73–74 | OCD-GOLD-01, 02, 05 |
| K33-P195-RISPERIDONE-MOA | 194 | 7519 | 193–194 | OCD-GOLD-03, 04 |
| K33-P284-ARIPIPRAZOLE-SAFETY-PROFILE | 283 | 7630 | 282–283 | OCD-GOLD-05 |

**Root cause per evidence item:**

- **P040-OCD-AUGMENTATION**: Question asks about "pharmacologic directions"
  for "partial response to an SRI". Chunk 7308 is about general
  augmentation strategies (STAR*D, lithium, thyroid hormone) with the
  OCD mention buried mid-chunk: "SSRIs typically produce partial
  improvement in patients with OCD, so the addition of a
  serotonin-dopamine antagonist may be helpful." The chunk's primary
  topic is general augmentation, not OCD-specific. Vector similarity is
  dominated by the general augmentation context.

- **P075-OCD-DOPAMINE-AUGMENTERS**: Question asks about "pharmacologic
  mechanisms" or "OCD augmentation". Chunk 7367 is primarily about NbN
  classification ("attributed to two domains. Pharmacology and Mode of
  Action...") with the OCD evidence buried mid-chunk: "low doses of
  risperidone or aripiprazole are proven helpful augmentation strategies
  for serotonergic..." The chunk's primary topic is NbN classification,
  not OCD treatment. Vector similarity is dominated by the NbN context.

- **P195-RISPERIDONE-MOA**: Question asks about "D2 partial agonism vs
  D2 antagonism" or "drugs with dopamine antagonism or partial agonism".
  Chunk 7519 is about risperidone's pharmacokinetics (CYP2D6, half-life)
  with the receptor affinity text at the end: "Risperidone and
  paliperidone share similar receptor affinities. Both have a high
  affinity for dopamine D2 receptors." The chunk's primary topic is
  pharmacokinetics, not receptor pharmacology. The question's mechanism
  language doesn't match the chunk's pharmacokinetics language.

- **P284-ARIPIPRAZOLE-SAFETY-PROFILE**: Question asks about "minimizing
  prolactin elevation, weight gain, and akathisia". Chunk 7630 is about
  aripiprazole's synthesis history ("Aripiprazole was synthesized in
  1988 and entered phase I...") with the safety profile text at the
  end: "benign side-effect profile, which includes fewer EPS and weight
  gain liability." The chunk's primary topic is drug development
  history, not safety comparison.

**Pattern:** In all four cases, the evidence passage is a minority of
the chunk text. The chunk's dominant topic is different from the
question's topic. Vector embeddings are dominated by the majority
context, drowning out the evidence passage.

### Type 2: RANKING BOUNDARY — evidence chunk in top 20 but outside top 10

The correct chunk IS retrieved, but ranked just outside the top 10
cutoff. A small ranking improvement would fix it.

| Evidence ID | Chunk ID | Rank | Missed in |
|---|---|---|---|
| K33-P076-OCD-SRI-SPECIFICITY | 7370 | 13 | OCD-GOLD-02, NBN-GOLD-01 |
| K33-P283-PARTIAL-AGONIST-SAFETY-RATIONALE | 7629 | 11 | OCD-GOLD-03 |
| K33-P283-ARIPIPRAZOLE-NBN | 7629 | 10 | NBN-GOLD-02 |

**Root cause:** Same semantic gap as Type 1, but less severe. The chunk
is semantically close enough to appear in top 20, but not close enough
for top 10. The question language partially matches but doesn't rank
the evidence chunk above other partially-relevant chunks.

### Type 3: CHUNK BOUNDARY SPLIT — right page in top 10, wrong chunk

A chunk overlapping the gold page IS in the top 10, but the anchor text
is in a different chunk on the same page. The chunk with the anchor is
ranked just outside the top 10.

| Evidence ID | Top-10 chunk | Anchor chunk | Missed in |
|---|---|---|---|
| K33-P076-OCD-SRI-SPECIFICITY | 7371 (rank 9, pp 75–76) | 7370 (rank 13, pp 75–76) | NBN-GOLD-01 |
| K33-P283-PARTIAL-AGONIST-SAFETY-RATIONALE | 7628 (rank 1, pp 281–282) | 7629 (rank 11, p 282) | OCD-GOLD-03 |
| K33-P283-ARIPIPRAZOLE-NBN | 7628 (rank 0, pp 281–282) | 7629 (not in top 10) | OCD-GOLD-04 |

**Root cause:** The 500-token chunking splits a single PDF page across
two chunks. The chunk without the evidence anchor happens to be more
semantically similar to the question and gets ranked higher. The chunk
with the anchor is ranked just below the cutoff.

This is a compound failure: Type 2 (ranking boundary) + chunk boundary
artifact. The page-aware matcher correctly identifies the page overlap
in the top-10 chunk, but the anchor requirement correctly rejects it
because the anchor text is in the adjacent chunk.

## Summary by case

| Case | Missed critical @10 | Failure types |
|---|---|---|
| OCD-GOLD-01 | P040, P075 | Type 1, Type 1 |
| OCD-GOLD-02 | P075, P076 | Type 1, Type 2 |
| OCD-GOLD-03 | P195, P283-SAFETY | Type 1, Type 2+3 |
| OCD-GOLD-04 | P195, P283-NBN | Type 1, Type 3 |
| OCD-GOLD-05 | P075, P284 | Type 1, Type 1 |
| NBN-GOLD-01 | P076 | Type 2+3 |
| NBN-GOLD-02 | P283-NBN | Type 2 |

## Root cause frequency

| Root cause | Occurrences | Evidence items affected |
|---|---|---|
| Semantic gap (not in top 20) | 8 | P040, P075 (×3), P195 (×2), P284 |
| Ranking boundary (in top 20, not top 10) | 5 | P076 (×2), P283-SAFETY, P283-NBN (×2) |
| Chunk boundary split | 3 | P076, P283-SAFETY, P283-NBN |

## Key observations

1. **P075-OCD-DOPAMINE-AUGMENTERS is the most impactful single failure.**
   It is missed in 3 of 7 cases (OCD-GOLD-01, 02, 05). The evidence
   text is buried in a chunk dominated by NbN classification language.
   Fixing this one evidence item would improve 3 cases.

2. **P195-RISPERIDONE-MOA is missed in 2 cases** (OCD-GOLD-03, 04).
   The evidence text is at the end of a pharmacokinetics-heavy chunk.

3. **P283-ARIPIPRAZOLE-NBN is missed in 2 cases** (OCD-GOLD-04, NBN-GOLD-02).
   The anchor is in chunk 7629, which is consistently ranked just
   outside the top 10.

4. **The dominant failure mode is Type 1 (semantic gap):** 8 of 13
   missed evidence items are not even in the top 20. This means the
   question language fundamentally doesn't match the evidence language
   under vector similarity.

5. **Chunk boundary splits (Type 3) are a secondary but fixable issue:**
   3 cases have the right page in the top 10 but the wrong chunk. A
   smaller chunk size or overlap adjustment could help, but the primary
   issue is still ranking.

6. **No false positives:** The matcher correctly rejects chunks on the
   right page that don't contain the anchor text. The page+anchor
   contract is working as designed.
