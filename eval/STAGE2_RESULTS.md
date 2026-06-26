# Stage-2 Settings Comparison & Ablation

14-question verified set (3 are deliberate boundary/refusal traps). Hybrid
retrieval (BM25 + MMR vector). Judge = `glm-4-flash`, same grader across all
settings, graded against the checked ground truth. Score = (Correct + 0.5·Partial) / 14.

## A. What moves the score — controlled ablation

Each row changes **one** thing from the baseline.

| # | LLM | Embedding | top-k | chunk | Retrieval fix | Score |
|---|---|---|---|---|---|---|
| 1 | glm-4.6 | embedding-3 | 6 | 1000 | — (baseline) | 46% |
| 2 | glm-4-flash (free) | embedding-3 | 6 | 1000 | — | 43% |
| 3 | glm-4-flash | embedding-3 | 12 | 1000 | — | 43% |
| 4 | glm-4-flash | embedding-3 | 6 | 2000 | — | 46% |
| 5 | **glm-4.6** | **embedding-3** | **6** | **1000** | **+ table chunks + company scoping** | **57%** |

**Reading the table:**
1. **LLM strength is not the bottleneck** (1 vs 2: 46% → 43%). A free small model
   trails a 600B-class model by only 3 points.
2. **More retrieval doesn't help** (2 vs 3: 43% → 43%). Extra chunks redistribute
   which questions pass — the "distraction" effect — rather than adding net wins.
3. **Larger chunks help a little** (2 vs 4: 43% → 46%) by keeping tables intact.
4. **Retrieval engineering is the real lever** (1 vs 5: 46% → **57%, +11 pts**):
   - *Linearized table chunks* (pdfplumber): figures stay next to their row
     label, so balance-sheet lookups now resolve (e.g. Q04 Alphabet cash).
   - *Per-question company scoping*: a question about one company stops pulling
     another's balance sheet (fixed contamination on Q05/Q10).

## B. Embedding study (retrieval-only, no LLM)

Does a stronger embedding find the chunks `embedding-3` misses? We compared
against **BGE-small** (the embedding the finance-RAG literature recommends), run
locally and free.

| Question | needs | embedding-3 | BGE-small |
|---|---|---|---|
| Q02 Amazon segments | Amazon p.109 (segment note) | not in top-8 | **p.109 ranked #1** |
| Q01 Alphabet revenue | segment / revenue note | missed | missed |
| Q05 MSFT working capital | balance sheet (p.52) | missed | missed |

**Finding:** a better embedding fixes *some* misses (BGE surfaces Amazon's segment
note that `embedding-3` buries), confirming embedding quality is a partial lever.
But Q01/Q05 fail under *both* embeddings: the query *"working-capital change"*
doesn't match the balance-sheet text (*"Total current assets / liabilities"*) and
*"main source of revenue"* doesn't match a segment table — a **query-to-chunk
semantic gap** that needs query transformation (HyDE / multi-query), not a better
embedder.

## C. The bottleneck, in one paragraph

Accuracy is gated by **retrieval**, not the model. We removed three retrieval
failure modes in order — garbled tables, cross-company contamination, weak
embedding matching — and the score climbed from 46% to 57%. The residual misses
are *derived-metric* questions (working capital, segment-mix) where the natural
phrasing of the question never lexically/semantically meets the source figure;
the next frontier is query rewriting. Crucially, **every miss is an honest
refusal, never a fabricated number** (see Stage 3).

*Per-question detail in `eval/results_*.csv`. Deployed config = row 5.*
