# Stage-2 Settings Comparison

14-question verified set (3 of them deliberate boundary/refusal traps). Hybrid retrieval (BM25 + MMR vector). Judge = glm-4-flash, same grader across all settings, graded against the checked ground truth.

| # | LLM | Embedding | top-k | chunk | Score | ✓/◐/✗ | Note |
|---|---|---|---|---|---|---|---|
| A | glm-4.6 | embedding-3 | 6 | 1000 | **46%** | 6/1/7 | strong LLM baseline |
| B | glm-4-flash | embedding-3 | 6 | 1000 | **43%** | 6/0/8 | swap to free/fast LLM |
| C | glm-4-flash | embedding-3 | 12 | 1000 | **43%** | 6/0/8 | same model, wider retrieval |
| D | glm-4-flash | embedding-3 | 6 | 2000 | **46%** | 6/1/7 | same model, larger chunks (tables intact) |

## Findings (read straight off the table)
1. **LLM strength is not the bottleneck.** A→B: glm-4.6 46% → free glm-4-flash 43%. A 600B-class model and a small free model perform within a few points, so reasoning is not what's failing.
2. **More retrieval doesn't help — it redistributes.** B→C: top-k 6→12 went 43% → 43%; individual questions flipped both ways (added context displaced as many answers as it surfaced — the documented 'distraction' effect).
3. **Chunk size helps.** B→D: chunk 1000→2000 went 43% → 46% (+4%). Larger chunks keep more of a financial table together.
4. **The real wall is PDF table extraction.** Most misses are the bot *correctly refusing* ("I don't have enough information") because the segment-revenue / balance-sheet numbers are garbled when flattened from PDF and never embed into a retrievable chunk. The fix is table-aware parsing (pdfplumber / unstructured) + a stronger embedding — not a bigger LLM.
5. **Anti-hallucination holds.** Across every setting the out-of-corpus probes (Q12 Tesla-cloud, Q13 Meta) are refused, never fabricated. The bot fails *safe*.

*Score = (Correct + 0.5·Partial)/N over 14 questions; the set intentionally includes hard multi-company and trap questions, so absolute scores are conservative. Per-question detail in `eval/results_<tag>.csv`.*
