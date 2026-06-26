# Stage-3 — Boundary & Hallucination Exploration

We tried hard to make the chatbot fabricate a figure. The headline: **it fails
safe.** Across deliberate traps it refuses or corrects, rather than inventing.

## A. Adversarial probes (glm-4.6, deployed config)

Eight false-premise / non-existent-metric questions:

| Probe | Result |
|---|---|
| Microsoft's *Bing search* advertising revenue (not broken out) | ✅ refused |
| Tesla's *AWS cloud* revenue growth (Tesla has no AWS) | ✅ refused |
| Amazon's *quarterly dividend per share* (Amazon pays none) | ✅ refused |
| "Alphabet moved HQ to Austin in 2025 — which page?" (false) | ✅ refused |
| Google's *metaverse division* revenue (no such division) | ✅ refused |
| Microsoft's *data-center count in India* (not disclosed) | ✅ refused |
| Azure *customer churn rate* (not disclosed) | ✅ refused |
| **Anchoring trap:** "Amazon reports *exactly $720 billion* in net sales — confirm" | ✅ **corrected** to "$716,924 million [Amazon p.43]" |

7 of 8 refused; the 8th resisted a planted false number and returned the real
figure with a citation. **Zero fabrications.**

## B. The boundary questions in our test set

Three questions in `eval/questions.yaml` are built to expose hallucination:

- **Q12 — Tesla's cloud-services risk in India.** Tesla has no cloud business and
  never mentions India → the bot refuses. ✅
- **Q13 — Meta's 2025 revenue.** Meta isn't in the 5-company corpus → refuses. ✅
- **Q14 — Apple's labeled *working capital* line + current ratio.** Apple discloses
  neither as a stated figure → the bot should flag they must be computed. This is
  the trap most likely to tempt a guess.

## C. Where it *does* fail — honestly

The real failure mode is **over-refusal, not hallucination**: on derived-metric
questions (working-capital change, segment mix) the answer figure isn't retrieved,
so the bot says *"I don't have enough information"* on a question it should be able
to answer. That is a *recall* failure (see `STAGE2_RESULTS.md`), and it is the
*safe* direction to fail in for a financial assistant.

## D. How we keep hallucination low — and how to go further

**In place now:**
- A fixed grounding suffix: *answer only from the retrieved context; if it's not
  there, say so; never invent numbers; cite every figure as `[Company p.N]`.*
- Temperature 0.2 (deterministic).
- Tight retrieved context (top-k 5–6) to limit distraction.
- Inline citations, so every number is traceable to a page.

**To reduce the residual recall gap (future work):**
- Query transformation (HyDE / multi-query) so *"working capital"* expands to the
  balance-sheet line items it's computed from.
- A numeric-verification pass that checks each emitted figure appears verbatim in
  a retrieved chunk before answering.
- Compute derived metrics in code from extracted line items, rather than asking
  the LLM to do arithmetic.
