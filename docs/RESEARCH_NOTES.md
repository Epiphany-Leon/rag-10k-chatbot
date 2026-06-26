# Research Notes

*Background research that informed the design and RAG configuration of 10-K RAG Studio. Compiled with web research; sources are linked inline.*

---

# Cherry Studio UI/UX Patterns → Streamlit Financial-RAG Chatbot

## What Cherry Studio Does Well

### 1. Layout: narrow icon rail + content panel + footer bar
A thin left **icon rail** (chat, assistants, knowledge base, settings, files) keeps top-level navigation always visible without stealing horizontal space. The wide center panel holds the active view, and a persistent **footer bar** carries low-frequency global controls (theme toggle, language, quick preferences). Settings live in a dedicated full panel reached via a gear icon, separate from the chat flow so the conversation stays uncluttered.

### 2. Two-tier conversation model: Assistants → Topics
The killer organizational idea. An **Assistant** is a reusable preset bundling a name, system prompt, default model, default temperature, attached knowledge base, and attached tools. Under each Assistant you spawn multiple **Topics** (independent conversations) that inherit the Assistant's persona and parameters, with strict **context isolation** so one Topic never bleeds into another. You configure the role once, then start parallel threads cheaply.

### 3. Model/provider switching
Providers are configured once in Settings → Model Services (API key, base URL, enabled models). Switching is then frictionless: a model picker per Assistant plus inline **`@`-mention** switching mid-conversation, so you can flip models within a single thread. Side-by-side multi-model comparison is offered for evaluating answer quality.

### 4. Parameter controls
Exposed but defaulted sanely, attached to the Assistant rather than scattered:
- **Temperature** — slider 0–1, default 0.7, labeled by behavior (low = accurate, high = creative)
- **Top-P** — default 1, with guidance ranges
- **Max tokens** — per-response cap, with task-based suggested ranges
- **Context window** — how many prior messages to retain (5–10 typical)

Each control carries a plain-language hint instead of a bare number.

### 5. Input box and message toolbar
A left-side toolbar on the composer groups actions: new topic, attach file, web search toggle, knowledge-base toggle, MCP/tools, `@`-model mention, quick phrases, clear context. The input box shows a **live token estimate** (current / max) in the corner, and supports `@` for model selection and `/` for slash commands. Long pasted text auto-converts to a file chip.

### 6. Source/citation display (RAG)
Retrieved chunks are attached **below each answer** as **clickable source references** that jump to the origin file. This is the transparency pattern: concise answer on top, verifiable sources underneath, one click to the document.

### 7. Visual theming
Light / dark / system themes, full Markdown rendering (code blocks, tables, math), bubble-vs-list message styles, and drag-and-drop organization with smart auto-naming of conversations.

---

## Recommendations for a Streamlit Financial-RAG Sidebar

Streamlit gives you one sidebar plus the main pane, so concentrate Cherry Studio's "settings rail + presets" ideas into the sidebar and reserve the main pane for chat + sources.

1. **Group the sidebar into labeled sections with `st.expander`**: "Model", "Generation", "Knowledge Base", "Conversation". Keep advanced controls collapsed by default so the first impression stays calm, mirroring Cherry Studio's settings-hidden-until-needed posture.

2. **Provider/model switcher at the top**: one `st.selectbox` for provider, a dependent `st.selectbox` for model. Store the API key via `st.secrets` and show a green "Connected" / red "Key missing" status caption so connection state is glanceable.

3. **Ship "Analyst Presets" as the Assistant analog**: a `st.selectbox` of finance-tuned presets (e.g. "Equity Research", "10-K Summarizer", "Earnings-Call Q&A", "Risk & Compliance") where each preset sets the system prompt, default model, temperature, and which knowledge base is attached. Selecting one repopulates all downstream controls.

4. **Temperature slider defaulted to 0.2 with a behavior label**: financial answers favor determinism, so `st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)` plus a caption ("lower = more precise, factual") borrowed from Cherry Studio's plain-language hinting.

5. **Expose Top-P, Max tokens, and Context window under an "Advanced" expander**: sensible defaults (top_p 1.0, max_tokens 1500, last 8 messages), each with a one-line `help=` tooltip so power users can tune without confronting beginners with knobs.

6. **Knowledge-base scope control**: a `st.multiselect` to pick which document sets are in play (e.g. "10-K filings", "Earnings transcripts", "Internal memos"), plus a `st.slider` for "Sources to retrieve (top-k)" defaulted to 4. This makes retrieval scope explicit and auditable, which matters in a finance context.

7. **Render citations below every answer in the main pane**: under each assistant reply, show retrieved chunks as an `st.expander("Sources (4)")` containing document name, page/section, ticker or filing date, and a relevance score. Make each source title a link or a button that reveals the chunk text. This is the single most important Cherry Studio pattern to copy for a financial-RAG tool, where traceability is non-negotiable.

8. **Conversation management in the sidebar**: a "New chat" button at top and a list of past sessions stored in `st.session_state` (auto-named from the first user message). Add a small clear-context / reset button. This recreates Topic isolation within Streamlit's single-pane limits.

9. **Live token / cost meter** beneath the chat input using `st.caption`: show estimated context tokens and an approximate per-query cost, echoing Cherry Studio's in-composer token counter. Useful for cost-conscious financial teams.

10. **Theme and density via `st.set_page_config` plus a `.streamlit/config.toml` theme**: a polished financial palette (deep navy or charcoal base, single accent for actions, restrained semantic green/red reserved strictly for gains/losses so color stays meaningful). Add a light/dark toggle if feasible.

11. **Confidence and grounding signal**: when retrieval similarity is weak, surface an `st.warning("Low-confidence: answer may not be fully grounded in your documents")` above the reply. Cherry Studio leans on visible sourcing; for finance, add an explicit grounding caveat to discourage acting on hallucinated figures.

12. **Quick-action prompt chips** under the input (rendered as a row of `st.button`s): "Summarize this filing", "Key risks", "Revenue trend", "Compare to last quarter". This mirrors Cherry Studio's quick-phrases and lowers the cold-start cost for finance users.

---

**Sources:**
- [Chat Interface — Cherry Studio Docs](https://docs.cherry-ai.com/docs/en-us/cherry-studio/preview/chat)
- [Knowledge Base Tutorial — Cherry Studio Docs](https://docs.cherry-ai.com/docs/en-us/knowledge-base/knowledge-base)
- [Cherry Studio: The Ultimate AI Desktop Client — andrew.ooo](https://andrew.ooo/posts/cherry-studio-unified-ai-desktop-client/)
- [Cherry Studio Complete Guide 2026 — Codersera](https://codersera.com/blog/cherry-studio-complete-guide-2026/)
- [Cherry Studio In-Depth Review — Skywork](https://skywork.ai/skypage/en/Cherry-Studio-An-In-Depth-Review-of-the-All-in-One-AI-Desktop-Client/1972882990813605888)
- [Cherry Studio — GitHub](https://github.com/cherryhq/cherry-studio)

---

# RAG Best Practices for QA over SEC 10-K Filings

## 1. Document Parsing (do this before chunking)
- **Parse from EDGAR HTML / iXBRL, not flattened PDF.** EDGAR HTML preserves hierarchical section structure (Item 1–15) and table cell hierarchies; PDF flattening destroys row/column relationships early and is the single biggest source of downstream errors.
- **Preserve tables as Markdown (or HTML).** Convert each table to a Markdown key-value/row-linearized representation so both the embedder and the LLM see coherent column-header → cell-value structure. Structure-aware linearization consistently beats fixed-window chunking on tabular questions.
- **Attach document-level metadata to every chunk:** `company`, `ticker`, `fiscal_year`, `filing_date`, `form_type`, `item/section`. Injecting document context yielded **+20–25 percentage points** accuracy in the Snowflake finance-RAG study and is the highest-ROI single change.

## 2. Chunking Strategy
- **Use structure-aware chunking, in priority order:** (1) Markdown-header / Item-boundary splitting → (2) recursive character splitting within sections → avoid pure semantic chunking (highest compute cost, weakest results in finance benchmarks).
- **Never split a table across chunks.** Keep each table as one atomic chunk; if oversized, split by row groups and repeat the header row in each piece.
- **Table-summary pattern:** keep the full table as the retrievable payload, but embed an LLM-generated 1–2 sentence summary of the table for retrieval. Convert figures/charts to text captions via a VLM if present.

## 3. Chunk Size & Overlap Trade-offs
| Regime | Chunk size | Overlap | Notes |
|---|---|---|---|
| Snowflake finance study (best general) | ~1,800 chars (≈350–450 tokens) | ~300 tokens | Larger chunks (≈14,400 chars) caused "context confusion," **−10–20%** accuracy |
| FinSage (precision-focused) | 200–256 tokens | section-summary appended | Small chunks + metadata context maximize precision |

- **Recommendation:** start at **~400–512 tokens with ~15–20% overlap (75–100 tokens)** for prose; **0 overlap but full-table atomic chunks** for tables. Smaller, granular chunks + high top-k beat large chunks; granular retrieval narrowed the quality gap between weak and strong generator models.

## 4. Embedding Model Choice
| Option | When to pick | Specs / cost |
|---|---|---|
| **BGE-M3** (open source) | Long financial sections, on-prem/privacy, high volume (>10M emb/mo) | 8K context (embed whole sections), strong long-doc + hybrid (dense+sparse) support; used by FinSage |
| **Qwen3-Embedding-8B** (open) | Top accuracy, have GPU infra | Leads MTEB (~70.6), beats OpenAI/Gemini in 2025–26 |
| **OpenAI text-embedding-3-large** | Fast start, low/medium volume, no infra | $0.13/1M tok; solid but stale (no update since Jan 2024) |
| **Gemini Embedding** | Cheap managed API | ~$0.025/1M tok, mid-pack quality |
- **Recommendation:** **BGE-M3** is the best default for 10-Ks (long context, hybrid retrieval, free, finance-validated). Use OpenAI/Gemini only if you want a managed API and have low volume. Domain-tuned embedders typically beat generic ones by 10–15% in-domain.

## 5. Retriever Tuning
- **Hybrid retrieval (BM25 sparse + dense) is mandatory for 10-Ks.** Exact-match lexical (ticker names, line-item labels like "deferred revenue") needs BM25; semantics need dense. Retriever + similarity-metric choice dominates overall system performance.
- **Top-k:** retrieve generously then rerank. Snowflake found **top-50 retrieval** materially improved accuracy with granular chunks. Practical: **retrieve k=20–50 → rerank → keep top 5–8** for the LLM context.
- **Add a cross-encoder reranker** (e.g., FlashRank / Cohere Rerank / BGE-reranker). FlashRank on financial reports raised factual alignment **+0.9 pts** and term coverage **+12%**. Cap rerank depth (~top-50 in) for latency.
- **MMR vs pure similarity:** use **MMR** (λ ≈ 0.5–0.7) when chunks are redundant (boilerplate risk-factor repetition) to diversify context; use **pure similarity + reranker** for precise single-fact numeric lookups. Reranking generally beats MMR for factual accuracy; MMR helps multi-aspect/comparative questions.
- **Advanced (FinSage multi-path):** combine BM25 + BGE-M3 dense + metadata-summary retrieval + HyDE; add a recency bonus favoring the latest filing. This reached 92.5% recall / 88% accuracy on a company dataset.

## 6. Reducing Hallucination on Financial Figures
- **Constrain generation strictly to retrieved context;** instruct the model to answer "not found in the provided filings" when the figure is absent. RAG grounding cuts hallucination **42–68%**.
- **Require inline citations** (chunk/section + filing) for every number; reject/flag answers whose figures don't appear verbatim in retrieved chunks.
- **Do not let the LLM do arithmetic.** Extract raw line items, then compute ratios/growth/CAGR in code (a tool/function call). LLM free-form math is the top financial error class (numerical miscalculation).
- **Watch temporal/entity confusion:** stamp every chunk with fiscal year + company so "2023 vs 2024" or two co-mentioned subsidiaries don't get mixed (temporal inconsistency + incorrect entity reference are documented finance-RAG error categories).
- **Beat "distraction":** keep final context tight (5–8 chunks). Extra retrieved passages measurably degrade financial reasoning even when grounding is correct.
- **Add a self-check / verifier pass** (or confidence field) that validates each emitted figure against the source chunk before returning.

## 7. Handling Numeric / Tabular Data
- Linearize tables to Markdown key-value rows; preserve units, scale ("in millions"), and currency in the chunk text.
- Store extracted numerics as **structured fields/metadata** so exact filters (year, metric) and code-side computation are possible (hybrid RAG + structured query).
- Embed a table summary for recall, retrieve the full table for the answer.
- For multi-table comparison questions, retrieve per-year tables separately and let code align them.

## 8. Evaluating Answer Accuracy
- **RAGAS metrics** (LLM-as-judge): **Faithfulness** (figures grounded in context), **Answer Relevancy**, **Context Precision**, **Context Recall**. RAGAS agreed with human judges ~95% (faithfulness), 78% (relevance), 70% (context).
- **For numeric short-answer QA, also use Exact Match + token-level F1** against gold figures (used by FinanceBench evaluators) since faithfulness alone misses wrong-but-grounded numbers.
- **Benchmark on FinanceBench / FinDER** (real 10-Ks). Reference bar: FinSage scored 49.66% accuracy on FinanceBench (hard benchmark) vs much higher on curated company sets, so calibrate expectations to your question difficulty.
- **Separate retrieval eval from generation eval:** track recall@k / context-recall (did we fetch the right chunk?) independently from answer correctness, so you know whether failures are retrieval or generation.

## Quick-Start Parameter Defaults
- Parse: EDGAR HTML → Markdown, tables atomic, metadata on every chunk
- Chunk: ~400–512 tokens prose, ~15–20% overlap; tables whole
- Embed: BGE-M3 (or OpenAI text-embedding-3-large for managed)
- Retrieve: hybrid BM25+dense, k=20–50 → cross-encoder rerank → top 5–8
- Generate: context-only prompt + citations; compute math in code
- Evaluate: RAGAS (faithfulness/precision/recall) + EM/F1 on numeric golds, on FinanceBench

## Sources
- [Snowflake — Long-Context Isn't All You Need: Retrieval & Chunking in Finance RAG](https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/)
- [FinSage: A Multi-aspect RAG System for Financial Filings QA (arXiv)](https://arxiv.org/html/2504.14493v3)
- [Optimizing Retrieval Strategies for Financial QA in RAG (arXiv)](https://arxiv.org/pdf/2503.15191)
- [NVIDIA — Finding the Best Chunking Strategy for Accurate AI Responses](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- [Unstructured — Preserving Table Structure for Better Retrieval](https://unstructured.io/blog/preserving-table-structure-for-better-retrieval)
- [Embedding Models 2026: Benchmark and Comparison (Ailog)](https://app.ailog.fr/en/blog/news/embedding-models-2026)
- [FlashRank Reranking & Query Expansion for RAG (arXiv)](https://arxiv.org/pdf/2601.03258)
- [FinDER: Financial Dataset for QA and Evaluating RAG (arXiv)](https://arxiv.org/pdf/2504.15800)
- [FRED: Financial Retrieval-Enhanced Detection of Hallucinations (arXiv)](https://arxiv.org/pdf/2507.20930)
- [RAGAS Evaluation Metrics Guide (Deepchecks)](https://deepchecks.com/rag-evaluation-metrics-answer-relevancy-faithfulness-accuracy/)
- [Daloopa — Processing Tabular Financial Data with LLMs](https://daloopa.com/blog/analyst-best-practices/processing-tabular-financial-data-with-large-language-models)
