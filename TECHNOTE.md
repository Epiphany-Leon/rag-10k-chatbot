# Tech Note — 10-K RAG Studio

*AI Essentials for Business · Final Project: Build and Evaluate RAG Chatbots*

---

## 1. Goal

Build a reliable, user-friendly chatbot that answers questions about the latest
10-K filings of Alphabet, Amazon, and Microsoft (with Apple and Tesla as bonus
context), and use it to study how different LLMs, embedding models, and RAG
settings change answer quality and hallucination.

## 2. Approach & architecture

A single Streamlit app wraps a Retrieval-Augmented-Generation pipeline:

1. **Load** — the 10-K PDFs are parsed page-by-page (`PyPDFLoader`); every page
   carries `{company, ticker, page}` metadata so retrieval can be scoped and
   every answer can be cited.
2. **Chunk** — `RecursiveCharacterTextSplitter`, default **chunk size 1000 /
   overlap 150**, split on paragraph → sentence → word boundaries.
3. **Embed + index** — chunks are embedded and stored in a **FAISS** index. The
   index is cached on disk keyed by `embedding model + chunk settings +
   companies`, so changing the LLM is free and only changing the embedding or
   chunking rebuilds it.
4. **Retrieve** — **hybrid retrieval** by default: an ensemble of **BM25**
   (lexical) + **MMR vector** search (default `k=5`, `fetch_k=20`, `λ=0.5`),
   optionally filtered to selected companies. Lexical match catches exact
   line-item labels ("cash and cash equivalents", "deferred revenue") that pure
   semantic search can miss; vectors catch paraphrased/comparative questions.
5. **Generate** — retrieved chunks are inserted into a grounding prompt and sent
   to the selected LLM; the answer cites `[Company p.N]`.

The design is **provider-agnostic**: LLM and embedding model are independent
dropdowns, and adding a model is one line in `ragstudio/config.py`.

### Why this design
- **FAISS** (local, no server) keeps deployment light and reproducible.
- **Independent LLM/embedding selection** is what makes Stage-2 comparisons cheap
  and is the core of the "try different settings" requirement.
- **Caching by content hash** means model comparisons don't pay re-embedding cost.

## 3. Model choices

| Role | Default | Alternatives explored |
|---|---|---|
| LLM | `gemini-2.0-flash` (free, fast, strong) | GPT-4o/4.1, DeepSeek-chat, Kimi, Llama-3 / Qwen (HF) |
| Embedding | `Gemini text-embedding-004` (768-d) | OpenAI `text-embedding-3-small/large`, BGE-small, MiniLM |

Strategy (per the assignment tips): start from the **most powerful setting**
(GPT-4o / OpenAI embeddings or Gemini) to establish the accuracy ceiling, then
swap in weaker / open-source models to measure the drop-off.

## 4. System prompt

The system prompt is editable in the UI (presets: *Financial Analyst*, *Concise
Assistant*, *Skeptical Auditor*, *Explainer*). Regardless of the chosen persona,
a fixed **grounding suffix** is always appended:

> Answer ONLY from the provided 10-K context. If the answer is not present, say
> so explicitly. Never invent numbers. Cite each figure as `[Company p.N]` with
> its reporting period.

This refuse-when-unsupported instruction is the single biggest lever against
hallucination on financial figures.

## 5. RAG parameters studied

- **chunk size / overlap** — 1000/150 default (~250 tokens). Finance-RAG
  research finds smaller, granular chunks (~400–512 tokens) plus higher top-k
  beat large chunks, which cause "context confusion"; large 14k-char chunks cost
  10–20% accuracy in one study.
- **embedding model** — defines the vector space; compared Gemini vs OpenAI vs
  open-source BGE/MiniLM. Domain/long-context embedders (e.g. BGE-M3) tend to
  win on long filings.
- **hybrid retrieval (BM25 + dense)** — research treats this as near-mandatory
  for 10-Ks because exact line-item lexical match matters; exposed as a toggle so
  we can measure vector-only vs hybrid.
- **top-k / MMR λ** — more chunks improve recall but dilute the prompt; keeping
  the final context tight (5–8 chunks) measurably reduces "distraction" errors.
- **temperature** — kept low (0.2) for factual consistency.

A fuller literature review (with sources) and the Cherry-Studio UX analysis that
shaped the sidebar are in [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md). The
verified reference figures behind the question set are in
[`docs/FACT_SHEETS.md`](docs/FACT_SHEETS.md).

## 6. Results — settings comparison (Stage 2)

> Populated from `eval/run_eval.py` over the verified question set in
> `eval/questions.yaml`. (Run the harness with your keys to fill this table.)

| Setting | LLM | Embedding | Score | Notes |
|---|---|---|---|---|
| Powerful | _TBD_ | _TBD_ | _TBD_ | baseline ceiling |
| Cheaper cloud | _TBD_ | _TBD_ | _TBD_ | |
| Open-source | _TBD_ | _TBD_ | _TBD_ | |

## 7. Boundary & hallucination findings (Stage 3)

> Document here the cases where the bot fails or hallucinates, and what fixed it.
> The question set includes deliberate `boundary` questions (figures the 10-K
> does not disclose, companies not in the corpus) to probe this.

- _Example to fill: asked for a metric not disclosed → model should refuse._
- _How we reduced hallucination: grounding suffix, lower temperature, citation
  requirement, top-k tuning._

## 8. Strengths & weaknesses

**Strengths:** grounded + cited answers; instant model/parameter switching;
company scoping; reproducible cached index; runs on a small server.

**Weaknesses (honest):** PDF table extraction can fragment numbers across chunks;
retrieval can miss a figure if it lives in a sparsely-worded table; open-source
small models follow the citation format less reliably.

## 9. Our challenge questions for other teams

Two questions designed to be hard for a naive RAG bot but answerable from the
filings (full ground truth in `eval/questions.yaml`):

- **Q11 (multi-doc reasoning):** *"Compare the year-over-year working capital
  trend for Alphabet, Amazon, and Microsoft. Which improved most and which was
  roughly flat?"* — requires pulling current-asset/liability figures from three
  separate filings and computing a trend (no filing states "working capital"
  outright). A weak retriever mixes companies; a weak model does the arithmetic
  wrong.
- **Q14 (false-premise trap):** *"What exact dollar amount does Apple's 10-K
  report on its 'working capital' line, and what current ratio does it disclose?"*
  — Apple's 10-K presents **neither** a labeled working-capital line nor a stated
  current ratio. The correct behaviour is to refuse the false premise and explain
  both must be computed; a hallucinating bot invents a number.

## 10. Team

| Member | Role |
|---|---|
| Weiting Xia | RAG pipeline & retrieval (chunking, FAISS, hybrid BM25+vector) |
| Jinfeng Chen | Model & provider integration (multi-LLM, embeddings) |
| Yiying Wang | Evaluation & question design (eval set, LLM-judge harness) |
| Wanying Li | 10-K analysis & ground truth (fact sheets, business insights) |
| Kaipu Liu | UI / front-end (Streamlit, parameter controls, compare mode) |
| Lihong Gao | Deployment & infrastructure (server, Caddy, GitHub) |

## 11. Reproducibility

- Code: this repository.
- Environment: `requirements.txt` (Python 3.11).
- Index: rebuildable via `python scripts/build_index.py`.
- Evaluation: `python eval/run_eval.py --tag <name>`.
