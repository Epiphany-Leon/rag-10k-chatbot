# 📊 10-K RAG Studio

A **Cherry-Studio-inspired RAG chatbot** that answers questions about the latest
10-K annual reports of **Alphabet, Amazon, Microsoft, Apple, and Tesla** —
grounded in the filings, with inline source citations.

Built for the *AI Essentials for Business* (JHU) final project: *Build and
Evaluate RAG Chatbots*. The app is a single, polished Streamlit interface where
you can **swap the LLM and embedding model and tune every RAG parameter live** —
which makes the assignment's "compare different settings" experiments a matter
of moving a slider.

> Retrieval-Augmented Generation = the model only answers from text retrieved
> out of the 10-Ks, so figures are traceable to a page instead of hallucinated.

---

## ✨ Features

- **5 model providers, one dropdown** — Google Gemini, OpenAI (GPT), DeepSeek,
  Kimi (Moonshot), and open-source models via Hugging Face. LLM and embedding
  model are chosen independently (e.g. Llama-3 answering over Gemini embeddings).
- **Live parameter controls** (the Cherry-Studio touch): temperature, top-p,
  max tokens, chunk size / overlap, top-k, MMR vs similarity, MMR λ, and an
  editable system prompt with presets.
- **Grounded answers with citations** — every reply cites `[Company p.N]` and
  the exact retrieved passages are shown in an expander.
- **Compare mode** — answer the same question with two models side-by-side over
  identical context. Direct evidence for the model-comparison write-up.
- **Company scoping** — restrict retrieval to any subset of the five filings.
- **Smart caching** — the FAISS index is keyed by `embedding + chunking +
  companies`, so switching the *LLM* is instant and only switching the
  *embedding/chunking* triggers a rebuild.
- **Evaluation harness** — `eval/run_eval.py` runs a verified question set
  through any configuration and grades answers with an LLM judge.

---

## 🏗️ Architecture

```
                ┌──────────────────────── Streamlit UI (app.py) ────────────────────────┐
                │  Sidebar: provider · model · embedding · temp/top-p · chunk · top-k    │
                │           persona editor · company scope · compare toggle              │
                └───────────────┬───────────────────────────────────────┬───────────────┘
                                │ question                               │ settings
                                ▼                                        ▼
   data/10k/*.pdf  ──▶  loaders.py  ──▶  indexer.py  ──▶  FAISS index (.cache/, persisted)
   (PyPDFLoader,        per-page         Recursive-                 │
    company+page         metadata        CharacterTextSplitter      │ retriever (MMR / similarity)
    metadata)                                                       ▼
                                                          chains.py: RagEngine
                                            grounding prompt + retrieved context ──▶ LLM ──▶ answer + citations
                                                                       ▲
                              providers.py  ── builds LLM / embeddings for any provider
```

- **Vector store:** FAISS (local, no server) — built once, persisted to `.cache/`.
- **Chunking:** `RecursiveCharacterTextSplitter`, defaults 1000 / 150.
- **Adding a model** = one line in `ragstudio/config.py`; nothing else changes.

---

## 🚀 Quickstart

```bash
# 1. Environment (conda recommended; a venv works too)
conda create -n chatbot python=3.11 -y
conda activate chatbot
pip install -r requirements.txt

# 2. API keys — copy the template and fill in at least one provider
cp .env.example .env
#   Google Gemini has a free tier: https://aistudio.google.com/app/apikey

# 3. (optional) pre-build the index so the first question is instant
python scripts/build_index.py

# 4. Run
streamlit run app.py
```

Open http://localhost:8501. Pick a provider/model in the sidebar and ask away.

---

## 🔑 Providers & keys

| Provider | Env var | Free tier | Notes |
|---|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | ✅ | LLM **and** embeddings; easiest start |
| OpenAI (GPT) | `OPENAI_API_KEY` | — | GPT-4o/4.1 + `text-embedding-3-*` |
| DeepSeek | `DEEPSEEK_API_KEY` | — | OpenAI-compatible, cheap |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` | — | OpenAI-compatible, long context |
| Zhipu GLM | `ZHIPU_API_KEY` | ✅ (`glm-4.5-flash`) | open-source GLM-4.5/4.6/5 (MIT), OpenAI-compatible |
| Hugging Face | `HUGGINGFACE_API_KEY` | ✅ (limited) | open-source LLMs + BGE/MiniLM embeddings |

You only need keys for the providers you actually use. Unconfigured providers
are hidden in the UI. **`.env` is git-ignored and never committed.**

---

## 🧪 Evaluation (Stage 2)

```bash
# Grade the default config against the verified question set
python eval/run_eval.py --tag gemini-flash

# A stronger setting…
python eval/run_eval.py --provider OpenAI --model gpt-4o \
    --embedding "OpenAI text-embedding-3-small" --tag gpt4o

# …vs a small open-source model on the same questions
python eval/run_eval.py --provider "Hugging Face (open-source)" \
    --model meta-llama/Llama-3.1-8B-Instruct --tag llama8b
```

Each run prints an accuracy score and writes `eval/results_<tag>.csv`. The
question set with checked ground-truth answers lives in `eval/questions.yaml`.

---

## 🌐 Deploy to a server

See [`deploy/README_DEPLOY.md`](deploy/README_DEPLOY.md). One command on an
Ubuntu/Debian box sets up a venv, systemd service, and nginx reverse proxy:

```bash
REPO=https://github.com/Epiphany-Leon/rag-10k-chatbot.git DOMAIN=your.ip bash deploy/deploy.sh
```

---

## 📁 Project structure

```
rag-10k-chatbot/
├── app.py                 # Streamlit UI (Cherry-Studio-style)
├── ragstudio/
│   ├── config.py          # model + embedding registry, defaults  ← add models here
│   ├── providers.py       # LLM / embedding factories (all providers)
│   ├── loaders.py         # load 10-K PDFs with company+page metadata
│   ├── indexer.py         # chunk → embed → FAISS (cached per settings)
│   └── chains.py          # RagEngine: grounded prompt + citations
├── prompts/personas.py    # system-prompt presets
├── eval/                  # question set + evaluation harness
├── scripts/build_index.py # CLI index builder
├── deploy/                # systemd + nginx + deploy.sh
├── data/10k/              # the five 10-K PDFs (public SEC filings)
├── requirements.txt
└── TECHNOTE.md            # approach, results, insights (graded deliverable)
```

---

## 🔒 Security note

Never commit API keys. Keys live only in your local `.env` (git-ignored). If a
key is ever exposed, rotate it in the provider console.

## 📜 Data

The 10-K filings are public documents filed with the U.S. SEC, included here for
reproducibility and educational use.
