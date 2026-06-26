"""
FastAPI backend for the Next.js front-end. Reuses the exact same `ragstudio`
RAG pipeline as the Streamlit app (same 68% retrieval), exposing it as a
streaming HTTP API.

  GET  /api/config  -> providers, models, embeddings, defaults, personas
  POST /api/chat    -> Server-Sent-Events stream: scope, sources, tokens, done

Run:  uvicorn server.main:app --host 0.0.0.0 --port 8600
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts.personas import DEFAULT_PERSONA, PERSONAS          # noqa: E402
from ragstudio.chains import (MultiQueryRetriever, RagEngine,   # noqa: E402
                              build_query_retriever, detect_companies)
from ragstudio.config import (COMPANIES, CORE_COMPANIES, DEFAULTS,  # noqa: E402
                              PROVIDERS, available_embeddings,
                              available_providers)
from ragstudio.indexer import build_index, chunk_corpus         # noqa: E402
from ragstudio.providers import MissingKeyError, build_llm      # noqa: E402

app = FastAPI(title="10-K RAG Studio API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- caches (process-local; index build is the expensive bit) --------------
_stores: dict = {}
_chunks: dict = {}


def get_store(embedding: str, cs: int, co: int, companies: list[str]):
    key = (embedding, cs, co, tuple(sorted(companies)))
    if key not in _stores:
        _stores[key] = build_index(embedding, cs, co, list(companies))
    return _stores[key]


def get_chunks(cs: int, co: int, companies: list[str]):
    key = (cs, co, tuple(sorted(companies)))
    if key not in _chunks:
        _chunks[key] = chunk_corpus(list(companies), cs, co)
    return _chunks[key]


# ---- /api/config -----------------------------------------------------------
@app.get("/api/config")
def get_config():
    provs = available_providers()
    return {
        "providers": [
            {"name": n, "models": PROVIDERS[n]["models"], "note": PROVIDERS[n].get("note", "")}
            for n in provs
        ],
        "all_providers": [
            {"name": n, "models": PROVIDERS[n]["models"], "note": PROVIDERS[n].get("note", ""),
             "available": n in provs}
            for n in PROVIDERS
        ],
        "embeddings": available_embeddings(),
        "companies": list(COMPANIES),
        "core_companies": CORE_COMPANIES,
        "personas": PERSONAS,
        "default_persona": DEFAULT_PERSONA,
        "defaults": DEFAULTS,
    }


# ---- /api/chat (SSE) -------------------------------------------------------
class ChatRequest(BaseModel):
    provider: str
    model: str
    question: str
    history: list = []
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 1024
    embedding: str = DEFAULTS["embedding"]
    search_type: str = "mmr"
    top_k: int = 5
    fetch_k: int = 20
    mmr_lambda: float = 0.5
    hybrid: bool = True
    dense_weight: float = 0.6
    expand: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 150
    companies: list | None = None
    persona: str = ""


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    def stream():
        try:
            companies = req.companies or CORE_COMPANIES
            store = get_store(req.embedding, req.chunk_size, req.chunk_overlap, companies)
            chunks = get_chunks(req.chunk_size, req.chunk_overlap, companies)
            scope = detect_companies(req.question, companies)
            retriever = build_query_retriever(
                store, chunks, scope, search_type=req.search_type, top_k=req.top_k,
                fetch_k=req.fetch_k, mmr_lambda=req.mmr_lambda, hybrid=req.hybrid,
                dense_weight=req.dense_weight)
            llm = build_llm(req.provider, req.model, req.temperature, req.top_p,
                            req.max_tokens)
            if req.expand:
                retriever = MultiQueryRetriever(retriever, llm, n=3, cap=2 * req.top_k)
            persona = req.persona or PERSONAS[DEFAULT_PERSONA]
            engine = RagEngine(llm, retriever, persona)

            yield _sse({"type": "scope", "scope": scope})
            docs = engine.retrieve(req.question)
            yield _sse({"type": "sources", "sources": [
                {"company": d.metadata.get("company"),
                 "page": d.metadata.get("page_label"),
                 "kind": d.metadata.get("type", "text"),
                 "snippet": d.page_content.strip().replace("\n", " ")[:400]}
                for d in docs]})
            t0 = time.time()
            for token in engine.stream(req.question, req.history, docs):
                yield _sse({"type": "token", "text": token})
            yield _sse({"type": "done", "elapsed": round(time.time() - t0, 1),
                        "model": req.model, "provider": req.provider})
        except MissingKeyError as e:
            yield _sse({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return {"status": "ok"}
