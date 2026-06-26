"""
Build (and cache) the FAISS vector index.

The cache key is a hash of everything that changes the vectors:
embedding model + chunk size + chunk overlap + which companies are included.
Switching the *LLM* does NOT invalidate the index (same vectors), so model
comparison is fast; switching the *embedding* or chunking rebuilds it.
"""
from __future__ import annotations

import hashlib
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import CACHE_DIR, CORPUS_VERSION, EMBEDDINGS
from .loaders import load_corpus
from .providers import build_embeddings


def _index_key(embedding: str, chunk_size: int, chunk_overlap: int,
               companies: list[str]) -> str:
    raw = (f"{CORPUS_VERSION}|{embedding}|{chunk_size}|{chunk_overlap}|"
           f"{','.join(sorted(companies))}")
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    safe = embedding.replace(" ", "_").replace("/", "-")
    return f"{safe}__cs{chunk_size}_co{chunk_overlap}__{digest}"


def _chunks_path(companies: list[str], chunk_size: int, chunk_overlap: int):
    raw = (f"{CORPUS_VERSION}|chunks|{chunk_size}|{chunk_overlap}|"
           f"{','.join(sorted(companies))}")
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"chunks__cs{chunk_size}_co{chunk_overlap}__{digest}.json"


def chunk_corpus(companies: list[str], chunk_size: int, chunk_overlap: int):
    """
    Load the 10-Ks and split them into chunks, persisting the result to disk as
    JSON. PDF parsing + table extraction (pdfplumber) is expensive, so caching
    the chunks means a cold process (or a small server) never re-parses the PDFs.
    """
    path = _chunks_path(companies, chunk_size, chunk_overlap)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return [Document(page_content=d["c"], metadata=d["m"]) for d in data]
        except Exception:  # noqa: BLE001 — fall through and rebuild
            pass
    docs = load_corpus(companies)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    try:
        path.write_text(json.dumps(
            [{"c": d.page_content, "m": d.metadata} for d in chunks]))
    except Exception:  # noqa: BLE001
        pass
    return chunks


def build_index(embedding: str, chunk_size: int, chunk_overlap: int,
                companies: list[str], force: bool = False):
    """
    Return a FAISS store for the given settings, loading from disk if a
    matching index was built before, otherwise building and persisting it.
    """
    if embedding not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding: {embedding}")

    key = _index_key(embedding, chunk_size, chunk_overlap, companies)
    path = CACHE_DIR / key
    emb = build_embeddings(embedding)

    if path.exists() and not force:
        return FAISS.load_local(
            str(path), emb, allow_dangerous_deserialization=True,
        )

    chunks = chunk_corpus(companies, chunk_size, chunk_overlap)
    store = FAISS.from_documents(chunks, emb)
    store.save_local(str(path))
    return store


def count_chunks(companies: list[str], chunk_size: int, chunk_overlap: int) -> int:
    """Number of chunks the current chunk settings produce (for the UI)."""
    return len(chunk_corpus(companies, chunk_size, chunk_overlap))
