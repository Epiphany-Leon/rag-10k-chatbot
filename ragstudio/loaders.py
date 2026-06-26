"""
Load the 10-K PDFs into LangChain Documents, tagging every page with the
company name and ticker so the retriever can be scoped per company.
"""
from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import COMPANIES, DATA_DIR


def load_company(name: str) -> list[Document]:
    """Load one company's 10-K, attaching {company, ticker, page} metadata."""
    spec = COMPANIES[name]
    path = DATA_DIR / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing 10-K for {name}: {path}")

    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata["company"] = name
        page.metadata["ticker"] = spec["ticker"]
        page.metadata["source"] = spec["file"]
        # PyPDFLoader pages are 0-indexed; show humans a 1-based page number.
        page.metadata["page_label"] = page.metadata.get("page", 0) + 1
    return pages


def load_corpus(companies: list[str] | None = None) -> list[Document]:
    """Load every requested company's 10-K into one flat list of pages."""
    names = companies or list(COMPANIES.keys())
    docs: list[Document] = []
    for name in names:
        docs.extend(load_company(name))
    return docs
