"""
Load the 10-K PDFs into LangChain Documents.

Two kinds of chunks are produced per filing:
  - **text**  : narrative pages via PyPDFLoader (risk factors, MD&A, business).
  - **table** : financial tables extracted with pdfplumber and *linearized* into
                "Row label: v1, v2" lines. PyPDF flattens tables into a stream
                where a figure drifts away from its label; the linearized form
                keeps each number next to its row label, so numeric questions
                ("cash and cash equivalents", "total current assets") retrieve
                and embed far better. This was the #1 bottleneck found in the
                Stage-2 evaluation.

Every chunk carries {company, ticker, page, page_label, type} metadata.
"""
from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import COMPANIES, DATA_DIR


def _linearize_table(table, company: str, page_label: int) -> str | None:
    """Turn a pdfplumber table (list of rows) into retrieval-friendly text."""
    rows = [[(c or "").strip() for c in row] for row in table]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    lines = [f"[{company} 10-K p.{page_label} — financial table]"]
    header = " | ".join(c for c in rows[0] if c)
    if header:
        lines.append(header)
    for r in rows[1:]:
        cells = [c for c in r if c]
        if not cells:
            continue
        label, values = cells[0], cells[1:]
        lines.append(f"{label}: {', '.join(values)}" if values else label)
    text = "\n".join(lines)
    return text if len(text) > 40 else None


def _table_docs(name: str, ticker: str, source: str, path) -> list[Document]:
    """Extract and linearize every financial table in the filing."""
    import pdfplumber

    docs: list[Document] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                for table in (page.extract_tables() or []):
                    text = _linearize_table(table, name, i + 1)
                    if text:
                        docs.append(Document(page_content=text, metadata={
                            "company": name, "ticker": ticker, "source": source,
                            "page": i, "page_label": i + 1, "type": "table",
                        }))
    except Exception:  # noqa: BLE001 — tables are a bonus; never fail the load
        return []
    return docs


def load_company(name: str) -> list[Document]:
    """Load one company's 10-K as narrative-text + linearized-table chunks."""
    spec = COMPANIES[name]
    path = DATA_DIR / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing 10-K for {name}: {path}")

    docs: list[Document] = []
    # 1. Narrative text (PyPDF is reliable for prose).
    for page in PyPDFLoader(str(path)).load():
        page.metadata["company"] = name
        page.metadata["ticker"] = spec["ticker"]
        page.metadata["source"] = spec["file"]
        page.metadata["type"] = "text"
        page.metadata["page_label"] = page.metadata.get("page", 0) + 1
        docs.append(page)
    # 2. Linearized financial tables (the numeric retrieval boost).
    docs.extend(_table_docs(name, spec["ticker"], spec["file"], path))
    return docs


def load_corpus(companies: list[str] | None = None) -> list[Document]:
    """Load every requested company's 10-K into one flat list."""
    names = companies or list(COMPANIES.keys())
    docs: list[Document] = []
    for name in names:
        docs.extend(load_company(name))
    return docs
