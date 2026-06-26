"""
The RAG chain, composed by hand (LangChain 1.x / LCEL style) so we keep full
control over retrieval, the grounding prompt, and source citations.

Flow:  question -> retriever -> context -> grounded prompt -> LLM -> answer
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import COMPANIES

# Always appended to the user's persona so the model stays grounded in the
# retrieved 10-K text, regardless of how the persona is edited in the UI.
GROUNDING_RULES = (
    "You answer ONLY using the context from the companies' 10-K filings shown "
    "below. If the context does not contain the figures needed, reply exactly: "
    "\"I don't have enough information in the provided 10-K filings to answer "
    "that.\" Never invent numbers. You MAY, however, COMPUTE a derived value "
    "from figures that ARE present in the context, for example working capital = "
    "total current assets − total current liabilities, a year-over-year change, "
    "a growth rate, or a ratio; show the arithmetic and cite the source lines. "
    "When you state a figure, cite its source inline like [Amazon p.4]. Be "
    "precise with financial figures and include the reporting period.\n\n"
    "Context from the 10-K filings:\n{context}"
)


def make_retriever(store, search_type: str, top_k: int,
                   fetch_k: int, mmr_lambda: float, companies=None):
    """Build a vector retriever, optionally scoped to a subset of companies."""
    search_kwargs: dict = {"k": top_k}
    if companies:
        # FAISS supports metadata filtering via a dict filter.
        search_kwargs["filter"] = {"company": {"$in": list(companies)}}
    if search_type == "mmr":
        search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": mmr_lambda})
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


QUERY_EXPANSION_PROMPT = (
    "You rewrite a question about company 10-K filings into search queries that "
    "use the exact wording found in financial statements. Output {n} short "
    "queries, one per line, no numbering, no commentary. Expand derived metrics "
    "into the line items they are computed from, e.g. 'working capital' -> "
    "'total current assets' and 'total current liabilities'; 'liquidity' -> "
    "'current assets', 'current ratio'; 'main source of revenue' -> 'revenue by "
    "segment', 'segment results'.\n\nQuestion: {q}"
)


class MultiQueryRetriever:
    """
    Wrap a retriever with LLM query expansion. The natural phrasing of a question
    ("how did working capital change?") often does not match the source text
    ("total current assets / liabilities"); expanding into financial-statement
    terms and unioning the results bridges that query-to-chunk gap (the residual
    miss identified in Stage-2).
    """

    def __init__(self, base, llm, n: int = 3, cap: int = 10):
        self.base, self.llm, self.n, self.cap = base, llm, n, cap

    def _expand(self, question: str) -> list[str]:
        try:
            text = self.llm.invoke(
                QUERY_EXPANSION_PROMPT.format(q=question, n=self.n)).content
        except Exception:  # noqa: BLE001 — fall back to the raw question
            return [question]
        lines = [ln.strip("-*•0123456789. \t") for ln in text.splitlines()]
        return [question] + [ln for ln in lines if ln][: self.n]

    def invoke(self, question: str) -> list[Document]:
        seen: dict[str, Document] = {}
        for q in self._expand(question):
            for d in self.base.invoke(q):
                seen.setdefault(d.page_content, d)
        return list(seen.values())[: self.cap]


def detect_companies(question: str, in_scope: list[str]) -> list[str]:
    """
    Return the in-scope companies named (or aliased) in the question — e.g.
    "Azure" → Microsoft, "Google" → Alphabet. Scoping retrieval to these stops
    a question about one company from pulling another's balance sheet (the
    cross-company contamination seen in Stage-2). Falls back to all in-scope
    companies when the question names none (e.g. a generic comparison).
    """
    ql = question.lower()
    hits = [name for name in in_scope
            if any(a in ql for a in COMPANIES[name].get("aliases", [name.lower()]))]
    return hits or list(in_scope)


def build_query_retriever(store, all_chunks, companies, *, search_type, top_k,
                          fetch_k, mmr_lambda, hybrid, dense_weight):
    """Build a retriever scoped to `companies` (vector + optional BM25)."""
    vr = make_retriever(store, search_type, top_k, fetch_k, mmr_lambda, companies)
    if not hybrid:
        return vr
    scoped = [d for d in all_chunks if d.metadata.get("company") in companies]
    return make_hybrid_retriever(vr, scoped or all_chunks, top_k, dense_weight)


def make_hybrid_retriever(vector_retriever, bm25_docs, top_k: int,
                          dense_weight: float = 0.6):
    """
    Combine BM25 (lexical) + the vector retriever (semantic) into an ensemble.

    10-K questions often hinge on exact line-item labels ("deferred revenue",
    "cash and cash equivalents") where lexical match matters, while comparative
    questions need semantics — hybrid retrieval covers both. `bm25_docs` should
    already be scoped to the selected companies.
    """
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = top_k
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[round(1 - dense_weight, 2), dense_weight],
    )


def _history_to_messages(history: list[dict]) -> list:
    """Convert Streamlit-style [{role, content}] into LangChain messages."""
    out = []
    for msg in history:
        if msg["role"] == "user":
            out.append(HumanMessage(msg["content"]))
        else:
            out.append(AIMessage(msg["content"]))
    return out


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, d in enumerate(docs, 1):
        company = d.metadata.get("company", "?")
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        blocks.append(f"[Source {i} — {company} 10-K, p.{page}]\n{d.page_content}")
    return "\n\n".join(blocks) if blocks else "(no relevant context found)"


class RagEngine:
    """Bundles a retriever + LLM + persona into a question-answering unit."""

    def __init__(self, llm, retriever, persona: str):
        self.llm = llm
        self.retriever = retriever
        self.persona = persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{persona}\n\n" + GROUNDING_RULES),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def _messages(self, question, history, docs):
        return self.prompt.format_messages(
            persona=self.persona,
            context=format_context(docs),
            chat_history=_history_to_messages(history),
            question=question,
        )

    def answer(self, question: str, history: list[dict],
               docs: list[Document] | None = None):
        """Return (answer_text, source_docs)."""
        docs = self.retrieve(question) if docs is None else docs
        resp = self.llm.invoke(self._messages(question, history, docs))
        return resp.content, docs

    def stream(self, question: str, history: list[dict], docs: list[Document]):
        """Yield answer tokens for a pre-retrieved set of docs."""
        for chunk in self.llm.stream(self._messages(question, history, docs)):
            text = getattr(chunk, "content", "")
            if text:
                yield text
