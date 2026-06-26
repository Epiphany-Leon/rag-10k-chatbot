"""
10-K RAG Studio — a Cherry-Studio-inspired RAG chatbot over 10-K filings.

Run:  streamlit run app.py
Keys: copy .env.example -> .env and fill in at least one provider key.
"""
from __future__ import annotations

import time

import streamlit as st

from prompts.personas import DEFAULT_PERSONA, PERSONAS
from ragstudio.chains import (MultiQueryRetriever, RagEngine,
                              build_query_retriever, detect_companies)
from ragstudio.config import (
    COMPANIES, CORE_COMPANIES, DEFAULTS, EMBEDDINGS, PROVIDERS,
    available_embeddings, available_providers,
)
from ragstudio.indexer import build_index, chunk_corpus, count_chunks
from ragstudio.providers import MissingKeyError, build_embeddings, build_llm

st.set_page_config(page_title="10-K RAG Studio", page_icon="📊", layout="wide")

# Finance starter questions shown as quick-action chips on an empty chat.
STARTERS = [
    "What is the main source of revenue for Alphabet, Amazon, and Microsoft?",
    "How much cash did Amazon hold at the end of its latest fiscal year?",
    "Compare the three companies by total revenue and net income.",
    "What does Amazon's 10-K say about business risk in China and India?",
]


# --------------------------------------------------------------------- caching
@st.cache_resource(show_spinner=False)
def get_store(embedding: str, chunk_size: int, chunk_overlap: int,
              companies: tuple[str, ...]):
    """Build-or-load the FAISS index for these settings (cached across reruns)."""
    return build_index(embedding, chunk_size, chunk_overlap, list(companies))


@st.cache_resource(show_spinner=False)
def get_chunks(chunk_size: int, chunk_overlap: int, companies: tuple[str, ...]):
    """Chunked docs for BM25 (no embedding cost) — cached across reruns."""
    return chunk_corpus(list(companies), chunk_size, chunk_overlap)


def apply_theme(mode: str) -> None:
    """Inject a light or dark palette over Streamlit's chrome (in-app toggle)."""
    if mode == "Dark":
        p = dict(bg="#1b1c20", elev="#26272c", side="#202127",
                 text="#e8e8ea", muted="#9a9aa2", border="#33343a")
    else:
        p = dict(bg="#ffffff", elev="#f6f6f8", side="#f3f3f6",
                 text="#1b1c20", muted="#6b6b73", border="#e6e6ec")
    st.markdown(
        f"""
        <style>
          .stApp {{ background-color: {p['bg']}; }}
          header[data-testid="stHeader"] {{ background: {p['bg']}; }}
          section[data-testid="stSidebar"] {{ background-color: {p['side']}; }}
          .stApp, .stApp p, .stApp span, .stApp li, .stApp label, .stApp h1,
          .stApp h2, .stApp h3, .stApp h4,
          [data-testid="stMarkdownContainer"] {{ color: {p['text']}; }}
          [data-testid="stChatMessage"] {{ background-color: {p['elev']};
              border-radius: 12px; }}
          [data-testid="stExpander"] {{ background-color: {p['elev']};
              border: 1px solid {p['border']}; border-radius: 10px; }}
          [data-testid="stCaptionContainer"] p {{ color: {p['muted']} !important; }}
          .stChatInput textarea, [data-baseweb="input"] input,
          [data-baseweb="textarea"] textarea {{
              background-color: {p['elev']}; color: {p['text']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- sidebar
def sidebar() -> dict:
    s = st.sidebar
    s.title("📊 10-K RAG Studio")
    s.caption("Chat with Alphabet · Amazon · Microsoft · Apple · Tesla 10-Ks")

    providers = available_providers()
    embeddings = available_embeddings()

    if not providers or not embeddings:
        s.error(
            "No API keys found. Copy `.env.example` → `.env` and add at least "
            "one provider key (Google Gemini has a free tier), then restart."
        )
    else:
        s.caption("🟢 Connected: " + ", ".join(providers))

    theme = s.radio("🎨 Theme", ["Light", "Dark"], horizontal=True,
                    help="Switch the app between light and dark.")

    # ---- Model -------------------------------------------------------------
    s.subheader("🤖 Model")
    provider = s.selectbox(
        "Provider", providers or list(PROVIDERS),
        index=(providers or list(PROVIDERS)).index(DEFAULTS["provider"])
        if DEFAULTS["provider"] in (providers or list(PROVIDERS)) else 0,
        help="Only providers with a configured API key are listed.",
    )
    model = s.selectbox("Model", PROVIDERS[provider]["models"])
    s.caption(f"ℹ️ {PROVIDERS[provider]['note']}")

    # ---- Generation parameters --------------------------------------------
    s.subheader("🎛️ Generation")
    temperature = s.slider("Temperature", 0.0, 1.5, DEFAULTS["temperature"], 0.05,
                           help="Lower = more factual/deterministic. For financial Q&A, keep low.")
    top_p = s.slider("Top-p", 0.1, 1.0, DEFAULTS["top_p"], 0.05)
    max_tokens = s.slider("Max output tokens", 256, 4096, DEFAULTS["max_tokens"], 128)

    # ---- Retrieval ---------------------------------------------------------
    s.subheader("🔎 Retrieval")
    embedding = s.selectbox(
        "Embedding model", embeddings or list(EMBEDDINGS),
        index=(embeddings or list(EMBEDDINGS)).index(DEFAULTS["embedding"])
        if DEFAULTS["embedding"] in (embeddings or list(EMBEDDINGS)) else 0,
        help="Changing this rebuilds the vector index (different vector space).",
    )
    search_type = s.radio("Search type", ["mmr", "similarity"], horizontal=True,
                          help="MMR diversifies retrieved chunks; similarity is pure nearest-neighbour.")
    top_k = s.slider("Top-k (chunks sent to LLM)", 1, 12, DEFAULTS["top_k"])
    fetch_k = DEFAULTS["fetch_k"]
    mmr_lambda = DEFAULTS["mmr_lambda"]
    if search_type == "mmr":
        fetch_k = s.slider("Fetch-k (candidate pool)", top_k, 40, DEFAULTS["fetch_k"])
        mmr_lambda = s.slider("MMR λ (0 = diverse, 1 = relevant)", 0.0, 1.0,
                              DEFAULTS["mmr_lambda"], 0.1)
    hybrid = s.toggle("Hybrid retrieval (BM25 + vector)", value=DEFAULTS["hybrid"],
                      help="Add lexical BM25 to semantic search — recommended for "
                           "exact financial line items like 'deferred revenue'.")
    dense_weight = DEFAULTS["dense_weight"]
    if hybrid:
        dense_weight = s.slider("Dense weight (vs BM25)", 0.0, 1.0,
                                DEFAULTS["dense_weight"], 0.1)
    expand = s.toggle("Query expansion (multi-query)", value=DEFAULTS["expand"],
                      help="Rewrite the question into financial-statement terms "
                           "before retrieving (e.g. 'working capital' → 'current "
                           "assets/liabilities'). More accurate, slightly slower.")

    # ---- Chunking ----------------------------------------------------------
    s.subheader("✂️ Chunking")
    chunk_size = s.slider("Chunk size", 300, 2000, DEFAULTS["chunk_size"], 100,
                          help="Larger = more context per chunk, fewer chunks. Rebuilds the index.")
    chunk_overlap = s.slider("Chunk overlap", 0, 400, DEFAULTS["chunk_overlap"], 50)

    # ---- Corpus ------------------------------------------------------------
    s.subheader("🏢 Companies")
    companies = s.multiselect("In scope", list(COMPANIES), default=CORE_COMPANIES,
                              help="Restrict retrieval to these filings.")
    if not companies:
        companies = CORE_COMPANIES

    # ---- Persona -----------------------------------------------------------
    s.subheader("🧠 System prompt")
    preset = s.selectbox("Persona preset", list(PERSONAS),
                         index=list(PERSONAS).index(DEFAULT_PERSONA))
    persona = s.text_area("Edit persona", PERSONAS[preset], height=120)

    # ---- Compare mode ------------------------------------------------------
    s.subheader("⚖️ Compare mode")
    compare = s.toggle("Answer with a second model side-by-side", value=False,
                       help="Same retrieved context, two LLMs — great for comparing settings.")
    provider_b = model_b = None
    if compare:
        provider_b = s.selectbox("Provider B", providers or list(PROVIDERS), key="pb")
        model_b = s.selectbox("Model B", PROVIDERS[provider_b]["models"], key="mb")

    # ---- Actions -----------------------------------------------------------
    s.divider()
    col1, col2 = s.columns(2)
    rebuild = col1.button("🔧 Rebuild index", use_container_width=True)
    if col2.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return dict(
        provider=provider, model=model, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, embedding=embedding, search_type=search_type,
        top_k=top_k, fetch_k=fetch_k, mmr_lambda=mmr_lambda,
        hybrid=hybrid, dense_weight=dense_weight, expand=expand,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, companies=companies,
        persona=persona, compare=compare, provider_b=provider_b, model_b=model_b,
        rebuild=rebuild, theme=theme,
    )


# --------------------------------------------------------------------- helpers
def render_sources(docs):
    with st.expander(f"📄 Retrieved sources ({len(docs)})"):
        for i, d in enumerate(docs, 1):
            company = d.metadata.get("company", "?")
            page = d.metadata.get("page_label", d.metadata.get("page", "?"))
            st.markdown(f"**[{i}] {company} 10-K — p.{page}**")
            snippet = d.page_content.strip().replace("\n", " ")
            st.caption(snippet[:400] + ("…" if len(snippet) > 400 else ""))


def make_engine(cfg, provider, model, retriever):
    llm = build_llm(provider, model, cfg["temperature"], cfg["top_p"],
                    cfg["max_tokens"])
    return RagEngine(llm, retriever, cfg["persona"])


# --------------------------------------------------------------------- main
def main():
    cfg = sidebar()
    apply_theme(cfg["theme"])

    st.session_state.setdefault("messages", [])

    # Build / load index.
    if cfg["rebuild"]:
        get_store.clear()
    try:
        with st.spinner("Embedding 10-K filings & building the vector index… "
                        "(first build can take a minute; it is cached afterwards)"):
            store = get_store(cfg["embedding"], cfg["chunk_size"],
                              cfg["chunk_overlap"], tuple(cfg["companies"]))
    except MissingKeyError as e:
        st.info("👋 Welcome to **10-K RAG Studio**. To start, add an API key.")
        st.warning(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to build the index: {e}")
        st.stop()

    # Chunks for BM25 (cached). The retriever is built per question so it can be
    # scoped to the company the question names (avoids cross-company mixups).
    chunks = get_chunks(cfg["chunk_size"], cfg["chunk_overlap"],
                        tuple(cfg["companies"]))

    # Header strip with the active configuration.
    st.title("📊 10-K RAG Studio")
    chips = (f"**{cfg['provider']} · {cfg['model']}**  |  emb: `{cfg['embedding']}`  |  "
             f"temp `{cfg['temperature']}`  |  top-k `{cfg['top_k']}`  |  "
             f"chunk `{cfg['chunk_size']}/{cfg['chunk_overlap']}`  |  "
             f"{len(cfg['companies'])} filings")
    st.caption(chips)

    # Replay history.
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("docs"):
                render_sources(m["docs"])

    # Quick-start chips on an empty conversation.
    pending = None
    if not st.session_state.messages:
        st.markdown("##### 💡 Try one of these")
        cols = st.columns(2)
        for i, q in enumerate(STARTERS):
            if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
                pending = q

    # New turn.
    question = st.chat_input("Ask about the companies' 10-K filings…") or pending
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages[:-1]

    scope = detect_companies(question, cfg["companies"])
    retriever = build_query_retriever(
        store, chunks, scope, search_type=cfg["search_type"], top_k=cfg["top_k"],
        fetch_k=cfg["fetch_k"], mmr_lambda=cfg["mmr_lambda"],
        hybrid=cfg["hybrid"], dense_weight=cfg["dense_weight"])
    if cfg["expand"]:
        exp_llm = build_llm(cfg["provider"], cfg["model"], 0.3, 1.0, 200)
        retriever = MultiQueryRetriever(retriever, exp_llm, n=3, cap=2 * cfg["top_k"])

    with st.chat_message("assistant"):
        if set(scope) != set(cfg["companies"]):
            st.caption(f"🔎 scoped to: {', '.join(scope)}")
        with st.spinner("Retrieving relevant 10-K passages…"):
            try:
                docs = retriever.invoke(question)
            except MissingKeyError as e:
                st.error(str(e)); st.stop()
        render_sources(docs)

        if not cfg["compare"]:
            engine = make_engine(cfg, cfg["provider"], cfg["model"], retriever)
            t0 = time.time()
            try:
                answer = st.write_stream(engine.stream(question, history, docs))
            except MissingKeyError as e:
                st.error(str(e)); st.stop()
            st.caption(f"⏱️ {time.time() - t0:.1f}s · {cfg['provider']} · {cfg['model']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "docs": docs})
        else:
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**A · {cfg['provider']} · {cfg['model']}**")
                engA = make_engine(cfg, cfg["provider"], cfg["model"], retriever)
                t0 = time.time()
                ansA = st.write_stream(engA.stream(question, history, docs))
                st.caption(f"⏱️ {time.time() - t0:.1f}s")
            with colB:
                st.markdown(f"**B · {cfg['provider_b']} · {cfg['model_b']}**")
                engB = make_engine(cfg, cfg["provider_b"], cfg["model_b"], retriever)
                t0 = time.time()
                ansB = st.write_stream(engB.stream(question, history, docs))
                st.caption(f"⏱️ {time.time() - t0:.1f}s")
            merged = (f"**A · {cfg['model']}**\n\n{ansA}\n\n---\n\n"
                      f"**B · {cfg['model_b']}**\n\n{ansB}")
            st.session_state.messages.append(
                {"role": "assistant", "content": merged, "docs": docs})


if __name__ == "__main__":
    main()
