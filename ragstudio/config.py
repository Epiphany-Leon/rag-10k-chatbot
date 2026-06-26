"""
Central configuration for 10-K RAG Studio.

Everything that is "swappable" in the UI lives here as data:
  - PROVIDERS  : chat-LLM providers and their model lists
  - EMBEDDINGS : embedding models (each defines its own vector space)
  - COMPANIES  : the 10-K documents available to the chatbot
  - DEFAULTS   : starting RAG parameters

Adding a new model = add one line here. No other file needs to change.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time. Never commit .env (see .gitignore).
load_dotenv()

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "10k"
CACHE_DIR = ROOT_DIR / ".cache"          # persisted FAISS indexes (git-ignored)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- chat LLM providers
# kind:
#   "gemini"            -> langchain_google_genai.ChatGoogleGenerativeAI
#   "openai"            -> langchain_openai.ChatOpenAI (api.openai.com)
#   "openai_compatible" -> langchain_openai.ChatOpenAI with a custom base_url
PROVIDERS: dict[str, dict] = {
    "Google Gemini": {
        "kind": "gemini",
        "env": "GOOGLE_API_KEY",
        "models": [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
        ],
        "note": "Free tier. Great starting point.",
    },
    "OpenAI": {
        "kind": "openai",
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "o4-mini",
        ],
        "note": "Strong general accuracy.",
    },
    "DeepSeek": {
        "kind": "openai_compatible",
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "note": "Cheap, OpenAI-compatible.",
    },
    "Kimi (Moonshot)": {
        "kind": "openai_compatible",
        "env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "note": "Long context, OpenAI-compatible.",
    },
    "Zhipu GLM": {
        "kind": "openai_compatible",
        "env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4.6", "glm-5.2", "glm-4.5-flash", "glm-4.5-air", "glm-4-flash"],
        "note": "Open-source GLM via Zhipu BigModel; glm-4.5-flash / glm-4-flash are free.",
    },
    "Hugging Face (open-source)": {
        "kind": "openai_compatible",
        "env": "HUGGINGFACE_API_KEY",
        "base_url": "https://router.huggingface.co/v1",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "deepseek-ai/DeepSeek-V3-0324",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "note": "Open-source models via HF Inference Providers.",
    },
}

# Models that ignore sampling params (reasoning models) — keep temperature/top_p off.
NO_SAMPLING_MODELS = {"o4-mini", "o3", "o3-mini", "deepseek-reasoner"}

# ---------------------------------------------------------------- embedding models
# Each embedding model defines a distinct vector space, so changing it forces a
# rebuild of the FAISS index (handled automatically by the indexer's cache key).
EMBEDDINGS: dict[str, dict] = {
    "Gemini text-embedding-004": {
        "kind": "gemini", "env": "GOOGLE_API_KEY",
        "model": "models/text-embedding-004", "dim": 768,
    },
    "Gemini embedding-001": {
        "kind": "gemini", "env": "GOOGLE_API_KEY",
        "model": "models/embedding-001", "dim": 768,
    },
    "OpenAI text-embedding-3-small": {
        "kind": "openai", "env": "OPENAI_API_KEY",
        "model": "text-embedding-3-small", "dim": 1536,
    },
    "OpenAI text-embedding-3-large": {
        "kind": "openai", "env": "OPENAI_API_KEY",
        "model": "text-embedding-3-large", "dim": 3072,
    },
    "HF bge-small-en-v1.5": {
        "kind": "hf_api", "env": "HUGGINGFACE_API_KEY",
        "model": "BAAI/bge-small-en-v1.5", "dim": 384,
    },
    "HF all-MiniLM-L6-v2": {
        "kind": "hf_api", "env": "HUGGINGFACE_API_KEY",
        "model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384,
    },
    "Zhipu embedding-3": {
        "kind": "openai_compatible", "env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "embedding-3", "dim": 2048,
    },
}

# ---------------------------------------------------------------- corpus
# The 10-K filings shipped with the repo. `ticker` is stored on every chunk so
# the retriever can be scoped to a subset of companies.
COMPANIES: dict[str, dict] = {
    "Alphabet":  {"ticker": "GOOGL", "file": "Alphabet_10k_2025.pdf"},
    "Amazon":    {"ticker": "AMZN",  "file": "Amazon_10k_2025.pdf"},
    "Microsoft": {"ticker": "MSFT",  "file": "Microsoft_10K_2025.pdf"},
    "Apple":     {"ticker": "AAPL",  "file": "Apple_10k_2025.pdf"},
    "Tesla":     {"ticker": "TSLA",  "file": "Tesla_10k_2025.pdf"},
}

# The three companies the assignment asks us to compare (Apple/Tesla are bonus).
CORE_COMPANIES = ["Alphabet", "Amazon", "Microsoft"]

# ---------------------------------------------------------------- defaults
DEFAULTS = {
    "provider": "Google Gemini",
    "model": "gemini-2.0-flash",
    "embedding": "Gemini text-embedding-004",
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 1024,
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "top_k": 5,
    "fetch_k": 20,
    "search_type": "mmr",          # "mmr" or "similarity"
    "mmr_lambda": 0.5,
    "hybrid": True,                # combine BM25 + vector retrieval
    "dense_weight": 0.6,           # ensemble weight on the vector retriever
}


def has_key(env_name: str) -> bool:
    """True if the given API-key env var is set to a non-empty value."""
    return bool(os.getenv(env_name, "").strip())


def available_providers() -> list[str]:
    """Providers whose API key is present in the environment."""
    return [name for name, spec in PROVIDERS.items() if has_key(spec["env"])]


def available_embeddings() -> list[str]:
    """Embedding models whose API key is present in the environment."""
    return [name for name, spec in EMBEDDINGS.items() if has_key(spec["env"])]
