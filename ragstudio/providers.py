"""
Factories that turn a (provider, model, params) selection into a LangChain
chat model or embedding object. All provider-specific quirks live here.
"""
from __future__ import annotations

import os

from .config import EMBEDDINGS, NO_SAMPLING_MODELS, PROVIDERS, has_key

# Fast, cheap models preferred for the query-rewriting step (it only reformulates
# the question, so it should never be slowed down by a heavy answer model).
_FAST_EXPANSION = [("Zhipu GLM", "glm-4-flash"), ("DeepSeek", "deepseek-chat"),
                   ("OpenAI", "gpt-4o-mini")]


def build_expansion_llm(default_provider: str, default_model: str):
    """A fast LLM for query expansion, independent of the (maybe slow) answer model."""
    for prov, model in _FAST_EXPANSION:
        spec = PROVIDERS.get(prov)
        if spec and has_key(spec["env"]):
            try:
                return build_llm(prov, model, 0.3, 1.0, 200)
            except Exception:  # noqa: BLE001
                continue
    return build_llm(default_provider, default_model, 0.3, 1.0, 200)


class MissingKeyError(RuntimeError):
    """Raised when the API key for a selected provider is not configured."""


# ------------------------------------------------------------------ chat LLMs
def build_llm(provider: str, model: str, temperature: float,
              top_p: float, max_tokens: int):
    """Return a LangChain chat model for the selected provider/model."""
    spec = PROVIDERS[provider]
    if not has_key(spec["env"]):
        raise MissingKeyError(
            f"{provider} needs {spec['env']} in your .env file."
        )

    # Reasoning models reject sampling params; omit them.
    sampling = {} if model in NO_SAMPLING_MODELS else {
        "temperature": temperature, "top_p": top_p,
    }

    if spec["kind"] == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.environ[spec["env"]],
            max_output_tokens=max_tokens,
            **sampling,
        )

    if spec["kind"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=os.environ[spec["env"]],
            max_tokens=max_tokens,
            timeout=180,
            **sampling,
        )

    # openai + openai_compatible both use ChatOpenAI; base_url switches the host.
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=os.environ[spec["env"]],
        base_url=spec.get("base_url"),
        max_tokens=max_tokens,
        timeout=180,
        **sampling,
    )


# ------------------------------------------------------------------ embeddings
def build_embeddings(name: str):
    """Return a LangChain embeddings object for the selected embedding model."""
    spec = EMBEDDINGS[name]

    if spec["kind"] == "hf_local":
        # Runs locally on CPU, no API key. Needs requirements-local.txt installed.
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=spec["model"],
            encode_kwargs={"normalize_embeddings": True},
        )

    if not has_key(spec["env"]):
        raise MissingKeyError(
            f"Embedding '{name}' needs {spec['env']} in your .env file."
        )

    if spec["kind"] == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=spec["model"], google_api_key=os.environ[spec["env"]],
        )

    if spec["kind"] in ("openai", "openai_compatible"):
        from langchain_openai import OpenAIEmbeddings
        compat = spec["kind"] == "openai_compatible"
        return OpenAIEmbeddings(
            model=spec["model"], api_key=os.environ[spec["env"]],
            base_url=spec.get("base_url"),
            # Non-OpenAI endpoints use a different tokenizer and cap batch size;
            # Zhipu allows max 64 inputs per embeddings request.
            check_embedding_ctx_length=not compat,
            chunk_size=64 if compat else 1000,
        )

    if spec["kind"] == "hf_api":
        # Hosted Hugging Face feature-extraction endpoint (no local torch).
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model=spec["model"], huggingfacehub_api_token=os.environ[spec["env"]],
        )

    raise ValueError(f"Unknown embedding kind: {spec['kind']}")
