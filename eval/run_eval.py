"""
Batch-evaluate the chatbot against the test-question set (eval/questions.yaml).

For each question it runs the RAG pipeline with a chosen configuration, then
uses an LLM judge to grade the answer against the ground truth
(CORRECT / PARTIAL / WRONG). Results are printed as a table and written to
eval/results_<tag>.csv — this is the evidence for the Stage-2 comparison.

Examples
--------
# Default config (Gemini flash + Gemini embeddings)
python eval/run_eval.py

# Compare a "powerful" vs a "weaker" setting
python eval/run_eval.py --provider OpenAI --model gpt-4o --embedding "OpenAI text-embedding-3-small" --tag openai
python eval/run_eval.py --provider "Hugging Face (open-source)" --model meta-llama/Llama-3.1-8B-Instruct --tag llama8b
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragstudio.chains import (MultiQueryRetriever, RagEngine,  # noqa: E402
                              build_query_retriever, detect_companies)
from ragstudio.config import CORE_COMPANIES, DEFAULTS           # noqa: E402
from ragstudio.indexer import build_index, chunk_corpus         # noqa: E402
from ragstudio.providers import build_llm                       # noqa: E402

JUDGE_PROMPT = (
    "You are grading a RAG chatbot's answer about company 10-K filings.\n"
    "Question: {q}\n\nExpected (ground truth): {gt}\n\nChatbot answer: {ans}\n\n"
    "Grade ONLY whether the chatbot answer is factually consistent with the "
    "ground truth. Reply on the first line with exactly one word: CORRECT, "
    "PARTIAL, or WRONG. On the second line give a one-sentence reason. "
    "For questions where the ground truth is that the bot should refuse / say it "
    "lacks the information, grade CORRECT only if the chatbot refused rather than "
    "inventing an answer."
)


def grade(judge, question, ground_truth, answer) -> tuple[str, str]:
    msg = JUDGE_PROMPT.format(q=question, gt=ground_truth, ans=answer)
    out = judge.invoke(msg).content.strip().splitlines()
    verdict = (out[0].strip().upper() if out else "WRONG")
    for tag in ("CORRECT", "PARTIAL", "WRONG"):
        if tag in verdict:
            verdict = tag
            break
    else:
        verdict = "WRONG"
    reason = out[1].strip() if len(out) > 1 else ""
    return verdict, reason


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.yaml"))
    p.add_argument("--provider", default=DEFAULTS["provider"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--embedding", default=DEFAULTS["embedding"])
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    p.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    p.add_argument("--judge-provider", default="Google Gemini")
    p.add_argument("--judge-model", default="gemini-2.0-flash")
    p.add_argument("--no-hybrid", action="store_true",
                   help="Use vector-only retrieval instead of BM25+vector hybrid.")
    p.add_argument("--no-expand", action="store_true",
                   help="Disable LLM multi-query expansion.")
    p.add_argument("--tag", default="default")
    args = p.parse_args()

    data = yaml.safe_load(Path(args.questions).read_text())
    questions = data["questions"] if isinstance(data, dict) else data

    companies = sorted({c for q in questions for c in CORE_COMPANIES})
    store = build_index(args.embedding, args.chunk_size, args.chunk_overlap,
                        CORE_COMPANIES)
    all_chunks = chunk_corpus(CORE_COMPANIES, args.chunk_size, args.chunk_overlap)
    llm = build_llm(args.provider, args.model, args.temperature,
                    DEFAULTS["top_p"], DEFAULTS["max_tokens"])
    judge = build_llm(args.judge_provider, args.judge_model, 0.0, 1.0, 512)

    def retriever_for(question):
        companies = detect_companies(question, CORE_COMPANIES)
        r = build_query_retriever(
            store, all_chunks, companies, search_type=DEFAULTS["search_type"],
            top_k=args.top_k, fetch_k=DEFAULTS["fetch_k"],
            mmr_lambda=DEFAULTS["mmr_lambda"], hybrid=not args.no_hybrid,
            dense_weight=DEFAULTS["dense_weight"])
        if not args.no_expand:
            r = MultiQueryRetriever(r, llm, n=3, cap=2 * args.top_k)
        return r

    rows, tally = [], {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0}
    print(f"\nConfig: {args.provider}/{args.model} | emb={args.embedding} | "
          f"k={args.top_k} | chunk={args.chunk_size}/{args.chunk_overlap}\n")
    for q in questions:
        try:
            engine = RagEngine(llm, retriever_for(q["question"]), "")
            answer, _ = engine.answer(q["question"], [])
            verdict, reason = grade(judge, q["question"], q["ground_truth"], answer)
        except Exception as e:  # noqa: BLE001 — one slow/failed call shouldn't kill the run
            answer, verdict, reason = f"(error: {e})", "WRONG", "request error"
        tally[verdict] = tally.get(verdict, 0) + 1
        rows.append({"id": q.get("id", ""), "category": q.get("category", ""),
                     "question": q["question"], "ground_truth": q["ground_truth"],
                     "answer": answer, "verdict": verdict, "reason": reason})
        print(f"  [{verdict:7}] {q.get('id','')}: {q['question'][:70]}")

    n = len(rows) or 1
    score = (tally["CORRECT"] + 0.5 * tally["PARTIAL"]) / n
    print(f"\nScore: {score:.0%}  "
          f"(CORRECT={tally['CORRECT']} PARTIAL={tally['PARTIAL']} WRONG={tally['WRONG']})")

    out = ROOT / "eval" / f"results_{args.tag}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
