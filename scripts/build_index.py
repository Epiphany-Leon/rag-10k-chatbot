"""
Pre-build the FAISS index from the command line (nicer than waiting in the UI
on first run, and handy for the deploy step).

Example:
    python scripts/build_index.py --embedding "Gemini text-embedding-004"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragstudio.config import CORE_COMPANIES, DEFAULTS, COMPANIES  # noqa: E402
from ragstudio.indexer import build_index, count_chunks           # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", default=DEFAULTS["embedding"])
    p.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    p.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    p.add_argument("--all-companies", action="store_true",
                   help="Index all 5 filings (default: the 3 core companies).")
    args = p.parse_args()

    companies = list(COMPANIES) if args.all_companies else CORE_COMPANIES
    n = count_chunks(companies, args.chunk_size, args.chunk_overlap)
    print(f"Companies : {', '.join(companies)}")
    print(f"Chunks    : {n} (size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"Embedding : {args.embedding}")
    print("Building index… (this calls the embedding API for every chunk)")
    t0 = time.time()
    build_index(args.embedding, args.chunk_size, args.chunk_overlap, companies,
                force=True)
    print(f"Done in {time.time() - t0:.1f}s. Index cached under .cache/")


if __name__ == "__main__":
    main()
