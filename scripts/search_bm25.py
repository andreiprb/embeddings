"""
Query a BM25 index and print top-k results.
Requires an index built by build_bm25.py.

Usage:
  python scripts/search_bm25.py --lang ro --k 5 --query "ce înseamnă 'vânturi potrivnice' în context de afaceri?"
  python scripts/search_bm25.py --lang en --k 10 --query "explain short squeeze in plain English"
  python scripts/search_bm25.py --lang ro --file queries.txt  # one query per line

Optional:
  --chunks data/chunks.parquet
"""
from __future__ import annotations

import argparse, json, pickle, re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    raise SystemExit("Missing dependency: install with `pip install rank-bm25`") from e

ROOT = Path(__file__).resolve().parents[1]

def normalize(text: str, lowercase: bool=True) -> str:
    s = " ".join((text or "").split())
    s = re.sub(r"[^\w\-\u00C0-\u017F']+", " ", s, flags=re.UNICODE)
    return s.lower().strip() if lowercase else s.strip()

def tokenize(text: str) -> List[str]:
    return [t for t in text.split(" ") if t]

def default_index_path(lang: str) -> Path:
    return ROOT / "indexes" / f"bm25_{lang}.pkl"

def load_index(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not {"bm25","chunk_ids","config"}.issubset(payload.keys()):
        raise SystemExit("Index payload missing required keys.")
    return payload

def topk(payload: Dict[str, Any], query: str, k: int=10):
    bm25 = payload["bm25"]
    ids = payload["chunk_ids"]
    toks = tokenize(normalize(query))
    scores = bm25.get_scores(toks)
    idx = np.argsort(scores)[::-1][:k]
    return [(ids[i], float(scores[i])) for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, required=True, choices=["ro","en"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--file", type=str, default=None, help="path to a file with one query per line")
    ap.add_argument("--index", type=str, default=None, help="explicit path to .pkl index")
    ap.add_argument("--chunks", type=str, default=None, help="optional chunks parquet/csv to attach text snippets")
    args = ap.parse_args()

    idx_path = Path(args.index) if args.index else default_index_path(args.lang)
    if not idx_path.exists():
        raise SystemExit(f"BM25 index not found: {idx_path}. Build with scripts/build_bm25.py first.")

    payload = load_index(idx_path)

    chunks_df = None
    if args.chunks:
        cp = Path(args.chunks)
        if cp.exists():
            if cp.suffix == ".parquet":
                chunks_df = pd.read_parquet(cp)
            elif cp.suffix == ".csv":
                chunks_df = pd.read_csv(cp)
        else:
            print(f"[WARN] chunks not found: {cp}")

    queries: List[str] = []
    if args.query:
        queries.append(args.query)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            queries.extend([line.strip() for line in f if line.strip()])

    if not queries:
        raise SystemExit("Provide --query or --file with at least one query.")

    for q in queries:
        results = topk(payload, q, k=args.k)
        out = {"query": q, "topk": []}
        for cid, score in results:
            item = {"chunk_id": cid, "score": score}
            if chunks_df is not None:
                row = chunks_df[chunks_df["chunk_id"]==cid]
                if not row.empty:
                    txt = str(row.iloc[0]["text"])
                    item["text"] = txt[:300] + ("..." if len(txt)>300 else "")
            out["topk"].append(item)
        print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
