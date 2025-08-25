"""
Build BM25 indexes (RO / EN) from `data/chunks.parquet` (or CSV fallback), using rank_bm25.

Outputs (by default):
  indexes/bm25_ro.pkl
  indexes/bm25_en.pkl
  indexes/bm25_meta.json

Usage:
  pip install rank-bm25 orjson pandas pyarrow
  python scripts/build_bm25.py
  python scripts/build_bm25.py --lang ro
  python scripts/build_bm25.py --chunks data/chunks.parquet --out indexes/bm25_ro.pkl --lang ro

Notes:
- Tokenization: simple, diacritics-preserving whitespace split with light normalization.
- We pickle the BM25 object + chunk_id list + preprocessing config for reproducibility.
"""
from __future__ import annotations

import argparse, json, pickle, re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

try:
    import orjson
    def dumps(obj): return orjson.dumps(obj)
except Exception:
    import json as _json
    def dumps(obj): return _json.dumps(obj).encode("utf-8")

try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    raise SystemExit("Missing dependency: install with `pip install rank-bm25`") from e

ROOT = Path(__file__).resolve().parents[1]

def normalize(text: str, lowercase: bool=True, keep_diacritics: bool=True) -> str:
    # Light normalization: collapse whitespace, remove most punctuation except intra-word apostrophes/hyphens
    if text is None: return ""
    s = " ".join(text.split())
    # Replace common punctuation with space
    s = re.sub(r"[^\w\-\u00C0-\u017F']+", " ", s, flags=re.UNICODE)  # keep diacritics range
    if lowercase:
        s = s.lower()
    return s.strip()

def tokenize(text: str) -> List[str]:
    return [t for t in text.split(" ") if t]

def build_one(df: pd.DataFrame, lang: str, k1: float=1.5, b: float=0.75) -> Dict[str, Any]:
    sub = df[df["lang"]==lang][["chunk_id","text"]].copy()
    if sub.empty:
        raise SystemExit(f"No chunks found for lang={lang}. Run the chunker first and check 'lang' values.")
    texts = sub["text"].tolist()
    chunk_ids = sub["chunk_id"].tolist()
    tokens = [tokenize(normalize(t)) for t in texts]
    bm25 = BM25Okapi(tokens, k1=k1, b=b)
    payload = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "config": {"k1": k1, "b": b, "norm": "whitespace+punct_strip", "lowercase": True}
    }
    return payload

def default_chunks_path() -> Path:
    # Prefer Parquet
    p1 = ROOT / "data" / "chunks.parquet"
    p2 = ROOT / "data" / "chunks.csv"
    if p1.exists(): return p1
    if p2.exists(): return p2
    raise SystemExit("Missing chunks. Expected data/chunks.parquet (or .csv). Run scripts/chunk_corpus.py first.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default=None, help="Path to chunks parquet/csv")
    ap.add_argument("--out", type=str, default=None, help="Output .pkl path (when building a single lang)")
    ap.add_argument("--lang", type=str, default=None, choices=["ro","en","both"], help="Language to build (default: both)")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    chunks_path = Path(args.chunks) if args.chunks else default_chunks_path()
    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found: {chunks_path}")
    if chunks_path.suffix == ".parquet":
        df = pd.read_parquet(chunks_path)
    elif chunks_path.suffix == ".csv":
        df = pd.read_csv(chunks_path)
    else:
        raise SystemExit("Unsupported chunks format. Use .parquet or .csv.")

    if not {"chunk_id","lang","text"}.issubset(df.columns):
        missing = {"chunk_id","lang","text"} - set(df.columns)
        raise SystemExit(f"Chunks file missing columns: {missing}")

    out_dir = ROOT / "indexes"
    out_dir.mkdir(parents=True, exist_ok=True)

    which = args.lang or "both"
    built = {}

    if which in ("ro","both"):
        payload = build_one(df, "ro", k1=args.k1, b=args.b)
        out_ro = Path(args.out) if (args.out and which=="ro") else out_dir / "bm25_ro.pkl"
        with open(out_ro, "wb") as f:
            pickle.dump(payload, f)
        built["ro"] = str(out_ro.relative_to(ROOT))

    if which in ("en","both"):
        payload = build_one(df, "en", k1=args.k1, b=args.b)
        out_en = Path(args.out) if (args.out and which=="en") else out_dir / "bm25_en.pkl"
        with open(out_en, "wb") as f:
            pickle.dump(payload, f)
        built["en"] = str(out_en.relative_to(ROOT))

    meta = {
        "chunks": str(chunks_path.relative_to(ROOT)) if str(chunks_path).startswith(str(ROOT)) else str(chunks_path),
        "built": built,
        "params": {"k1": args.k1, "b": args.b},
    }
    with open(out_dir / "bm25_meta.json", "wb") as f:
        f.write(dumps(meta))

    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
