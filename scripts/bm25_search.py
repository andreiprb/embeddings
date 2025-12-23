"""
Query a BM25 index and print top-k results. Requires an index (bm25_build.py).

Input:
  indexes/bm25/{}.pkl (ro / en)

Outputs:
  reports/bm25/search/{}.json (ro_ro, ro_en, en_en, en_ro)

Usage:
  Monolingual:
    python scripts/bm25_search.py --file queries/monolingual/en_en.jsonl
    python scripts/bm25_search.py --file queries/monolingual/ro_ro.jsonl
  Cross lingual:
    python scripts/bm25_search.py --file queries/crosslingual/ro_en_translated.jsonl
    python scripts/bm25_search.py --file queries/crosslingual/en_ro_translated.jsonl
  Metaphor lists:
    python scripts/bm25_search.py --file _misc/metaphors/METAPHOR_LIST_RO.md
    python scripts/bm25_search.py --file _misc/metaphors/METAPHOR_LIST_EN.md
"""

import argparse
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
import yaml

try:
    import orjson
    def dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2)
except Exception:
    import json as _json
    def dumps(obj): return _json.dumps(obj, indent=2).encode("utf-8")


ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(config_name: str) -> dict:
    """
    Loads a YAML config file from the configs/ folder.

    :param config_name: File name in configs/ folder (without YAML).

    :return: The YAML config as a dict.
    """
    config_path = ROOT / "configs" / config_name

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    

def normalize(text: str) -> str:
    """
    Text normalization using regex.

    :param text: The text to be normalized.

    :return: The normalized text.
    """
    if text is None: return ""
    s = " ".join(text.split())
    s = re.sub(
        pattern=r"[^\w\-\u00C0-\u017F']+",
        repl=" ",
        string=s,
        flags=re.UNICODE
    )

    return s.lower().strip()


def tokenize(text: str) -> List[str]:
    """
    Simple word tokenizer

    :param text: The text to be tokenized.

    :return: The list of tokens.
    """
    return [t for t in text.split(" ") if t]


def get_top_k(payload: Dict[str, Any],
              query: str,
              retrieved: int,
              random: int,
              top: int) -> List:
    """
    Retrieve the top-k most relevant chunks for a given query using BM25.

    :param payload: Dictionary containing the BM25 model and metadata,
    :param query: The input query string to search for.
    :param retrieved: The number of top results to extract.
    :param random: The number of random results to return.
    :param top: The number of top results to return.

    :return: A list of tuples (chunk_id, score), sorted by score descending.
    """
    bm25 = payload["bm25"]
    ids = payload["chunk_ids"]

    tokens = tokenize(normalize(query))
    scores = bm25.get_scores(tokens)

    idx = np.argsort(scores)[::-1][:retrieved]

    top_idx = idx[:top]
    remaining_idx = idx[top:]

    if random > 0 and len(remaining_idx) > 0:
        random_count = min(random, len(remaining_idx))
        random_idx = np.random.choice(
            remaining_idx,
            random_count,
            replace=False
        )
    else:
        random_idx = np.array([], dtype=int)

    final_idx = np.concatenate([top_idx, random_idx])
    final_idx = final_idx[np.argsort(scores[final_idx])[::-1]]

    return [(ids[i], float(scores[i])) for i in final_idx]


def parse_md_metaphors(md_path: Path) -> List[str]:
    """
    Parses a .md file containing a numbered list of metaphors.

    The expected format:
      # Title (ignored)
      1. Metaphor 1
      2. Metaphor 2
      ...

    :param md_path: Full path to the .md file containing the metaphors.

    :return: A list of metaphor strings (one per line), stripped of numbering
    and titles.
    """
    items: List[str] = []
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            m = re.match(r"^\s*\d+\.\s*(.+)$", line)
            if m:
                items.append(m.group(1).strip())
    return items



def main():
    np.random.seed(42)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        type=str,
        default=None,
        help="path to a file with one query per line"
    )
    args = ap.parse_args()

    global_config = load_yaml_config("global.yaml")
    paths_config = load_yaml_config("paths.yaml")

    bm25_group = global_config.get("bm25")
    retrieved_k = bm25_group.get("retrieved_k")
    random_k = bm25_group.get("random_k")
    top_k = bm25_group.get("top_k")

    category = Path(args.file).stem.split("/")[-1]

    in_path = Path(args.file)
    suffix = in_path.suffix.lower()

    name = category.lower()

    # Determine target language from filename
    # e.g., en_ro.jsonl -> target is ro
    # e.g., en_en.jsonl -> target is en
    # e.g., en_ro_translated.jsonl -> target is ro
    if name.startswith("en_ro"):
        lang = "ro"
    elif name.startswith("ro_en"):
        lang = "en"
    elif name.startswith("ro_ro"):
        lang = "ro"
    elif name.startswith("en_en"):
        lang = "en"
    elif "ro" in name:
        lang = "ro"
    elif "en" in name:
        lang = "en"
    else:
        lang = "ro"

    if lang not in ("ro", "en"):
        lang = "ro"

    queries: List[str] = []
    if suffix == ".md":
        queries = parse_md_metaphors(in_path)
        if not queries:
            raise SystemExit("Fișierul .md nu conține metafore numerotate.")
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            queries.extend([line.strip() for line in f if line.strip()])

    if not queries:
        raise SystemExit("Provide --file with at least one query.")

    bm25_group = paths_config.get("bm25")
    if suffix == ".md":
        out_rel = f"reports/bm25/search/{lang}_metaphors.jsonl"
    else:
        bm25_search_str = bm25_group.get("search_report").format(category)
        out_rel = bm25_search_str

    idx_path = Path(ROOT, bm25_group.get("pkl").format(lang))
    if not idx_path.exists():
        raise SystemExit(f"BM25 index not found: {idx_path}.")

    with open(idx_path, "rb") as f:
        payload = pickle.load(f)

    outs = []
    for q in queries:
        results = get_top_k(
            payload=payload,
            query=q,
            retrieved=retrieved_k,
            random=random_k,
            top=top_k
        )
        out = {"query": q,
               "top_k": [{"chunk_id": cid, "score": score} for cid, score in
                         results]}
        outs.append(out)

    out_search_path = Path(ROOT, out_rel)
    out_search_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".md":
        with open(out_search_path, "wb") as f:
            for obj in outs:
                f.write(dumps(obj) + b"\n")
    else:
        with open(out_search_path, "wb") as f:
            f.write(dumps(outs))

if __name__ == "__main__":
    # main()
    print("[FROZEN] No need to run again")
