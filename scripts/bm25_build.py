"""
Build BM25 indexes (ro / en) from data/chunks/chunks.parquet, using rank_bm25.

Input:
  data/chunks/chunks.parquet

Outputs:
  indexes/bm25/{}.pkl (ro / en)
  reports/bm25/report.json
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
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


def build_one(df: pd.DataFrame,
              lang: str,
              k1: float,
              b: float) -> Dict[str, Any]:
    """
    Build a BM25 index for a specific language from the given chunks DataFrame.

    :param df: DataFrame containing text chunks with columns:
        - 'chunk_id': unique identifier for each chunk
        - 'lang': language code (e.g., 'ro', 'en')
        - 'text': the chunk text to index
    :param lang: Language to filter and index.
    :param k1: BM25 term frequency saturation parameter.
    :param b: BM25 length normalization parameter.

    :return: A dictionary containing:
        - 'bm25': the trained BM25Okapi object
        - 'chunk_ids': list of chunk IDs in the same order as the BM25 corpus
        - 'config': BM25 configuration (k1, b, normalization details)

    :raises SystemExit: If no chunks are found for the specified language.
    """
    sub = df[df["lang"]==lang][["chunk_id","text"]].copy()
    if sub.empty:
        raise SystemExit(f"No chunks found for lang={lang}. Run the chunker "
                         "first and check 'lang' values.")
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


def main():
    global_config = load_yaml_config("global.yaml")
    paths_config = load_yaml_config("paths.yaml")

    langs = global_config.get("languages")

    chunk_group = paths_config.get("chunks")
    chunks_parquet_str = chunk_group.get("parquet")

    bm25_group = paths_config.get("bm25")
    bm25_pkl_str = bm25_group.get("pkl")
    bm25_meta_str = bm25_group.get("report")

    bm25_cfg = global_config.get("bm25")
    bm25_k1 = float(bm25_cfg.get("k1", 1.5))
    bm25_b = float(bm25_cfg.get("b", 0.75))

    chunks_path = Path(ROOT, chunks_parquet_str)

    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found: {chunks_path}")

    if chunks_path.suffix == ".parquet":
        df = pd.read_parquet(chunks_path)
    else:
        raise SystemExit("Unsupported chunks format. Use .parquet.")

    required_cols = {"chunk_id", "lang", "text"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise SystemExit(f"Chunks file missing columns: {missing}")

    built = {}

    for lang in langs:
        out_lang_path = bm25_pkl_str.format(lang)
        out_lang = Path(ROOT, out_lang_path)

        out_lang.parent.mkdir(parents=True, exist_ok=True)

        payload = build_one(df=df, lang=lang, k1=bm25_k1, b=bm25_b)

        with open(out_lang, "wb") as f:
            pickle.dump(payload, f)

        built[lang] = str(out_lang.relative_to(ROOT))

    out_meta_path = Path(ROOT, bm25_meta_str)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "chunks": chunks_parquet_str,
        "built": built,
        "params": {"k1": bm25_k1, "b": bm25_b},
    }

    with open(out_meta_path, "wb") as f:
        f.write(dumps(meta))


if __name__ == "__main__":
    # main()
    print("[FROZEN] No need to run again")
