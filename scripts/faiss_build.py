"""
Build FAISS indexes (RO/EN) per embedding model using configs.

Inputs:
  chunks.parquet

Outputs:
  faiss.index, faiss.meta (pickle with chunk_ids and metadata)

Usage:
  python scripts/faiss_build.py --model e5-multilingual
  python scripts/faiss_build.py
"""


import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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

    :param config_name: File name in configs/ folder (with .yaml).
    :return: Parsed YAML as dict (empty dict if file missing/empty).
    """
    path = ROOT / "configs" / config_name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def families_passage_template(prompts_cfg: dict, family: str) -> str:
    fam = (prompts_cfg or {}).get("families", {}).get(family, {})
    return fam.get("passage", "{text}")


def make_passages(texts: List[str], tmpl: str) -> List[str]:
    return [tmpl.replace(
        "{text}", t if isinstance(t, str) else ""
    ) for t in texts]


def embed_corpus(model_id: str,
                 texts: List[str],
                 batch_size: int | None = None,
                 device: str | None = None) -> np.ndarray:
    model = SentenceTransformer(
        model_name_or_path=model_id,
        device=device if device else None,
        trust_remote_code=True)
    kwargs = {
        "batch_size": int(batch_size) if batch_size else 256,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    embs = model.encode(texts, **kwargs)
    embs = np.ascontiguousarray(embs.astype("float32", copy=False))

    mask = np.isfinite(embs).all(axis=1)
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"  [warn] dropping {dropped} embeddings with non-finite values")
        embs = embs[mask]
    return embs


def build_hnsw(embs: np.ndarray,
               m: int,
               efc: int,
               efs: int,
               threads: int) -> faiss.Index:
    try:
        faiss.omp_set_num_threads(int(threads))
    except Exception:
        pass

    dim = int(embs.shape[1])
    index = faiss.IndexHNSWFlat(dim, int(m))
    index.hnsw.efConstruction = int(efc)
    index.hnsw.efSearch = int(efs)
    index.add(embs)

    return index


def save_index_payload(index: faiss.Index,
                       chunk_ids: List[str],
                       path_index: Path,
                       path_meta: Path,
                       meta: Dict) -> None:
    ensure_dir(path_index)
    faiss.write_index(index, str(path_index))
    payload = {
        "dim": int(index.d),
        "count": int(len(chunk_ids)),
        "chunk_ids": list(map(str, chunk_ids)),
        "meta": meta,
    }
    ensure_dir(path_meta)
    with open(path_meta, "wb") as f:
        pickle.dump(payload, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model name from models.yaml (repeatable)."
    )
    args = ap.parse_args()

    global_cfg = load_yaml_config("global.yaml")
    paths_cfg  = load_yaml_config("paths.yaml")
    models_cfg = load_yaml_config("models.yaml")

    try:
        prompts_cfg = load_yaml_config("prompts.yaml")
    except Exception:
        prompts_cfg = {}

    langs: List[str] = list(global_cfg.get("languages"))

    faiss_group = global_cfg.get("faiss")
    m           = int(faiss_group.get("m"))
    efc         = int(faiss_group.get("ef_construction"))
    efs         = int(faiss_group.get("ef_search"))
    batch_size  = faiss_group.get("batch_size")
    device      = faiss_group.get("device")

    chunks_parquet_str = paths_cfg.get("chunks").get("parquet")
    faiss_group = paths_cfg.get("faiss")

    faiss_faiss_tpl = faiss_group.get("faiss")
    faiss_pkl_tpl  = faiss_group.get("pkl")

    chunks_path = Path(ROOT, chunks_parquet_str)

    if not chunks_path.exists():
        raise SystemExit(f"Chunks parquet not found: {chunks_path}")

    df = pd.read_parquet(chunks_path)
    for col in ("chunk_id", "lang", "text"):
        if col not in df.columns:
            raise SystemExit(f"Expected column '{col}' in {chunks_path}, found: {list(df.columns)}")

    all_specs: List[dict] = list((models_cfg.get("dual_encoders") or []))
    if not all_specs:
        raise SystemExit("models.yaml has no 'dual_encoders'.")

    if args.model:
        wanted = set(args.model)
        specs = [s for s in all_specs if s.get("name") in wanted]
        missing = wanted - {s.get("name") for s in specs}
        if missing:
            raise SystemExit(f"Unknown --model: {', '.join(sorted(missing))}")
    else:
        specs = all_specs

    for spec in specs:
        name   = spec.get("name")
        hf_id  = spec.get("hf_id")
        family = spec.get("family", "labse")
        if not name or not hf_id:
            print(f"[warn] skip bad spec: {spec}")
            continue

        print(f"\n=== {name} | {hf_id} | family={family} ===")
        passage_tmpl = families_passage_template(prompts_cfg, family)

        for lang in langs:
            sub = df[df["lang"] == lang].reset_index(drop=True)
            if sub.empty:
                print(f"  skip {lang}: 0 chunks")
                continue

            chunk_ids: List[str] = sub["chunk_id"].astype(str).tolist()
            texts = sub["text"].astype(str).tolist()
            passages = make_passages(texts, passage_tmpl)

            embs = embed_corpus(hf_id, passages, batch_size=batch_size, device=device)
            if embs.shape[0] != len(chunk_ids):
                n = min(len(chunk_ids), int(embs.shape[0]))
                if n < len(chunk_ids):
                    print(f"  [warn] trimming chunk_ids {len(chunk_ids)} → {n}")
                chunk_ids = chunk_ids[:n]
                embs = embs[:n]

            print(f"  {lang}: {embs.shape[0]} chunks → dim={embs.shape[1]}")

            index = build_hnsw(embs, m=m, efc=efc, efs=efs, threads=int(os.getenv("OMP_NUM_THREADS", "1")))
            print(f"  Index built (ntotal={index.ntotal})")

            idx_path  = Path(ROOT, faiss_faiss_tpl.format(model=name, lang=lang))
            meta_path = Path(ROOT, faiss_pkl_tpl.format(model=name, lang=lang))
            meta = {
                "model_name": name,
                "hf_id": hf_id,
                "family": family,
                "normalized": True,
                "similarity": (global_cfg.get("retrieval") or {}).get("similarity", "cosine"),
                "faiss": {"index": type(index).__name__, "m": int(m), "ef_construction": int(efc),
                           "ef_search": int(efs), "threads": int(os.getenv("OMP_NUM_THREADS", "1"))},
                "source_chunks": str(chunks_path.relative_to(ROOT)) if chunks_path.is_relative_to(ROOT) else str(chunks_path),
                "languages": langs,
            }
            save_index_payload(index, chunk_ids, idx_path, meta_path, meta)
            print(f"  saved → {idx_path} and {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    # main()
    print("[FROZEN] No need to run again")
