"""
Token-window chunker with overlap for RO/EN business corpora — DOCX ONLY.

Inputs:
- EITHER provide CLI args:
    --docx_ro /path/to/romanian.docx
    --docx_en /path/to/english.docx
- OR run with **no args** and let defaults come from `configs/global.yaml`:
    paths:
      raw_docs: data/raw_docs/corpus_{}.docx
      chunks_parquet: data/chunks.parquet

Reads defaults from `configs/global.yaml`:
    chunking.size_tokens (default 300)
    chunking.overlap_tokens (default 60)
    paths.raw_docs (optional format string with `{}`)
    paths.chunks_parquet (optional)

Writes:
    data/chunks.parquet (CSV fallback if Parquet is not available)
    data/chunk_stats.json

Usage examples:
  # using config defaults (no args)
  python scripts/chunk_corpus.py

  # explicit inputs
  python scripts/chunk_corpus.py --docx_ro ./data/raw_docs/corpus_ro.docx --docx_en ./data/raw_docs/corpus_en.docx

  # override size/overlap/output
  python scripts/chunk_corpus.py --size 300 --overlap 60 --out data/chunks.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Optional deps
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

import pandas as pd

# Optional DOCX readers
try:
    from docx import Document  # python-docx
except Exception:
    Document = None            # type: ignore

try:
    import docx2txt            # fallback
except Exception:
    docx2txt = None            # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTS = [ROOT / "data" / "chunks.parquet", ROOT / "data" / "chunks.csv"]
CONFIG_PATH = ROOT / "configs" / "global.yaml"
STATS_PATH = ROOT / "data" / "chunk_stats.json"


# ------------------------------
# Config & basic tokenization
# ------------------------------
def load_config_chunking():
    size_tokens = 300
    overlap_tokens = 60
    if yaml and CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ch = cfg.get("chunking", {}) or {}
        size_tokens = int(ch.get("size_tokens", size_tokens))
        overlap_tokens = int(ch.get("overlap_tokens", overlap_tokens))
    return size_tokens, overlap_tokens


def load_config_paths() -> Dict[str, Optional[Path]]:
    """
    Reads 'paths' section from global.yaml.
    - raw_docs: a format string with '{}' -> expands to ro/en paths
    - chunks_parquet: default output
    Returns dict with keys: docx_ro, docx_en, out_chunks (Path or None)
    """
    result = {"docx_ro": None, "docx_en": None, "out_chunks": None}
    if not (yaml and CONFIG_PATH.exists()):
        return result
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    p = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    raw_pat = p.get("raw_docs")
    if isinstance(raw_pat, str) and "{}" in raw_pat:
        ro = Path(raw_pat.format("ro"))
        en = Path(raw_pat.format("en"))
        # make relative paths project-rooted
        if not ro.is_absolute():
            ro = (ROOT / ro).resolve()
        if not en.is_absolute():
            en = (ROOT / en).resolve()
        result["docx_ro"] = ro
        result["docx_en"] = en
    outp = p.get("chunks_parquet")
    if isinstance(outp, str):
        out = Path(outp)
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        result["out_chunks"] = out
    return result


def whitespace_tokens(text: str) -> List[str]:
    # Simple, reproducible tokenization: normalize whitespace then split
    if not text:
        return []
    return " ".join(text.split()).split(" ")


def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)


def chunk_tokens(tokens: List[str], size: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) slices using sliding window.
    end_idx is exclusive.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    n = len(tokens)
    if n == 0:
        return []
    out = []
    stride = size - overlap
    start = 0
    while start < n:
        end = min(start + size, n)
        out.append((start, end))
        if end == n:
            break
        start += stride
    return out


# ------------------------------
# DOCX ingestion (RO / EN)
# ------------------------------
def _docx_text_python_docx(path: Path) -> str:
    # Extract paragraphs and table cell text using python-docx
    doc = Document(str(path))
    parts: List[str] = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)
    for tbl in getattr(doc, "tables", []):
        for row in tbl.rows:
            for cell in row.cells:
                t = cell.text.strip()
                if t:
                    parts.append(t)
    return "\n".join(parts)


def _docx_text_docx2txt(path: Path) -> str:
    return (docx2txt.process(str(path)) or "").strip()


def read_docx_text(path: Path) -> str:
    if path.suffix.lower() != ".docx":
        raise SystemExit(f"Unsupported file type: {path.name} (need .docx)")
    if Document is not None:
        try:
            return _docx_text_python_docx(path)
        except Exception:
            if docx2txt is not None:
                return _docx_text_docx2txt(path)
            raise
    else:
        if docx2txt is not None:
            return _docx_text_docx2txt(path)
        raise SystemExit(
            "DOCX reading requires either 'python-docx' or 'docx2txt'. "
            "Install one: pip install python-docx  (or)  pip install docx2txt"
        )


def build_docs_from_docx(docx_ro: Optional[Path], docx_en: Optional[Path]) -> pd.DataFrame:
    """
    Build an in-memory docs DataFrame from up to two DOCX files.
    Schema: doc_id, lang, source, title, url, published_at, text, checksum, tokens_est
    """
    import hashlib

    rows: List[Dict] = []
    if docx_ro:
        if not docx_ro.exists():
            raise SystemExit(f"DOCX not found: {docx_ro}")
        text_ro = read_docx_text(docx_ro).strip()
        title_ro = docx_ro.stem
        checksum_ro = hashlib.sha256(text_ro.encode("utf-8", errors="ignore")).hexdigest()
        doc_id_ro = hashlib.sha1(f"ro\n{title_ro}\n{text_ro[:2000]}".encode("utf-8", errors="ignore")).hexdigest()
        rows.append({
            "doc_id": doc_id_ro,
            "lang": "ro",
            "source": str(docx_ro),
            "title": title_ro,
            "url": "",
            "published_at": None,
            "text": text_ro,
            "checksum": checksum_ro,
            "tokens_est": len(text_ro.split()),
        })
    if docx_en:
        if not docx_en.exists():
            raise SystemExit(f"DOCX not found: {docx_en}")
        text_en = read_docx_text(docx_en).strip()
        title_en = docx_en.stem
        checksum_en = hashlib.sha256(text_en.encode("utf-8", errors="ignore")).hexdigest()
        doc_id_en = hashlib.sha1(f"en\n{title_en}\n{text_en[:2000]}".encode("utf-8", errors="ignore")).hexdigest()
        rows.append({
            "doc_id": doc_id_en,
            "lang": "en",
            "source": str(docx_en),
            "title": title_en,
            "url": "",
            "published_at": None,
            "text": text_en,
            "checksum": checksum_en,
            "tokens_est": len(text_en.split()),
        })

    if not rows:
        raise SystemExit("No DOCX inputs provided. Use --docx_ro and/or --docx_en, or set paths.raw_docs in configs/global.yaml.")
    return pd.DataFrame(rows, columns=[
        "doc_id","lang","source","title","url","published_at","text","checksum","tokens_est"
    ])


# ------------------------------
# Per-document chunking
# ------------------------------
def process_doc(row: Dict, size: int, overlap: int):
    doc_id = row["doc_id"]
    lang = row.get("lang", "")
    text = row.get("text", "") or ""
    toks = whitespace_tokens(text)
    spans = chunk_tokens(toks, size, overlap)
    chunks = []
    for i, (s, e) in enumerate(spans):
        chunk_id = f"{doc_id}_{i:04d}"
        chunk_text = join_tokens(toks[s:e])
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": i,
            "lang": lang,
            "start_token": s,
            "end_token": e,            # exclusive
            "token_count": e - s,
            "text": chunk_text
        })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="chunk size in tokens (default from config)")
    parser.add_argument("--overlap", type=int, default=None, help="overlap size in tokens (default from config)")
    parser.add_argument("--out", type=str, default=None, help="path to chunks parquet/csv")
    parser.add_argument("--docx_ro", type=str, default=None, help="Romanian DOCX file (required if no defaults)")
    parser.add_argument("--docx_en", type=str, default=None, help="English DOCX file (required if no defaults)")
    args = parser.parse_args()

    # Load defaults
    cfg_size, cfg_overlap = load_config_chunking()
    defaults = load_config_paths()

    size = args.size if args.size is not None else cfg_size
    overlap = args.overlap if args.overlap is not None else cfg_overlap
    if overlap >= size:
        raise SystemExit(f"Invalid overlap={overlap} (must be < size={size}).")

    # Resolve inputs: CLI args override config defaults
    docx_ro = Path(args.docx_ro).resolve() if args.docx_ro else defaults["docx_ro"]
    docx_en = Path(args.docx_en).resolve() if args.docx_en else defaults["docx_en"]

    if not (docx_ro or docx_en):
        raise SystemExit(
            "No inputs. Provide --docx_ro/--docx_en or set 'paths.raw_docs' in configs/global.yaml "
            "(e.g., paths.raw_docs: data/raw_docs/corpus_{}.docx)."
        )

    # Build docs DF from DOCX files
    df = build_docs_from_docx(docx_ro, docx_en)

    # Chunk
    all_chunks = []
    for row in df.itertuples(index=False):
        rowd = {k: getattr(row, k) for k in df.columns}
        all_chunks.extend(process_doc(rowd, size=size, overlap=overlap))

    if not all_chunks:
        raise SystemExit("No chunks produced. Check that DOCX texts are non-empty.")

    chunks_df = pd.DataFrame(all_chunks, columns=[
        "doc_id","chunk_id","chunk_index","lang","start_token","end_token","token_count","text"
    ])

    # Resolve output path: CLI --out overrides config; then fallback to default
    out_path = Path(args.out).resolve() if args.out else (defaults["out_chunks"] or DEFAULT_OUTS[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    try:
        if out_path.suffix == ".parquet":
            chunks_df.to_parquet(out_path, index=False)
            actual_path = out_path
        else:
            if out_path.suffix == ".csv":
                chunks_df.to_csv(out_path, index=False, encoding="utf-8")
                actual_path = out_path
            else:
                # unrecognized extension → default to parquet
                chunks_df.to_parquet(DEFAULT_OUTS[0], index=False)
                actual_path = DEFAULT_OUTS[0]
    except Exception:
        # Parquet not available → CSV fallback
        fallback = DEFAULT_OUTS[1] if out_path.suffix == ".parquet" else out_path
        chunks_df.to_csv(fallback, index=False, encoding="utf-8")
        actual_path = fallback

    # Stats
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "config": {"size_tokens": size, "overlap_tokens": overlap},
        "inputs": {
            "docx_ro": str(docx_ro) if docx_ro else None,
            "docx_en": str(docx_en) if docx_en else None,
        },
        "docs_processed": int(len(df)),
        "chunks_total": int(len(chunks_df)),
        "by_lang": {lang: int(n) for lang, n in chunks_df["lang"].value_counts().to_dict().items()},
        "output": str(actual_path.relative_to(ROOT)) if str(actual_path).startswith(str(ROOT)) else str(actual_path)
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("OK")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
