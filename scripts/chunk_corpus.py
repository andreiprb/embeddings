#!/usr/bin/env python3
"""
Sentence-aware chunker (spaCy, streaming blocks) with token budget and sentence overlap — DOCX ONLY.

Run with no args (uses configs/global.yaml defaults):
  paths:
    raw_docs: data/raw_docs/corpus_{}.docx # expands to ..._ro.docx and ..._en.docx
    chunks_parquet: data/chunks.parquet
  chunking:
    size_tokens: 300
    overlap_tokens: 60
  spacy:
    ro: ro_core_news_sm
    en: en_core_web_sm
    fallback: xx_sent_ud_sm
    block_chars: 100000

Or override via CLI:
  python scripts/chunk_corpus.py --docx_ro ro.docx --docx_en en.docx --size 300 --overlap 60 --out data/chunks.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# --- optional deps ---
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

import pandas as pd

# spaCy
try:
    import spacy
    from spacy.language import Language
except Exception as e:
    raise SystemExit("spaCy is required. Install: pip install spacy") from e

# DOCX
try:
    from docx import Document  # python-docx
except Exception:
    Document = None  # type: ignore

try:
    import docx2txt  # fallback
except Exception:
    docx2txt = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTS = [ROOT / "data" / "chunks.parquet", ROOT / "data" / "chunks.csv"]
CONFIG_PATH = ROOT / "configs" / "global.yaml"
STATS_PATH = ROOT / "data" / "chunk_stats.json"


# ------------------------------
# Config helpers
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
    result = {"docx_ro": None, "docx_en": None, "out_chunks": None}
    if not (yaml and CONFIG_PATH.exists()):
        return result
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    p = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    raw_pat = p.get("raw_docs")
    if isinstance(raw_pat, str) and "{}" in raw_pat:
        ro = Path(raw_pat.format("ro")); en = Path(raw_pat.format("en"))
        if not ro.is_absolute(): ro = (ROOT / ro).resolve()
        if not en.is_absolute(): en = (ROOT / en).resolve()
        result["docx_ro"] = ro; result["docx_en"] = en
    outp = p.get("chunks_parquet")
    if isinstance(outp, str):
        out = Path(outp);
        if not out.is_absolute(): out = (ROOT / out).resolve()
        result["out_chunks"] = out
    return result


def load_config_spacy() -> Dict[str, Optional[str | int]]:
    cfg_out: Dict[str, Optional[str | int]] = {
        "ro": "ro_core_news_sm",
        "en": "en_core_web_sm",
        "fallback": "xx_sent_ud_sm",
        "block_chars": 100_000,  # stream at ~100k chars per block
    }
    if yaml and CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        sp = (cfg.get("spacy") or {}) if isinstance(cfg, dict) else {}
        for k in ("ro", "en", "fallback"):
            if isinstance(sp.get(k), str):
                cfg_out[k] = sp[k]
        if isinstance(sp.get("block_chars"), int):
            cfg_out["block_chars"] = int(sp["block_chars"])
    return cfg_out


# ------------------------------
# Basic token helpers (for budgets)
# ------------------------------
def normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def whitespace_tokens(text: str) -> List[str]:
    if not text:
        return []
    return normalize_spaces(text).split(" ")


def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)


# ------------------------------
# spaCy sentence split (streaming)
# ------------------------------
def get_spacy_pipeline(lang: str, names: Dict[str, Optional[str | int]]) -> "Language":
    """
    Load a lightweight pipeline that can produce sentence boundaries.
    We exclude heavy components (parser/ner/etc). If model missing, use fallback or blank+sentencizer.
    """
    want = names.get(lang)
    fallback = names.get("fallback")
    def _load(name: Optional[str]) -> Optional[Language]:
        if not name: return None
        try:
            # exclude heavy pipes, keep senter if present
            nlp = spacy.load(name, exclude=["tok2vec","tagger","morphologizer","attribute_ruler","lemmatizer","ner","parser"])
            if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None

    nlp = _load(want) or _load(fallback)
    if nlp is None:
        # last resort: blank with sentencizer
        try:
            nlp = spacy.blank(lang)
        except Exception:
            nlp = spacy.blank("xx")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    # set a generous max_length (we will still stream, but be safe)
    try:
        nlp.max_length = max(int(getattr(nlp, "max_length", 1_000_000)), 2_000_000)
    except Exception:
        pass
    return nlp


def _split_into_blocks(text: str, block_chars: int) -> List[str]:
    """
    Split long text into blocks ~block_chars, preferably at paragraph boundaries.
    """
    if len(text) <= block_chars:
        return [text.strip()]
    # Prefer paragraph breaks
    paras = [p.strip() for p in re.split(r"\n{2,}|\r\n\r\n", text) if p.strip()]
    blocks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in paras:
        if cur_len + len(p) + 2 <= block_chars:
            cur.append(p); cur_len += len(p) + 2
        else:
            if cur:
                blocks.append("\n\n".join(cur).strip())
            # very long single paragraph -> hard wrap by chars
            if len(p) > block_chars:
                for i in range(0, len(p), block_chars):
                    blocks.append(p[i:i+block_chars].strip())
                cur = []; cur_len = 0
            else:
                cur = [p]; cur_len = len(p)
    if cur:
        blocks.append("\n\n".join(cur).strip())
    return blocks


def iter_spacy_sentences_streaming(nlp: "Language", text: str, block_chars: int) -> List[str]:
    """
    Yield sentences by processing the text in blocks to avoid E088.
    """
    blocks = _split_into_blocks(text, block_chars=block_chars)
    # Ensure max_length covers the largest block
    try:
        nlp.max_length = max(int(getattr(nlp, "max_length", 1_000_000)), max(len(b) for b in blocks) + 1000)
    except Exception:
        pass

    sents: List[str] = []
    for doc in nlp.pipe(blocks, batch_size=4):
        for s in doc.sents:
            st = s.text.strip()
            if st:
                sents.append(st)
    if not sents:
        t = text.strip()
        return [t] if t else []
    return sents


# ------------------------------
# DOCX ingestion
# ------------------------------
def _docx_text_python_docx(path: Path) -> str:
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
        raise SystemExit("Install one DOCX reader: pip install python-docx (or) pip install docx2txt")


def build_docs_from_docx(docx_ro: Optional[Path], docx_en: Optional[Path]) -> pd.DataFrame:
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
            "doc_id": doc_id_ro, "lang": "ro", "source": str(docx_ro), "title": title_ro,
            "url": "", "published_at": None, "text": text_ro, "checksum": checksum_ro,
            "tokens_est": len(whitespace_tokens(text_ro)),
        })
    if docx_en:
        if not docx_en.exists():
            raise SystemExit(f"DOCX not found: {docx_en}")
        text_en = read_docx_text(docx_en).strip()
        title_en = docx_en.stem
        checksum_en = hashlib.sha256(text_en.encode("utf-8", errors="ignore")).hexdigest()
        doc_id_en = hashlib.sha1(f"en\n{title_en}\n{text_en[:2000]}".encode("utf-8", errors="ignore")).hexdigest()
        rows.append({
            "doc_id": doc_id_en, "lang": "en", "source": str(docx_en), "title": title_en,
            "url": "", "published_at": None, "text": text_en, "checksum": checksum_en,
            "tokens_est": len(whitespace_tokens(text_en)),
        })
    if not rows:
        raise SystemExit("No DOCX inputs provided. Use --docx_ro and/or --docx_en, or set paths.raw_docs in configs/global.yaml.")
    return pd.DataFrame(rows, columns=[
        "doc_id","lang","source","title","url","published_at","text","checksum","tokens_est"
    ])


# ------------------------------
# Sentence-aware packing (streaming)
# ------------------------------
def chunk_by_sentences_streaming(
    text: str,
    lang: str,
    nlp: "Language",
    size_tokens: int,
    overlap_tokens: int,
    block_chars: int
) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start_token_idx, end_token_idx, chunk_text).
    - stream sentences from spaCy over blocks to avoid E088
    - pack whole sentences up to size_tokens
    - overlap with sentences until >= overlap_tokens (min 1 sentence)
    - if a single sentence > size_tokens, split that sentence by token windows
    """
    # iterate sentences as a flat list (streaming)
    sentences = iter_spacy_sentences_streaming(nlp, text, block_chars=block_chars)

    chunks: List[Tuple[int,int,str]] = []
    doc_offset = 0  # tokens committed before current window
    cur_sents: List[List[str]] = []   # list of token lists
    cur_len = 0

    for sent in sentences:
        toks = whitespace_tokens(sent)
        L = len(toks)

        # If a single very long sentence -> hard split with stride
        if L > size_tokens and not cur_sents:
            stride = max(1, size_tokens - overlap_tokens)
            start = 0
            while start < L:
                end = min(start + size_tokens, L)
                chunk_text = join_tokens(toks[start:end])
                chunks.append((doc_offset + start, doc_offset + end, chunk_text))
                if end == L:
                    break
                start += stride
            doc_offset += L
            continue

        # Greedy pack into current window
        if cur_len + L <= size_tokens or not cur_sents:
            cur_sents.append(toks); cur_len += L
        else:
            # finalize current chunk
            total = cur_len
            start_tok = doc_offset
            end_tok = start_tok + total
            chunk_text = " ".join(join_tokens(ts) for ts in cur_sents)
            chunks.append((start_tok, end_tok, chunk_text))

            # compute sentence overlap
            overlap_sum = 0
            carry: List[List[str]] = []
            for ts in reversed(cur_sents):
                carry.insert(0, ts)
                overlap_sum += len(ts)
                if overlap_sum >= overlap_tokens:
                    break
            if not carry and cur_sents:
                carry = [cur_sents[-1]]
                overlap_sum = len(carry[0])

            # advance offsets
            committed = total - overlap_sum
            doc_offset += committed

            # start new window with carry + current sentence
            cur_sents = carry[:] + [toks]
            cur_len = overlap_sum + L

    # flush tail
    if cur_sents:
        total = cur_len
        start_tok = doc_offset
        end_tok = start_tok + total
        chunk_text = " ".join(join_tokens(ts) for ts in cur_sents)
        chunks.append((start_tok, end_tok, chunk_text))

    return chunks


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="max tokens per chunk (default from config)")
    parser.add_argument("--overlap", type=int, default=None, help="target token overlap between chunks (default from config)")
    parser.add_argument("--out", type=str, default=None, help="path to chunks parquet/csv")
    parser.add_argument("--docx_ro", type=str, default=None, help="Romanian DOCX file (required if no defaults)")
    parser.add_argument("--docx_en", type=str, default=None, help="English DOCX file (required if no defaults)")
    args = parser.parse_args()

    # load defaults
    cfg_size, cfg_overlap = load_config_chunking()
    defaults = load_config_paths()
    spacy_cfg = load_config_spacy()

    size = args.size if args.size is not None else cfg_size
    overlap = args.overlap if args.overlap is not None else cfg_overlap
    if size <= 0:
        raise SystemExit("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise SystemExit(f"Invalid overlap={overlap} (must be >=0 and < size={size}).")

    # resolve inputs (CLI overrides config)
    docx_ro = Path(args.docx_ro).resolve() if args.docx_ro else defaults["docx_ro"]
    docx_en = Path(args.docx_en).resolve() if args.docx_en else defaults["docx_en"]
    if not (docx_ro or docx_en):
        raise SystemExit(
            "No inputs. Provide --docx_ro/--docx_en or set 'paths.raw_docs' in configs/global.yaml "
            "(e.g., paths.raw_docs: data/raw_docs/corpus_{}.docx)."
        )

    # build docs
    df = build_docs_from_docx(docx_ro, docx_en)

    # load spaCy pipelines once per language
    nlp_ro = get_spacy_pipeline("ro", spacy_cfg)
    nlp_en = get_spacy_pipeline("en", spacy_cfg)
    block_chars = int(spacy_cfg.get("block_chars") or 100_000)

    # chunk
    records: List[Dict] = []
    for row in df.itertuples(index=False):
        doc_id = getattr(row, "doc_id")
        lang = getattr(row, "lang")
        text = getattr(row, "text") or ""
        nlp = nlp_ro if lang == "ro" else nlp_en
        chunks = chunk_by_sentences_streaming(text, lang, nlp, size_tokens=size, overlap_tokens=overlap, block_chars=block_chars)
        for idx, (s_tok, e_tok, chunk_text) in enumerate(chunks):
            records.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{idx:04d}",
                "chunk_index": idx,
                "lang": lang,
                "start_token": int(s_tok),
                "end_token": int(e_tok),   # exclusive
                "token_count": int(e_tok - s_tok),
                "text": chunk_text
            })

    if not records:
        raise SystemExit("No chunks produced. Check that DOCX texts are non-empty.")

    chunks_df = pd.DataFrame(records, columns=[
        "doc_id","chunk_id","chunk_index","lang","start_token","end_token","token_count","text"
    ])

    # resolve output path
    out_path = Path(args.out).resolve() if args.out else (defaults["out_chunks"] or DEFAULT_OUTS[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # write
    try:
        if out_path.suffix == ".parquet":
            chunks_df.to_parquet(out_path, index=False)
            actual_path = out_path
        else:
            if out_path.suffix == ".csv":
                chunks_df.to_csv(out_path, index=False, encoding="utf-8")
                actual_path = out_path
            else:
                chunks_df.to_parquet(DEFAULT_OUTS[0], index=False)
                actual_path = DEFAULT_OUTS[0]
    except Exception:
        fallback = DEFAULT_OUTS[1] if out_path.suffix == ".parquet" else out_path
        chunks_df.to_csv(fallback, index=False, encoding="utf-8")
        actual_path = fallback

    # stats
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "config": {
            "size_tokens": size, "overlap_tokens": overlap, "spacy_block_chars": block_chars
        },
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
