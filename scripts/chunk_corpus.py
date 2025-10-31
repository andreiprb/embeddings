"""
Sentence-aware chunker (spaCy, streaming blocks) with token budget and sentence
overlap — DOCX ONLY.

Inputs:
  data/raw/corpus_{}.docx (RO / EN)

Output:
  data/chunks/chunks.parquet
  reports/chunks/chunk_stats.json

"""

import re
import hashlib
from pathlib import Path
import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
import yaml
import spacy
from spacy.language import Language
from docx import Document

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


def normalize_spaces(text: str) -> str:
    """
    Normalize whitespace in a string.

    :param text: Input text to normalize (may be None or empty).

    :return: The normalized string with single-space separators.
    """
    return " ".join((text or "").split())


def whitespace_tokens(text: str) -> List[str]:
    """
    Tokenize text by ASCII spaces using lightweight normalization.

    :param text: Input text to tokenize (may be None or empty).

    :return: List of tokens obtained by splitting on single spaces.
    """

    if not text:
        return []

    return normalize_spaces(text).split(" ")


def join_tokens(tokens: List[str]) -> str:
    """
    Join tokens with a single ASCII space.

    :param tokens: Sequence of pre-cleaned tokens to join.

    :return: The joined text with single-space separators.
    """
    return " ".join(tokens)


def get_spacy_pipeline(lang: str,
                       names: Dict[str, Optional[str | int]]) -> Language:
    """
    Load a lightweight pipeline that can produce sentence boundaries.
    We exclude heavy components (parser/ner/etc). If the model is missing, use
    fallback or blank + sentencizer.

    :param lang: Language code (2-letter).
    :param names: Mapping of language code -> model name or version.

    :return: The loaded spaCy pipeline.
    """
    want = names.get(lang)
    fallback = names.get("fallback")
    def _load(name: Optional[str]) -> Optional[Language]:
        if not name: return None
        try:
            nlp = spacy.load(
                name=name,
                exclude=[
                    "tok2vec",
                    "tagger",
                    "morphologizer",
                    "attribute_ruler",
                    "lemmatizer",
                    "ner",
                    "parser"
                ]
            )
            if {"senter", "sentencizer"}.isdisjoint(nlp.pipe_names):
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None

    nlp = _load(want) or _load(fallback)
    if nlp is None:
        try:
            nlp = spacy.blank(lang)
        except Exception:
            nlp = spacy.blank("xx")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    try:
        nlp.max_length = max(
            int(getattr(nlp, "max_length", 1_000_000)),
            2_000_000
        )
    except Exception:
        pass
    return nlp


def _split_into_blocks(text: str, block_chars: int) -> List[str]:
    """
    Split long text into blocks ~block_chars, preferably at paragraph boundaries.
    """
    if len(text) <= block_chars:
        return [text.strip()]

    paras = [
        p.strip() for p in re.split(
            r"\n{2,}|\r\n\r\n",
            text
        ) if p.strip()
    ]

    blocks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for p in paras:
        if cur_len + len(p) + 2 <= block_chars:
            cur.append(p); cur_len += len(p) + 2

        else:
            if cur:
                blocks.append("\n\n".join(cur).strip())

            if len(p) > block_chars:
                for i in range(0, len(p), block_chars):
                    blocks.append(p[i:i+block_chars].strip())

                cur = []; cur_len = 0

            else:
                cur = [p]; cur_len = len(p)

    if cur:
        blocks.append("\n\n".join(cur).strip())

    return blocks


def iter_spacy_sentences_streaming(nlp: Language,
                                   text: str,
                                   block_chars: int) -> List[str]:
    """
    Yield sentences by processing the text in blocks to avoid E088.
    """
    blocks = _split_into_blocks(text, block_chars=block_chars)

    try:
        nlp.max_length = max(
            int(getattr(nlp, "max_length", 1_000_000)),
            max(len(b) for b in blocks) + 1000
        )
    except Exception:
        pass

    sentences: List[str] = []

    for doc in nlp.pipe(blocks, batch_size=4):
        for s in doc.sents:
            st = s.text.strip()
            if st:
                sentences.append(st)

    if not sentences:
        t = text.strip()
        return [t] if t else []

    return sentences



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


def read_docx_text(path: Path) -> str:
    if path.suffix.lower() != ".docx":
        raise SystemExit(f"Unsupported file type: {path.name} (need .docx)")

    return _docx_text_python_docx(path)


def build_docs_from_docx(docx_ro: Optional[Path],
                         docx_en: Optional[Path]) -> pd.DataFrame:
    rows: List[Dict] = []

    if docx_ro:
        if not docx_ro.exists():
            raise SystemExit(f"DOCX not found: {docx_ro}")

        text_ro = read_docx_text(docx_ro).strip()
        title_ro = docx_ro.stem
        checksum_ro = hashlib.sha256(
            text_ro.encode("utf-8", errors="ignore")
        ).hexdigest()
        doc_id_ro = hashlib.sha1(
            f"ro\n{title_ro}\n{text_ro[:2000]}".encode(
                encoding="utf-8",
                errors="ignore"
            )
        ).hexdigest()
        rows.append({
            "doc_id": doc_id_ro,
            "lang": "ro",
            "source": str(docx_ro),
            "title": title_ro,
            "url": "",
            "published_at": None,
            "text": text_ro,
            "checksum": checksum_ro,
            "tokens_est": len(whitespace_tokens(text_ro)),
        })
    if docx_en:
        if not docx_en.exists():
            raise SystemExit(f"DOCX not found: {docx_en}")

        text_en = read_docx_text(docx_en).strip()
        title_en = docx_en.stem
        checksum_en = hashlib.sha256(
            text_en.encode(encoding="utf-8", errors="ignore")
        ).hexdigest()
        doc_id_en = hashlib.sha1(
            f"en\n{title_en}\n{text_en[:2000]}".encode(
                encoding="utf-8", errors="ignore"
            )
        ).hexdigest()
        rows.append({
            "doc_id": doc_id_en,
            "lang": "en",
            "source": str(docx_en),
            "title": title_en,
            "url": "",
            "published_at": None,
            "text": text_en,
            "checksum": checksum_en,
            "tokens_est": len(whitespace_tokens(text_en)),
        })

    if not rows:
        raise SystemExit("No DOCX inputs provided. Use --docx_ro and/or --docx"
                         "_en, or set paths.raw_docs in configs/global.yaml.")

    return pd.DataFrame(rows, columns=[
        "doc_id",
        "lang",
        "source",
        "title",
        "url",
        "published_at",
        "text",
        "checksum",
        "tokens_est"
    ])


def chunk_by_sentences_streaming(
    text: str,
    lang: str,
    nlp: Language,
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
    sentences = iter_spacy_sentences_streaming(
        nlp=nlp,
        text=text,
        block_chars=block_chars
    )

    chunks: List[Tuple[int,int,str]] = []
    doc_offset = 0
    cur_sentences: List[List[str]] = []
    cur_len = 0

    for sent in sentences:
        tokens = whitespace_tokens(sent)
        L = len(tokens)

        if L > size_tokens and not cur_sentences:
            stride = max(1, size_tokens - overlap_tokens)
            start = 0

            while start < L:
                end = min(start + size_tokens, L)
                chunk_text = join_tokens(tokens[start:end])
                chunks.append(
                    (doc_offset + start, doc_offset + end, chunk_text)
                )

                if end == L:
                    break

                start += stride

            doc_offset += L
            continue

        if cur_len + L <= size_tokens or not cur_sentences:
            cur_sentences.append(tokens); cur_len += L

        else:
            total = cur_len
            start_tok = doc_offset
            end_tok = start_tok + total
            chunk_text = " ".join(join_tokens(ts) for ts in cur_sentences)
            chunks.append((start_tok, end_tok, chunk_text))

            overlap_sum = 0
            carry: List[List[str]] = []
            for ts in reversed(cur_sentences):
                carry.insert(0, ts)
                overlap_sum += len(ts)
                if overlap_sum >= overlap_tokens:
                    break
            if not carry and cur_sentences:
                carry = [cur_sentences[-1]]
                overlap_sum = len(carry[0])

            committed = total - overlap_sum
            doc_offset += committed

            cur_sentences = carry[:] + [tokens]
            cur_len = overlap_sum + L

    if cur_sentences:
        total = cur_len
        start_tok = doc_offset
        end_tok = start_tok + total
        chunk_text = " ".join(join_tokens(ts) for ts in cur_sentences)
        chunks.append((start_tok, end_tok, chunk_text))

    return chunks


def main():
    global_config = load_yaml_config("global.yaml")

    chunking_group = global_config.get("chunking")
    size = chunking_group.get("size")
    overlap = chunking_group.get("overlap")
    spacy_cfg = chunking_group.get("spacy")

    if size <= 0:
        raise SystemExit("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise SystemExit(f"Invalid overlap={overlap} "
                         f"(must be >=0 and < size={size}).")

    paths_config = load_yaml_config("paths.yaml")
    data_group = paths_config.get("data")
    raw_docx_str = data_group.get("raw")

    chunks_group = paths_config.get("chunks")
    chunks_parquet_str = chunks_group.get("parquet")
    report_str = chunks_group.get("report")

    docx_ro_str = raw_docx_str.format("ro")
    docx_en_str = raw_docx_str.format("en")

    docx_ro = Path(ROOT, docx_ro_str)
    docx_en = Path(ROOT, docx_en_str)

    df = build_docs_from_docx(docx_ro, docx_en)

    nlp_ro = get_spacy_pipeline("ro", spacy_cfg)
    nlp_en = get_spacy_pipeline("en", spacy_cfg)
    block_chars = int(spacy_cfg.get("block_chars") or 100_000)

    records: List[Dict] = []
    for row in df.itertuples(index=False):
        doc_id = getattr(row, "doc_id")
        lang = getattr(row, "lang")
        text = getattr(row, "text") or ""
        nlp = nlp_ro if lang == "ro" else nlp_en
        chunks = chunk_by_sentences_streaming(
            text=text,
            lang=lang,
            nlp=nlp,
            size_tokens=size,
            overlap_tokens=overlap,
            block_chars=block_chars
        )
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
        raise SystemExit("No chunks produced. Check DOCX files.")

    chunks_df = pd.DataFrame(records, columns=[
        "doc_id",
        "chunk_id",
        "chunk_index",
        "lang",
        "start_token",
        "end_token",
        "token_count",
        "text"
    ])

    chunks_path = Path(ROOT, chunks_parquet_str)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_df.to_parquet(chunks_path, index=False)

    out_path = Path(ROOT, report_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_lang = {
        lang: int(n) for lang, n in chunks_df["lang"]
            .value_counts()
            .to_dict()
            .items()
    }

    stats = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "size_tokens": size,
            "overlap_tokens": overlap,
            "spacy_block_chars": block_chars
        },
        "inputs": {
            "docx_ro": str(docx_ro_str),
            "docx_en": str(docx_en_str)
        },
        "docs_processed": int(len(df)),
        "chunks_total": int(len(chunks_df)),
        "by_lang": by_lang,
        "output": chunks_parquet_str
    }

    with open(out_path, "wb") as f:
        f.write(dumps(stats))


if __name__ == "__main__":
    # main()
    print("[FROZEN] No need to run again")

