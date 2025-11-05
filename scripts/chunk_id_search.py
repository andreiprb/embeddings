"""
Interactive text -> chunk_id lookup from Parquet.

Inputs:
  data/chunks/chunks.parquet (expects columns: chunk_id, text)

Behavior:
  - Loads the Parquet once into a fast in-memory structure.
  - Prompts the user for a TEXT fragment (as it appeared in source docs).
  - Normalizes the input text with the same whitespace normalization used
    during chunking.
  - Searches all normalized chunk texts for that fragment.
  - Prints all matching chunk_id(s), plus a short preview.
  - Empty line or Ctrl+D/Ctrl+C exits.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import textwrap

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "chunks" / "chunks.parquet"


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def format_text_block(text: str, width: int = 80) -> str:
    paragraphs = text.split("\n\n")
    wrapped = [
        textwrap.fill(p.strip(), width=width, replace_whitespace=False)
        for p in paragraphs if p.strip()
    ]
    return "\n\n".join(wrapped)


def load_chunks(path: Path) -> List[Dict[str, str]]:
    try:
        df = pd.read_parquet(path, columns=["chunk_id", "text"])
    except FileNotFoundError:
        raise SystemExit(f"Parquet not found: {path}")
    except Exception as e:
        raise SystemExit(f"Error reading Parquet ({path}): {e}")

    missing = {"chunk_id", "text"} - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in Parquet: {missing}")

    df = df.dropna(subset=["chunk_id"]).drop_duplicates(subset=["chunk_id"], keep="first")

    rows: List[Dict[str, str]] = []
    for row in df.itertuples(index=False):
        chunk_id = getattr(row, "chunk_id")
        raw_text = getattr(row, "text") or ""
        norm_text = normalize_spaces(raw_text)
        rows.append({
            "chunk_id": chunk_id,
            "text": raw_text,
            "norm_text": norm_text,
        })
    return rows


def find_chunks_by_text(chunks: List[Dict[str, str]], query: str) -> List[Tuple[str, str]]:
    q = normalize_spaces(query)
    if not q:
        return []
    matches: List[Tuple[str, str]] = []
    for ch in chunks:
        if q in ch["norm_text"]:
            matches.append((ch["chunk_id"], ch["text"]))
    return matches


def prompt_loop(chunks: List[Dict[str, str]]) -> None:
    print(f"Loaded {len(chunks):,} chunks from {PARQUET_PATH}")
    print("Type a TEXT fragment (will be normalized) and press Enter. Empty line to quit.\n")

    while True:
        try:
            q = input("text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            print("Bye.")
            break

        matches = find_chunks_by_text(chunks, q)
        if not matches:
            print("No chunks contain that text.\n")
            continue

        print(f"Found {len(matches)} chunk(s):")
        for cid, raw_txt in matches:
            print(f"\n{cid}")
            print("-" * 60)
            print(format_text_block(raw_txt, width=80))
            print("-" * 60)
        print()


def main() -> None:
    chunks = load_chunks(PARQUET_PATH)
    prompt_loop(chunks)


if __name__ == "__main__":
    main()
