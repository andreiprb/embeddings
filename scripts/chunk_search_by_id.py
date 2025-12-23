"""
Interactive chunk lookup from Parquet.

Inputs:
  data/chunks/chunks.parquet (expects columns: chunk_id, text)

Behavior:
  - Loads the Parquet once into a fast in-memory lookup.
  - Prompts the user for a string `chunk_id` and prints the matching text.
  - Text is line-wrapped at 80 characters for readability.
  - Empty line or Ctrl+D/Ctrl+C exits.
"""

from pathlib import Path
from typing import Dict, Optional
import textwrap

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "chunks" / "chunks.parquet"


def format_text_block(text: str, width: int = 80) -> str:
    paragraphs = text.split("\n\n")
    wrapped = [
        textwrap.fill(p.strip(), width=width, replace_whitespace=False)
        for p in paragraphs if p.strip()
    ]
    return "\n\n".join(wrapped)


def load_lookup(path: Path) -> Dict[str, str]:
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
    return dict(df.set_index("chunk_id")["text"])  # type: ignore[return-value]


def get_text_by_chunk_id(lookup: Dict[str, str], chunk_id: str) -> Optional[str]:
    return lookup.get(chunk_id)


def prompt_loop(lookup: Dict[str, str]) -> None:
    print(f"Loaded {len(lookup):,} chunks from {PARQUET_PATH}")
    print("Type a chunk_id and press Enter. Empty line to quit.\n")

    while True:
        try:
            q = input("chunk_id> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            print("Bye.")
            break

        text = get_text_by_chunk_id(lookup, q)
        if text is None:
            print("Not found.\n")
        else:
            print("\n" + format_text_block(
                text,
                width=80
            ) + "\n" + "-" * 40 + "\n")


def main() -> None:
    lookup = load_lookup(PARQUET_PATH)
    prompt_loop(lookup)


if __name__ == "__main__":
    main()
