import json
from pathlib import Path

BM25_SEARCH_FOLDER = Path("../reports/bm25/search")
OUTPUT_PATH = Path("../reports/bm25/qrels_base/qrels_base.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def infer_from_field(filename: str) -> str:
    base = Path(filename).stem
    parts = base.split("_")
    if len(parts) >= 2:
        src, tgt = parts[0], parts[1]
        return f"bm25_{src}_{tgt}"
    raise ValueError(f"Nume de fișier neașteptat pentru deducerea 'from': {filename}")

def process_file(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    from_field = infer_from_field(fp.name)

    for rec in data:
        q_info = rec.get("query")
        if isinstance(q_info, str):
            try:
                q_info = json.loads(q_info)
            except json.JSONDecodeError:
                # dacă nu e JSON valid, lăsăm qid necunoscut
                q_info = {}
        elif not isinstance(q_info, dict):
            q_info = {}

        qid = q_info.get("qid", None)
        if qid is None:
            qid = rec.get("qid", "UNKNOWN_QID")

        top_k = rec.get("top_k", [])
        for idx, item in enumerate(top_k[:20]):
            chunk_id = item.get("chunk_id", "UNKNOWN_CHUNK")
            score = item.get("score", None)
            rank = 1 if idx < 5 else 0

            yield {
                "qid": qid,
                "chunk_id": chunk_id,
                "rank": rank,
                "score": score,
                "from": from_field,
                "tags": [],
            }

def main():
    files = sorted(BM25_SEARCH_FOLDER.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nu am găsit fișiere JSON în {BM25_SEARCH_FOLDER.resolve()}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        total = 0
        for fp in files:
            for row in process_file(fp):
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"Written {total} lines in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
