"""
Summarize query distribution of queries and separates them into 4 categories.

Input:
  queries/eval.jsonl

Outputs:
  reports/queries/queries_report.json
  queries/monolingual/{}.jsonl (ro_ro / en_en)
  queries/crosslingual/{}.jsonl (ro_en / en_ro)
"""

import json
import re
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"[line {i}] Invalid JSON: {e}")
            rows.append(obj)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    qid_pat = re.compile(r"^(ro|en)_(ro|en)_(\d{3})$")
    pair_counts = Counter()
    slice_counts = Counter()
    source_counts = Counter()
    paper_counts = Counter()
    papers_total = 0

    total = 0
    bad_qid_format: List[Tuple[int, str]] = []
    bad_alignment: List[Tuple[int, str, str, str]] = []
    missing_fields: List[Tuple[int, List[str]]] = []

    for idx, r in enumerate(rows, 1):
        total += 1
        qid = str(r.get("qid", ""))
        lang = r.get("lang")
        tgt = r.get("target_lang")
        slc = r.get("slice")
        src = r.get("source")

        missing = [k for k in (
            "qid", "query", "lang", "target_lang", "slice", "source"
        ) if r.get(k) is None]
        if missing:
            missing_fields.append((idx, missing))

        m = qid_pat.fullmatch(qid)
        if not m:
            bad_qid_format.append((idx, qid))
            pair = f"{lang}_{tgt}"
        else:
            q_lang, q_tgt, _ = m.groups()
            pair = f"{q_lang}_{q_tgt}"
            if lang != q_lang or tgt != q_tgt:
                bad_alignment.append((idx, qid, str(lang), str(tgt)))

        pair_counts[pair] += 1

        if slc:
            slice_counts[str(slc)] += 1

        if src:
            s = str(src)
            if s.startswith("paper"):
                source_counts["paper"] += 1
                papers_total += 1
                m2 = re.fullmatch(r"paper(\d+)", s)
                if m2:
                    paper_counts[s] += 1
                else:
                    paper_counts[s] += 1
            elif s == "metaphor_list":
                source_counts["metaphor_list"] += 1
            elif s == "synthetic":
                source_counts["synthetic"] += 1
            else:
                pass

    issues = {
        "bad_qid_format": bad_qid_format,
        "bad_alignment": bad_alignment,
        "missing_fields": missing_fields,
        "num_bad_qid_format": len(bad_qid_format),
        "num_bad_alignment": len(bad_alignment),
        "num_missing_fields": len(missing_fields),
    }

    return {
        "total": total,
        "pairs": dict(pair_counts),
        "slices": dict(slice_counts),
        "sources": dict(source_counts),
        "papers_total": papers_total,
        "papers_breakdown": dict(paper_counts),
        "issues": issues,
    }


def print_summary(S: Dict[str, Any]) -> None:
    total = S["total"]
    pairs = S["pairs"]
    slices = S["slices"]
    sources = S["sources"]
    papers_total = S["papers_total"]
    paper_break = S["papers_breakdown"]
    issues = S["issues"]

    def pct(x: int) -> str:
        return f"{(100.0 * x / total):5.1f}%" if total else "  0.0%"

    print("\n=== QUERY SUMMARY ===")
    print(f"Total QIDs: {total}")

    print("\nBy language pair (lang_target):")
    for key in sorted(pairs.keys()):
        print(f"  {key:8s} : {pairs[key]:4d}  ({pct(pairs[key])})")

    print("\nBy slice:")
    for key in sorted(slices.keys()):
        print(f"  {key:16s} : {slices[key]:4d}  ({pct(slices[key])})")

    print("\nBy source (aggregated):")
    for key in ["metaphor_list", "paper", "synthetic"]:
        val = sources.get(key, 0)
        print(f"  {key:16s} : {val:4d}  ({pct(val)})")

    def paper_sort_key(k):
        m = re.fullmatch(r"paper(\d+)", k)
        if k == "paper":
            return 0, 0, k

        if m:
            return 1, int(m.group(1)), k

        return 2, 0, k

    print("\nPapers (overall + breakdown):")
    print(f"  papers_total      : {papers_total:4d}  ({pct(papers_total)})")
    for key in sorted(paper_break.keys(), key=paper_sort_key):
        print(
            f"  {key:16s} : {paper_break[key]:4d}  ({pct(paper_break[key])})"
        )

    print("\nValidation issues:")
    print(f"  bad_qid_format    : {issues['num_bad_qid_format']}")
    print(f"  bad_alignment     : {issues['num_bad_alignment']}")
    print(f"  missing_fields    : {issues['num_missing_fields']}")
    if issues["bad_qid_format"]:
        print("  examples bad_qid_format:")
        for (i, q) in issues["bad_qid_format"]:
            print(f"    line {i}: {q}")
    if issues["bad_alignment"]:
        print("  examples bad_alignment (qid vs lang/target_lang):")
        for (i, q, l, t) in issues["bad_alignment"]:
            print(f"    line {i}: {q}  fields=({l}->{t})")
    if issues["missing_fields"]:
        print("  examples missing_fields:")
        for (i, miss) in issues["missing_fields"]:
            print(f"    line {i}: missing {miss}")

    print("\nDone.\n")


def write_csv(S: Dict[str, Any], out_path: Path) -> None:
    """
    Writes a flat CSV with category, key, count.
    Uses aggregated sources (metaphor_list, paper, synthetic).
    """
    rows = []
    def add(cat: str, key: str, val: int) -> None:
        rows.append((cat, key, str(val)))

    add("total", "total_qids", S["total"])
    for k, v in S["pairs"].items():
        add("pair", k, v)
    for k, v in S["slices"].items():
        add("slice", k, v)
    # aggregated sources only
    for k in ["metaphor_list", "paper", "synthetic"]:
        add("source", k, S["sources"].get(k, 0))
    add("papers", "papers_total", S["papers_total"])
    for k, v in S["papers_breakdown"].items():
        add("papers_breakdown", k, v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("category,key,count\n")
        for cat, key, val in rows:
            f.write(f"{cat},{key},{val}\n")


def main():
    paths_config = load_yaml_config("paths.yaml")

    queries_group = paths_config.get("queries")
    eval_str = queries_group.get("eval")
    report_str = queries_group.get("report")

    ro_ro_str = queries_group.get("ro_ro")
    en_en_str = queries_group.get("en_en")
    ro_en_str = queries_group.get("ro_en")
    en_ro_str = queries_group.get("en_ro")

    in_path = Path(ROOT, eval_str)
    if not in_path.exists():
        print(f"ERROR: file not found: {in_path}")
        sys.exit(1)

    try:
        rows = load_jsonl(in_path)
    except SystemExit as e:
        print(str(e))
        sys.exit(1)

    S = summarize(rows)
    print_summary(S)

    assert S["issues"]["num_bad_alignment"] == 0, (
        "QID/lang alignment errors detected: "
        f"{S['issues']['num_bad_alignment']} rows where qid != "
        f"'{'{lang}_{target_lang}_NNN'}'. "
        "Re-run with --show-errors N to see examples and fix "
        "qid/lang/target_lang."
    )

    out_path = Path(ROOT, report_str)

    write_csv(S, out_path)
    print(f"CSV written to: {report_str}")

    for file_str in [ro_ro_str, en_en_str, ro_en_str, en_ro_str]:
        path = Path(ROOT, file_str)
        path.parent.mkdir(parents=True, exist_ok=True)

    outputs = {
        "ro_ro": open(Path(ROOT, ro_ro_str), "w", encoding="utf-8"),
        "en_en": open(Path(ROOT, en_en_str), "w", encoding="utf-8"),
        "ro_en": open(Path(ROOT, ro_en_str), "w", encoding="utf-8"),
        "en_ro": open(Path(ROOT, en_ro_str), "w", encoding="utf-8"),
    }

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qid", "")
            for prefix in outputs.keys():
                if qid.startswith(prefix + "_"):
                    if "query" in obj and obj["query"]:
                        obj["query"] = obj["query"].lower()

                    json.dump(obj, outputs[prefix], ensure_ascii=False)
                    outputs[prefix].write("\n")
                    break

    for f in outputs.values():
        f.close()


if __name__ == "__main__":
    # main()
    print("[FROZEN] No need to run again")
