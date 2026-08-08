"""
scripts/summarize_collection.py

Scans data/raw/*/*.jsonl, counts papers per venue/year, and reports how many
have an open-access PDF link available. Run from repo root:

    python scripts/summarize_collection.py
"""

import json
import glob
from pathlib import Path

def summarize():
    files = sorted(glob.glob("data/raw/*/*_records.jsonl"))

    if not files:
        print("No records found yet. Looking for files matching: data/raw/*/*_records.jsonl")
        return

    total_papers = 0
    total_with_pdf = 0
    rows = []

    for filepath in files:
        count = 0
        with_pdf = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                if rec.get("raw_file", {}).get("pdf_url"):
                    with_pdf += 1

        rows.append((filepath, count, with_pdf))
        total_papers += count
        total_with_pdf += with_pdf

    # Print a clean table
    print(f"{'File':<45} {'Papers':>8} {'With PDF':>10} {'Coverage':>10}")
    print("-" * 76)
    for filepath, count, with_pdf in rows:
        coverage = f"{(with_pdf/count*100):.0f}%" if count else "0%"
        print(f"{filepath:<45} {count:>8} {with_pdf:>10} {coverage:>10}")
    print("-" * 76)
    overall_coverage = f"{(total_with_pdf/total_papers*100):.0f}%" if total_papers else "0%"
    print(f"{'TOTAL':<45} {total_papers:>8} {total_with_pdf:>10} {overall_coverage:>10}")


if __name__ == "__main__":
    summarize()