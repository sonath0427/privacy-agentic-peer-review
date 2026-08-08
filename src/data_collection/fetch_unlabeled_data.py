"""
src/data_collection/fetch_unlabeled_data.py

Scrapes papers (no public reviews available) from KDD / IJCAI / KR via the
Semantic Scholar Graph API, and saves them in the project's unified JSON schema.
Mirrors fetch_openreview_data.py's conventions (Sonath) so records from both
scripts merge cleanly downstream.

RESUMABLE: writes one JSON record per line as it goes (not all at the end),
and skips PDFs that are already downloaded. Safe to stop (Ctrl+C) and
re-run the exact same command to continue where you left off.

Usage (run from repo root):
    python src/data_collection/fetch_unlabeled_data.py --venue KDD --year 2025 --outdir data/raw/kdd2025 --download-pdfs

Requires (add to requirements.txt):
    requests, tqdm, python-dotenv

Auth:
    Optional. Reads SEMANTIC_SCHOLAR_API_KEY from a .env file in the repo root
    (already gitignored — never commit this file). Works without a key, but
    you'll hit rate limits much faster — get a free key at:
    https://www.semanticscholar.org/product/api
"""

import os
import json
import time
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

S2_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,abstract,year,venue,openAccessPdf,externalIds,url"

# Semantic Scholar matches on the venue string as stored in its own metadata.
# These are reasonable starting queries — spot-check the first run's results
# and adjust the strings here if you see mismatched papers coming through.
VENUE_QUERIES = {
    ("KDD", 2025): "KDD",
    ("IJCAI", 2025): "IJCAI",
    ("KR", 2025): "KR",
}

VENUE_TIERS = {
    "KDD": "A*",
    "IJCAI": "A*",
    "KR": "A",
}


def get_api_key():
    load_dotenv()
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY")  # None is fine — API works without it


def fetch_all_papers(venue_query, year, api_key=None):
    """
    Paginates through the Semantic Scholar bulk search endpoint until exhausted.

    NOTE: `query` is required by the endpoint even when filtering by venue, so we
    use a wildcard query and rely on the dedicated `venue` param for the actual
    filtering — this avoids fuzzy-matching papers that merely mention the venue
    name in their title/abstract rather than being published there.

    Retries on 429 (rate limited) with exponential backoff instead of crashing —
    the shared unauthenticated pool can throttle you even for small requests if
    someone else is hammering it at the same moment.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": "*",          # required field, but we filter by venue below, not by keyword
        "venue": venue_query,  # exact venue filter (comma-separated list also accepted)
        "year": str(year),
        "fields": FIELDS,
    }

    all_papers = []
    token = None
    while True:
        if token:
            params["token"] = token

        resp = _get_with_retry(S2_BULK_SEARCH_URL, params, headers)
        payload = resp.json()

        all_papers.extend(payload.get("data", []))
        token = payload.get("token")
        if not token:
            break
        time.sleep(1)  # be polite between pages

    return all_papers


def _get_with_retry(url, params, headers, max_retries=6):
    """
    GET request that retries on 429 with exponential backoff.
    Respects the Retry-After header if the server sends one.
    """
    delay = 5  # start at 5s, doubles each retry: 5, 10, 20, 40, 80, 160
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=30)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", delay))
            print(f"  [rate limited] 429 received — waiting {wait}s before retry "
                  f"({attempt}/{max_retries})...")
            time.sleep(wait)
            delay *= 2
            continue

        resp.raise_for_status()  # raises for any other error status (4xx/5xx besides 429)
        return resp

    raise RuntimeError(
        f"Still rate-limited after {max_retries} retries. "
        f"Try again later, or get a free API key: "
        f"https://www.semanticscholar.org/product/api"
    )


def _has_open_access_pdf(paper):
    oa = paper.get("openAccessPdf")
    return oa is not None and oa.get("url")


def parse_paper_to_record(paper, venue_name, year, tier):
    """Same record shape as Sonath's parse_note_to_record, adapted for Semantic Scholar input."""
    paper_id = paper.get("paperId")

    return {
        "paper_id": f"{venue_name.lower()}{year}_{paper_id}",
        "source": {
            "venue": venue_name,
            "year": year,
            "tier": tier,
            "track": "research",
            "url": paper.get("url", "")
        },
        "collector": "Kavyanga",
        "raw_file": {
            "pdf_path": None,
            "pdf_url": paper.get("openAccessPdf", {}).get("url") if _has_open_access_pdf(paper) else None,
            "file_type": "pdf",
            "has_reviews": False
        },
        "parsed": None,
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "reviews": [],
        "meta_review": {},
        "labels": {
            "has_ground_truth": False,
            "recommendation": None
        },
        "quality_flags": {
            "has_open_access_pdf": _has_open_access_pdf(paper)
        }
    }


def download_pdf(pdf_url, save_path: Path):
    """Downloads PDF unless it already exists on disk (resumable). Retries on 429."""
    if save_path.exists() and save_path.stat().st_size > 0:
        return str(save_path)

    if not pdf_url:
        return None  # no open-access PDF available for this paper

    delay = 5
    for attempt in range(1, 5):
        try:
            resp = requests.get(pdf_url, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", delay))
                print(f"  [rate limited] waiting {wait}s before retrying PDF download...")
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return str(save_path)
        except Exception as e:
            print(f"  [warn] could not download PDF for {save_path.stem}: {e}")
            return None
    return None


def load_already_processed_records(out_file: Path):
    """
    Read existing records.jsonl (if any) and return a dict of paper_id -> record.
    Used both to skip re-writing metadata AND to still allow PDF backfill for
    papers whose metadata was already collected before --download-pdfs was used.
    """
    processed = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    processed[rec["paper_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupted last line (e.g. from a mid-write crash)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True, choices=["KDD", "IJCAI", "KR"])
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    venue_query = VENUE_QUERIES.get((args.venue, args.year), args.venue)
    tier = VENUE_TIERS.get(args.venue, "unknown")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_dir = outdir / "pdfs"
    if args.download_pdfs:
        pdf_dir.mkdir(exist_ok=True)

    out_file = outdir / f"{args.venue.lower()}_{args.year}_records.jsonl"

    # ---- Resume support: figure out what's already done ----
    already_processed = load_already_processed_records(out_file)
    if already_processed:
        print(f"Resuming: {len(already_processed)} papers already saved in {out_file}.")
        if args.download_pdfs:
            print("--download-pdfs is on: will backfill missing PDFs for these too, "
                  "without duplicating their metadata.")

    api_key = get_api_key()
    print(f"Fetching papers for {args.venue} {args.year} (query='{venue_query}') ...")
    papers = fetch_all_papers(venue_query, args.year, api_key=api_key)
    print(f"Found {len(papers)} candidate papers.")

    if args.limit:
        papers = papers[:args.limit]
        print(f"Limiting to first {args.limit} for this run.")

    # Open in APPEND mode so we never overwrite previous progress
    with open(out_file, "a", encoding="utf-8") as f:
        for paper in tqdm(papers, desc="Processing"):
            record_id = f"{args.venue.lower()}{args.year}_{paper.get('paperId')}"

            if record_id in already_processed:
                # Metadata already saved — only action left is a possible PDF backfill.
                if args.download_pdfs:
                    existing = already_processed[record_id]
                    pdf_url = existing.get("raw_file", {}).get("pdf_url")
                    save_path = pdf_dir / f"{paper.get('paperId')}.pdf"
                    download_pdf(pdf_url, save_path)
                continue  # never rewrite/duplicate the metadata line

            record = parse_paper_to_record(paper, args.venue, args.year, tier)

            if args.download_pdfs:
                pdf_url = record["raw_file"]["pdf_url"]
                save_path = pdf_dir / f"{paper.get('paperId')}.pdf"
                record["raw_file"]["pdf_path"] = download_pdf(pdf_url, save_path)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # ensure it's written to disk immediately, not buffered

    print(f"Done. Records saved to {out_file}")


if __name__ == "__main__":
    main()