"""
src/data_collection/fetch_openreview_data.py

Scrapes accepted AND rejected papers + full review threads from
ICLR / NeurIPS (OpenReview API v2) and saves them in the project's
unified JSON schema.

RESUMABLE: writes one JSON record per line as it goes (not all at the end),
and skips PDFs that are already downloaded. Safe to stop (Ctrl+C) and
re-run the exact same command to continue where you left off.

Usage (run from repo root):
    python src/data_collection/fetch_openreview_data.py --venue ICLR --year 2025 --outdir data/raw/iclr2025 --download-pdfs

Requires (already in requirements.txt):
    openreview-py, pandas, tqdm, requests, python-dotenv

Auth:
    Reads OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD from a .env file
    in the repo root (already gitignored — never commit this file).
"""

import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openreview.api import OpenReviewClient
from tqdm import tqdm

VENUE_IDS = {
    ("ICLR", 2024): "ICLR.cc/2024/Conference",
    ("ICLR", 2025): "ICLR.cc/2025/Conference",
    ("NeurIPS", 2023): "NeurIPS.cc/2023/Conference",
    ("NeurIPS", 2024): "NeurIPS.cc/2024/Conference",
}


def get_client():
    load_dotenv()
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD not found. "
            "Check your .env file exists in the repo root."
        )
    return OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )


def fetch_all_submissions(client, venue_id):
    submissions = client.get_all_notes(
        content={"venueid": venue_id},
        details="replies"
    )
    if not submissions:
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/Submission",
            details="replies"
        )
    return submissions


def _extract_value(field):
    if isinstance(field, dict):
        return field.get("value")
    return field


def parse_note_to_record(note, venue_name, year, tier="A*"):
    content = note.content
    paper_id = note.id

    reviews = []
    meta_review = None
    decision = None

    for reply in note.details.get("replies", []):
        invitations = reply.get("invitations", []) or [reply.get("invitation", "")]
        reply_content = reply.get("content", {})

        if any("Official_Review" in inv for inv in invitations):
            reviews.append({
                "reviewer_id": reply.get("signatures", ["unknown"])[0],
                "score": _extract_value(reply_content.get("rating")),
                "confidence": _extract_value(reply_content.get("confidence")),
                "review_text": _extract_value(reply_content.get("summary"))
                                or _extract_value(reply_content.get("review")),
                "criteria_scores": {
                    "soundness": _extract_value(reply_content.get("soundness")),
                    "presentation": _extract_value(reply_content.get("presentation")),
                    "contribution": _extract_value(reply_content.get("contribution")),
                }
            })
        elif any("Meta_Review" in inv for inv in invitations):
            meta_review = {
                "decision_text": _extract_value(reply_content.get("metareview"))
                                  or _extract_value(reply_content.get("summary")),
            }
        elif any("Decision" in inv for inv in invitations):
            decision = _extract_value(reply_content.get("decision"))

    return {
        "paper_id": f"{venue_name.lower()}{year}_{paper_id}",
        "source": {
            "venue": venue_name,
            "year": year,
            "tier": tier,
            "track": "main",
            "url": f"https://openreview.net/forum?id={paper_id}"
        },
        "collector": "Sonath",
        "raw_file": {
            "pdf_path": None,
            "file_type": "pdf",
            "has_reviews": len(reviews) > 0
        },
        "parsed": None,
        "title": _extract_value(content.get("title")),
        "abstract": _extract_value(content.get("abstract")),
        "reviews": reviews,
        "meta_review": meta_review or {},
        "labels": {
            "has_ground_truth": decision is not None,
            "recommendation": decision
        },
        "quality_flags": {}
    }


def download_pdf(client, note, outdir: Path):
    """Downloads PDF unless it already exists on disk (resumable)."""
    save_path = outdir / f"{note.id}.pdf"

    # Skip if already downloaded successfully
    if save_path.exists() and save_path.stat().st_size > 0:
        return str(save_path)

    try:
        pdf_bytes = client.get_pdf(note.id)
        with open(save_path, "wb") as f:
            f.write(pdf_bytes)
        return str(save_path)
    except Exception as e:
        print(f"  [warn] could not download PDF for {note.id}: {e}")
        return None


def load_already_processed_ids(out_file: Path):
    """Read existing records.jsonl (if any) and return set of paper_ids already saved."""
    processed = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    processed.add(rec["paper_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupted last line (e.g. from a mid-write crash)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True, choices=["ICLR", "NeurIPS"])
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    venue_id = VENUE_IDS.get((args.venue, args.year))
    if not venue_id:
        raise ValueError(
            f"No venue id mapped for {args.venue} {args.year}. "
            f"Check openreview.net and add it to VENUE_IDS in this script."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_dir = outdir / "pdfs"
    if args.download_pdfs:
        pdf_dir.mkdir(exist_ok=True)

    out_file = outdir / f"{args.venue.lower()}_{args.year}_records.jsonl"

    # ---- Resume support: figure out what's already done ----
    already_processed = load_already_processed_ids(out_file)
    if already_processed:
        print(f"Resuming: {len(already_processed)} papers already saved in {out_file}, will skip those.")

    client = get_client()
    print(f"Fetching submissions for {venue_id} ...")
    submissions = fetch_all_submissions(client, venue_id)
    print(f"Found {len(submissions)} submissions (accepted + rejected + withdrawn).")

    if args.limit:
        submissions = submissions[:args.limit]
        print(f"Limiting to first {args.limit} for this run.")

    # Open in APPEND mode so we never overwrite previous progress
    with open(out_file, "a", encoding="utf-8") as f:
        for note in tqdm(submissions, desc="Processing"):
            record_id = f"{args.venue.lower()}{args.year}_{note.id}"
            if record_id in already_processed:
                continue  # already done, skip entirely (metadata AND pdf)

            record = parse_note_to_record(note, args.venue, args.year)
            if args.download_pdfs:
                record["raw_file"]["pdf_path"] = download_pdf(client, note, pdf_dir)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # ensure it's written to disk immediately, not buffered

    print(f"Done. Records saved to {out_file}")


if __name__ == "__main__":
    main()