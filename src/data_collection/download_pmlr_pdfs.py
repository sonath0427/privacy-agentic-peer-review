import argparse
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


VOLUMES = {
    ("ICML", 2024): "235",
    ("ICML", 2025): "267",
    ("UAI", 2024): "244",
    ("UAI", 2025): "286",
}


def get_pdf_links(volume):
    proceedings_url = f"https://proceedings.mlr.press/v{volume}/"

    print("Reading proceedings page:")
    print(proceedings_url)

    response = requests.get(
        proceedings_url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/125 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()

        if href.lower().endswith(".pdf"):
            pdf_url = urljoin(proceedings_url, href)
            pdf_links.append(pdf_url)

    # Remove duplicates while preserving order
    pdf_links = list(dict.fromkeys(pdf_links))

    return pdf_links


def get_filename_from_url(url):
    parsed = urlparse(url)
    filename = Path(parsed.path).name

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return filename


def is_valid_existing_pdf(path):
    if not path.exists():
        return False

    if path.stat().st_size < 1000:
        return False

    try:
        with open(path, "rb") as f:
            header = f.read(4)

        return header == b"%PDF"

    except Exception:
        return False


def download_pdf(
    session,
    url,
    output_dir,
    sleep_seconds=1.0,
    max_retries=3,
):
    filename = get_filename_from_url(url)

    save_path = output_dir / filename
    temp_path = output_dir / f"{filename}.part"

    # Skip already downloaded valid PDFs
    if is_valid_existing_pdf(save_path):
        return "skipped"

    # Remove bad/partial files from previous attempts
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass

    for attempt in range(1, max_retries + 1):

        try:
            response = session.get(
                url,
                timeout=(10, 30),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/125 Safari/537.36"
                    )
                },
            )

            # -----------------------------
            # Rate limit
            # -----------------------------
            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = 20 * attempt
                else:
                    wait_time = 20 * attempt

                wait_time = min(wait_time, 120)

                print(
                    f"\n429 rate limit for {filename}. "
                    f"Waiting {wait_time}s "
                    f"(attempt {attempt}/{max_retries})"
                )

                time.sleep(wait_time)
                continue

            # -----------------------------
            # Temporary server failures
            # -----------------------------
            if response.status_code in [
                500,
                502,
                503,
                504,
            ]:
                wait_time = min(10 * attempt, 60)

                print(
                    f"\nHTTP {response.status_code} for {filename}. "
                    f"Waiting {wait_time}s "
                    f"(attempt {attempt}/{max_retries})"
                )

                time.sleep(wait_time)
                continue

            # -----------------------------
            # Other HTTP errors
            # -----------------------------
            if response.status_code != 200:

                time.sleep(sleep_seconds)

                return (
                    f"failed HTTP {response.status_code}"
                )

            content = response.content

            # -----------------------------
            # Validate PDF
            # -----------------------------
            if len(content) < 1000:
                time.sleep(sleep_seconds)

                return "failed file-too-small"

            if not content.startswith(b"%PDF"):
                time.sleep(sleep_seconds)

                return "failed not-pdf"

            # -----------------------------
            # Save safely
            # -----------------------------
            with open(temp_path, "wb") as f:
                f.write(content)

            temp_path.replace(save_path)

            # Be polite to server
            time.sleep(sleep_seconds)

            return "downloaded"

        except requests.exceptions.Timeout:

            wait_time = min(5 * attempt, 30)

            print(
                f"\nTimeout for {filename}. "
                f"Retrying in {wait_time}s "
                f"(attempt {attempt}/{max_retries})"
            )

            time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:

            wait_time = min(5 * attempt, 30)

            print(
                f"\nConnection error for {filename}: {e}"
            )

            print(
                f"Retrying in {wait_time}s "
                f"(attempt {attempt}/{max_retries})"
            )

            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:

            wait_time = min(5 * attempt, 30)

            print(
                f"\nRequest error for {filename}: {e}"
            )

            print(
                f"Retrying in {wait_time}s "
                f"(attempt {attempt}/{max_retries})"
            )

            time.sleep(wait_time)

        except Exception as e:

            return f"failed unexpected-error: {e}"

    return "failed after retries"


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download ICML/UAI papers from PMLR "
            "with retries and resume support."
        )
    )

    parser.add_argument(
        "--venue",
        required=True,
        choices=[
            "ICML",
            "UAI",
        ],
    )

    parser.add_argument(
        "--year",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where PDFs will be stored.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help=(
            "Seconds to wait after each successful "
            "download. Default: 1."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N PDFs.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per failed PDF. Default: 3.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help=(
            "Pause after this many newly downloaded PDFs. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--batch-pause",
        type=int,
        default=20,
        help=(
            "Seconds to pause after each batch. "
            "Default: 20."
        ),
    )

    args = parser.parse_args()

    key = (args.venue, args.year)

    if key not in VOLUMES:
        supported = ", ".join(
            f"{venue} {year}"
            for venue, year in VOLUMES
        )

        raise ValueError(
            f"No PMLR volume configured for "
            f"{args.venue} {args.year}. "
            f"Supported: {supported}"
        )

    volume = VOLUMES[key]

    output_dir = Path(args.outdir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_file = (
        output_dir / "failed_downloads.txt"
    )

    pdf_links = get_pdf_links(volume)

    print()
    print(
        f"Found {len(pdf_links)} PDF links."
    )

    if args.limit is not None:
        pdf_links = pdf_links[: args.limit]

        print(
            f"Processing first "
            f"{len(pdf_links)} PDFs only."
        )

    session = requests.Session()

    session.headers.update(
        {
            "Accept": (
                "application/pdf,"
                "application/octet-stream;q=0.9,"
                "*/*;q=0.8"
            ),
            "Connection": "keep-alive",
        }
    )

    downloaded = 0
    skipped = 0

    failed = []

    successful_since_pause = 0

    try:

        for url in tqdm(
            pdf_links,
            desc="Downloading PDFs",
        ):

            result = download_pdf(
                session=session,
                url=url,
                output_dir=output_dir,
                sleep_seconds=args.sleep,
                max_retries=args.max_retries,
            )

            if result == "downloaded":

                downloaded += 1
                successful_since_pause += 1

                if (
                    args.batch_size > 0
                    and successful_since_pause
                    >= args.batch_size
                ):
                    print(
                        f"\nDownloaded "
                        f"{successful_since_pause} "
                        f"new PDFs."
                    )

                    print(
                        f"Pausing for "
                        f"{args.batch_pause} seconds..."
                    )

                    time.sleep(
                        args.batch_pause
                    )

                    successful_since_pause = 0

            elif result == "skipped":

                skipped += 1

            else:

                failed.append(
                    (url, result)
                )

    except KeyboardInterrupt:

        print()
        print(
            "Download interrupted by user."
        )

        print(
            "Already downloaded PDFs "
            "will be kept."
        )

    finally:

        if failed:

            with open(
                failed_file,
                "w",
                encoding="utf-8",
            ) as f:

                for url, reason in failed:

                    f.write(
                        f"{reason}\t{url}\n"
                    )

        session.close()

    print()
    print("=" * 50)
    print("Finished")
    print("=" * 50)

    print(
        f"Downloaded : {downloaded}"
    )

    print(
        f"Skipped    : {skipped}"
    )

    print(
        f"Failed     : {len(failed)}"
    )

    print(
        f"Total      : {len(pdf_links)}"
    )

    if failed:

        print()
        print(
            "Failed URLs saved to:"
        )

        print(
            failed_file
        )

    print()
    print(
        "You can run the same command again."
    )

    print(
        "Existing valid PDFs will "
        "automatically be skipped."
    )


if __name__ == "__main__":
    main()