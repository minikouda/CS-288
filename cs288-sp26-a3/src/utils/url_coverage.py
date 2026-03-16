import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, cast
from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """Normalize URLs for stable membership checks across minor formatting differences."""
    parsed = urlparse((url or "").strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    # Keep host and path; ignore query/fragment for coverage checks.
    return f"{scheme}://{netloc}{path}"


def load_urls_from_jsonl(path: str, url_field: str = "url") -> Set[str]:
    urls: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            url = row.get(url_field)
            if isinstance(url, str) and url.strip():
                urls.add(normalize_url(url))
    return urls


def load_urls_from_page_json_dir(directory: str, pattern: str = "page_*.json") -> Set[str]:
    """Load URLs from JSON files (one JSON object per file) under a directory."""
    urls: Set[str] = set()
    for file_path in sorted(Path(directory).glob(pattern)):
        if not file_path.is_file():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            row = json.load(f)
            url = row.get("url")
            if isinstance(url, str) and url.strip():
                urls.add(normalize_url(url))
    return urls


def compare_reference_urls(
    reference_jsonl: str,
    corpus_jsonl: str,
    page_json_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Compare URLs in reference file against URLs in corpus file.

    Returns a dictionary with summary stats and URL lists.
    """
    reference_urls = load_urls_from_jsonl(reference_jsonl)
    corpus_urls = load_urls_from_jsonl(corpus_jsonl)

    matched = sorted(reference_urls & corpus_urls)
    missing = sorted(reference_urls - corpus_urls)

    report: Dict[str, Any] = {
        "reference_total": len(reference_urls),
        "corpus_total": len(corpus_urls),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "matched_urls": matched,
        "missing_urls": missing,
    }

    if page_json_dir:
        page_urls = load_urls_from_page_json_dir(page_json_dir)
        matched_in_pages = sorted(reference_urls & page_urls)
        missing_in_pages = sorted(reference_urls - page_urls)
        report.update(
            {
                "page_dir": page_json_dir,
                "page_urls_total": len(page_urls),
                "matched_in_pages_count": len(matched_in_pages),
                "missing_in_pages_count": len(missing_in_pages),
                "matched_in_pages_urls": matched_in_pages,
                "missing_in_pages_urls": missing_in_pages,
            }
        )

    return report


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "URL Coverage Report",
        "=" * 40,
        f"Reference URLs: {report['reference_total']}",
        f"Corpus URLs:    {report['corpus_total']}",
        f"Matched URLs:   {report['matched_count']}",
        f"Missing URLs:   {report['missing_count']}",
        "",
    ]

    missing_urls = cast(List[str], report["missing_urls"])
    lines.append("Missing URLs:")
    if missing_urls:
        lines.extend([f"- {url}" for url in missing_urls])
    else:
        lines.append("- None")

    if "page_urls_total" in report:
        lines.extend(
            [
                "",
                "Page JSON Coverage",
                "=" * 40,
                f"Page URLs:       {report['page_urls_total']}",
                f"Matched URLs:    {report['matched_in_pages_count']}",
                f"Missing URLs:    {report['missing_in_pages_count']}",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check which reference URLs are present in a corpus JSONL file."
    )
    parser.add_argument(
        "--reference",
        default="data/reference.jsonl",
        help="Path to reference JSONL (default: data/reference.jsonl)",
    )
    parser.add_argument(
        "--corpus",
        default="data/raw/corpus_raw2.jsonl",
        help="Path to corpus JSONL (default: data/raw/corpus_raw2.jsonl)",
    )
    parser.add_argument(
        "--page-dir",
        default="data/processed",
        help="Directory containing page_*.json files (default: data/processed)",
    )

    args = parser.parse_args()
    report = compare_reference_urls(args.reference, args.corpus, args.page_dir)
    print(format_report(report))


if __name__ == "__main__":
    main()
