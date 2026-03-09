"""Scrape Paul Graham's essays from paulgraham.com."""

import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://paulgraham.com/"
ARTICLES_URL = urljoin(BASE_URL, "articles.html")
DATA_DIR = Path(__file__).parent.parent / "data"
RATE_LIMIT_SECONDS = 1


def get_essay_urls() -> list[dict]:
    """Fetch the article index and return a list of {title, url} dicts."""
    resp = requests.get(ARTICLES_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    essays = []
    seen = set()
    for link in soup.find_all("a"):
        href = link.get("href", "")
        title = link.get_text(strip=True)
        if not href or not title:
            continue
        # PG essays are .html files in the root directory
        if href.endswith(".html") and "/" not in href and href != "articles.html":
            url = urljoin(BASE_URL, href)
            if url not in seen:
                seen.add(url)
                essays.append({"title": title, "url": url, "filename": href})
    return essays


def extract_essay_text(html: str) -> str:
    """Extract the main text content from an essay's HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # PG essays typically have their content in a <font> tag inside a <table>
    # or directly in the body. Try to get the main text.
    # Remove script and style elements
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # Try to find the main content table (PG's typical layout)
    # The essays usually have content inside nested tables
    text = soup.get_text(separator="\n")

    # Clean up: collapse multiple blank lines, strip each line
    lines = [line.strip() for line in text.splitlines()]
    # Remove leading/trailing empty lines and collapse multiple empty lines
    cleaned = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False

    return "\n".join(cleaned).strip()


def scrape_essays(max_essays: int | None = None) -> list[dict]:
    """Scrape all essays and save them to data/. Returns metadata list."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    essays = get_essay_urls()
    if max_essays:
        essays = essays[:max_essays]

    metadata = []
    for i, essay in enumerate(essays):
        slug = essay["filename"].replace(".html", "")
        txt_path = DATA_DIR / f"{slug}.txt"
        meta_path = DATA_DIR / f"{slug}.json"

        # Skip if already downloaded
        if txt_path.exists() and meta_path.exists():
            print(f"[{i+1}/{len(essays)}] Skipping (cached): {essay['title']}")
            with open(meta_path) as f:
                metadata.append(json.load(f))
            continue

        print(f"[{i+1}/{len(essays)}] Downloading: {essay['title']}")
        try:
            resp = requests.get(essay["url"], timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error downloading {essay['url']}: {e}")
            continue

        text = extract_essay_text(resp.text)
        if len(text) < 100:
            print(f"  Skipping (too short, likely not an essay): {essay['title']}")
            continue

        meta = {
            "title": essay["title"],
            "url": essay["url"],
            "slug": slug,
            "char_count": len(text),
        }

        txt_path.write_text(text, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        metadata.append(meta)

        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\nScraped {len(metadata)} essays to {DATA_DIR}")
    return metadata


if __name__ == "__main__":
    scrape_essays()
