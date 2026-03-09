"""Recursive text chunker that splits paragraphs → sentences → characters."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Separators ordered from most to least preferred split boundary
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text, keeping the separator at the end of each piece."""
    if not separator:
        return list(text)
    parts = text.split(separator)
    # Re-attach separator to each part except the last
    result = [p + separator for p in parts[:-1]]
    if parts[-1]:
        result.append(parts[-1])
    return result


def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text recursively, trying coarser separators first.

    Tries to split on paragraph boundaries, then sentences, then words,
    then characters. Each split is further subdivided if chunks exceed
    chunk_size.
    """
    if separators is None:
        separators = SEPARATORS

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator that exists in the text
    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    parts = _split_on_separator(text, separator)
    remaining_separators = (
        separators[separators.index(separator) + 1 :]
        if separator in separators
        else separators[-1:]
    )

    # Merge parts into chunks that respect chunk_size
    chunks = []
    current = ""

    for part in parts:
        candidate = current + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                sub_chunks = recursive_split(
                    part, chunk_size, chunk_overlap, remaining_separators
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Apply overlap: each chunk starts with the tail of the previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:]
            # Snap to word boundary
            space_idx = overlap_text.find(" ")
            if space_idx != -1:
                overlap_text = overlap_text[space_idx + 1 :]
            overlapped.append(overlap_text + chunks[i])
        chunks = overlapped

    return [c.strip() for c in chunks if c.strip()]


def chunk_essays(
    chunk_size: int = 500, chunk_overlap: int = 50
) -> list[dict]:
    """Load all essays from data/ and chunk them. Returns list of chunk dicts."""
    all_chunks = []

    meta_files = sorted(DATA_DIR.glob("*.json"))
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        txt_path = DATA_DIR / f"{meta['slug']}.txt"
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        text_chunks = recursive_split(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "title": meta["title"],
                    "url": meta["url"],
                    "slug": meta["slug"],
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                },
            })

    print(f"Created {len(all_chunks)} chunks from {len(meta_files)} essays")
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_essays()
    # Print some stats
    lengths = [len(c["text"]) for c in chunks]
    print(f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")
