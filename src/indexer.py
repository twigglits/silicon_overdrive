"""Embed chunks and build a FAISS index."""

import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from .chunker import chunk_essays

INDEX_DIR = Path(__file__).parent.parent / "index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100


def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    """Embed a list of texts using OpenAI, batching for efficiency."""
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    return np.array(all_embeddings, dtype="float32")


def build_index(chunk_size: int = 500, chunk_overlap: int = 50) -> None:
    """Chunk essays, embed them, and build a FAISS index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # Chunk the essays
    chunks = chunk_essays(chunk_size, chunk_overlap)
    if not chunks:
        print("No chunks to index. Run the scraper first.")
        return

    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    # Embed
    print(f"Embedding {len(texts)} chunks...")
    embeddings = get_embeddings(texts, client)

    # Normalize for inner product (cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Build FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Save chunk texts + metadata together
    chunk_data = [{"text": t, "metadata": m} for t, m in zip(texts, metadata)]
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump(chunk_data, f, indent=2)

    print(f"Index built: {index.ntotal} vectors saved to {INDEX_DIR}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load a pre-built FAISS index and chunk data."""
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "chunks.json") as f:
        chunks = json.load(f)

    return index, chunks


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    build_index()
