"""Vector similarity search over the FAISS index."""

import numpy as np
from openai import OpenAI

from .indexer import EMBEDDING_MODEL, load_index


class Retriever:
    """Loads the FAISS index and retrieves relevant chunks for a query."""

    def __init__(self, client: OpenAI | None = None):
        self.index, self.chunks = load_index()
        self.client = client or OpenAI()

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL, input=[query]
        )
        embedding = np.array([response.data[0].embedding], dtype="float32")
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for the top-k most similar chunks to the query.

        Returns a list of dicts with keys: text, metadata, score.
        """
        query_embedding = self._embed_query(query)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score),
            })
        return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    retriever = Retriever()
    results = retriever.retrieve("What does Paul Graham think about startups?")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['metadata']['title']}")
        print(r["text"][:200] + "...")
