"""End-to-end RAG pipeline tying retriever and generator together."""

from openai import OpenAI

from .generator import generate
from .retriever import Retriever


class RAGPipeline:
    """Wires together retrieval and generation for question answering."""

    def __init__(self, top_k: int = 5, client: OpenAI | None = None):
        self.top_k = top_k
        self.client = client or OpenAI()
        self.retriever = Retriever(client=self.client)

    def ask(self, question: str) -> dict:
        """Answer a question using RAG.

        Returns a dict with keys:
            answer: The generated answer with inline citations
            sources: List of source dicts (title, url)
            chunks: The retrieved chunks (for debugging/evaluation)
        """
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        result = generate(question, chunks, client=self.client)
        result["chunks"] = chunks
        return result
