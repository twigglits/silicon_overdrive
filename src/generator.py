"""LLM generation with source grounding and citation."""

from openai import OpenAI

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Paul Graham's essays.

IMPORTANT RULES:
1. Answer ONLY based on the provided context passages. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided essays to answer this question."
3. Cite your sources by referencing the passage numbers in square brackets, e.g. [1], [2].
4. You may synthesize information across multiple passages, but every claim must be traceable to at least one passage.
5. Keep your answer concise and well-structured."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context blocks."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk["metadata"]["title"]
        blocks.append(f"[{i}] (From: \"{title}\")\n{chunk['text']}")
    return "\n\n".join(blocks)


def generate(
    query: str,
    retrieved_chunks: list[dict],
    client: OpenAI | None = None,
) -> dict:
    """Generate an answer grounded in the retrieved chunks.

    Returns a dict with keys: answer, sources.
    """
    client = client or OpenAI()

    context = format_context(retrieved_chunks)
    user_prompt = f"""Context passages:

{context}

Question: {query}

Answer the question using ONLY the context passages above. Cite passage numbers in [brackets]."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    # Build source list from retrieved chunks
    sources = []
    seen = set()
    for i, chunk in enumerate(retrieved_chunks, 1):
        key = chunk["metadata"]["url"]
        if key not in seen:
            seen.add(key)
            sources.append({
                "number": i,
                "title": chunk["metadata"]["title"],
                "url": chunk["metadata"]["url"],
            })

    return {"answer": answer, "sources": sources}
