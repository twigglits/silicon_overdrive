"""Faithfulness evaluation for the RAG pipeline."""

import json
from pathlib import Path

from openai import OpenAI

from .generator import LLM_MODEL
from .pipeline import RAGPipeline

EVAL_DIR = Path(__file__).parent.parent / "eval"

FAITHFULNESS_PROMPT = """You are an evaluation judge. Your job is to assess whether a generated answer is faithful to the provided context passages.

Context passages:
{context}

Generated answer:
{answer}

Score the faithfulness of the answer on a scale from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context passages.
- 0.5: Some claims are supported, but others are not found in the context or are embellished.
- 0.0: The answer contains claims that contradict or are entirely unsupported by the context.

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reasoning": "<brief explanation>"}}"""


def evaluate_faithfulness(
    answer: str, chunks: list[dict], client: OpenAI
) -> dict:
    """Judge whether the answer is faithful to the retrieved chunks."""
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks)
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": FAITHFULNESS_PROMPT.format(
                    context=context, answer=answer
                ),
            }
        ],
        temperature=0.0,
        max_tokens=256,
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {"score": float(result["score"]), "reasoning": result["reasoning"]}
    except (json.JSONDecodeError, KeyError):
        return {"score": 0.0, "reasoning": "Failed to parse evaluation response"}


def run_evaluation() -> list[dict]:
    """Run the full evaluation suite and print results."""
    with open(EVAL_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    client = OpenAI()
    pipeline = RAGPipeline(client=client)

    results = []
    print(f"Running evaluation on {len(qa_pairs)} questions...\n")

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = qa["expected_answer"]

        print(f"[{i+1}/{len(qa_pairs)}] {question}")
        result = pipeline.ask(question)
        answer = result["answer"]
        chunks = result["chunks"]

        faithfulness = evaluate_faithfulness(answer, chunks, client)

        results.append({
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "faithfulness_score": faithfulness["score"],
            "faithfulness_reasoning": faithfulness["reasoning"],
            "num_sources": len(result["sources"]),
        })

        print(f"  Faithfulness: {faithfulness['score']:.2f}")
        print(f"  Reasoning: {faithfulness['reasoning']}\n")

    # Print results table
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Question':<60} {'Faithfulness':>12} {'Sources':>8}")
    print("-" * 100)
    for r in results:
        q = r["question"][:57] + "..." if len(r["question"]) > 60 else r["question"]
        print(f"{q:<60} {r['faithfulness_score']:>12.2f} {r['num_sources']:>8}")

    avg_score = sum(r["faithfulness_score"] for r in results) / len(results)
    print("-" * 100)
    print(f"{'Average':<60} {avg_score:>12.2f}")
    print("=" * 100)

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_evaluation()
