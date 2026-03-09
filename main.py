"""CLI entry point for the RAG-powered Q&A system."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def interactive_mode():
    """Run interactive Q&A loop."""
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    print("RAG Q&A over Paul Graham's Essays")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = pipeline.ask(question)

        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for src in result["sources"]:
            print(f"  [{src['number']}] {src['title']} — {src['url']}")
        print()


def single_query(query: str):
    """Answer a single question and exit."""
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.ask(query)

    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  [{src['number']}] {src['title']} — {src['url']}")


def run_eval():
    """Run the evaluation suite."""
    from src.evaluate import run_evaluation

    run_evaluation()


def main():
    parser = argparse.ArgumentParser(
        description="RAG-powered Q&A over Paul Graham's essays"
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Ask a single question"
    )
    parser.add_argument(
        "--evaluate", "-e", action="store_true", help="Run evaluation suite"
    )

    args = parser.parse_args()

    if args.evaluate:
        run_eval()
    elif args.query:
        single_query(args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
