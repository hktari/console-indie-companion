#!/usr/bin/env python3
"""
RAG Quality Test Suite for Tunic Knowledge Base

Tests the RAG pipeline against 10 common Tunic player questions.
Evaluates whether retrieved results contain relevant content.

Usage:
    python tests/test_rag_quality.py          # Run standalone
    pytest tests/test_rag_quality.py -v       # Run via pytest
"""

import sys
from pathlib import Path
from typing import NamedTuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import query_tunic_knowledge


class QueryTest(NamedTuple):
    """A test query with expected relevance keywords."""

    question: str
    keywords: list[str]
    topic: str


# 10 Common Tunic Player Questions
TEST_QUERIES = [
    QueryTest(
        question="How do I beat the Garden Knight boss?",
        keywords=["garden knight", "boss", "guardian", "defeat", "fight"],
        topic="Boss Strategy",
    ),
    QueryTest(
        question="Where do I find the shield?",
        keywords=["shield", "defense", "block", "item", "location"],
        topic="Item Location",
    ),
    QueryTest(
        question="What is the instruction manual in Tunic?",
        keywords=["manual", "booklet", "instruction", "page", "guide"],
        topic="Game Mechanic",
    ),
    QueryTest(
        question="How do I open the sealed door in the Eastern Vault?",
        keywords=["eastern", "vault", "sealed", "door", "key", "lock"],
        topic="Puzzle Solution",
    ),
    QueryTest(
        question="What are the golden coins used for?",
        keywords=["coin", "gold", "currency", "money", "shop", "trade"],
        topic="Item Use",
    ),
    QueryTest(
        question="How do I get to the West Garden?",
        keywords=["west garden", "garden", "path", "access", "route"],
        topic="Navigation",
    ),
    QueryTest(
        question="What happens when you ring all three bells?",
        keywords=["bell", "ring", "three", "progression", "ending"],
        topic="Progression",
    ),
    QueryTest(
        question="Where is the hero's grave?",
        keywords=["hero", "grave", "graveyard", "location", "secret"],
        topic="Secret Location",
    ),
    QueryTest(
        question="How does the magic system work?",
        keywords=["magic", "spell", "mp", "mana", "combat", "ability"],
        topic="Combat Mechanic",
    ),
    QueryTest(
        question="What is the Far Shore?",
        keywords=["far shore", "shore", "ending", "lore", "story"],
        topic="Lore",
    ),
]


def check_relevance(results: list[str], keywords: list[str]) -> bool:
    """
    Check if any result contains at least one relevance keyword.

    Args:
        results: List of result chunks from RAG query
        keywords: List of keywords to search for (case-insensitive)

    Returns:
        True if any keyword found in any result, False otherwise
    """
    if not results:
        return False

    # Combine all results into one searchable text
    combined_text = " ".join(results).lower()

    # Check if any keyword appears in the results
    for keyword in keywords:
        if keyword.lower() in combined_text:
            return True

    return False


def find_matching_keyword(results: list[str], keywords: list[str]) -> str:
    """Find the first matching keyword in results."""
    if not results:
        return ""

    combined_text = " ".join(results).lower()
    for keyword in keywords:
        if keyword.lower() in combined_text:
            return keyword

    return ""


def run_quality_test() -> tuple[int, list[dict]]:
    """
    Run all 10 test queries and evaluate relevance.

    Returns:
        Tuple of (score, detailed_results)
        - score: Number of queries that returned relevant results (0-10)
        - detailed_results: List of dicts with query details and results
    """
    score = 0
    detailed_results = []

    print("\n" + "=" * 80)
    print("RAG QUALITY TEST - Tunic Knowledge Base")
    print("=" * 80)
    print(f"\nRunning {len(TEST_QUERIES)} test queries...\n")

    for i, test_query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {test_query.topic}: {test_query.question}")

        try:
            # Query the RAG pipeline
            results = query_tunic_knowledge(test_query.question, n_results=5)

            # Check relevance
            is_relevant = check_relevance(results, test_query.keywords)
            matching_keyword = find_matching_keyword(results, test_query.keywords)

            if is_relevant:
                score += 1
                status = "✓ PASS"
                print(f"  {status} - Found keyword: '{matching_keyword}'")
            else:
                status = "✗ FAIL"
                print(f"  {status} - No relevant keywords found")

            # Store detailed result
            detailed_results.append(
                {
                    "number": i,
                    "topic": test_query.topic,
                    "question": test_query.question,
                    "keywords": test_query.keywords,
                    "status": "PASS" if is_relevant else "FAIL",
                    "matching_keyword": matching_keyword,
                    "results": results,
                    "result_count": len(results),
                }
            )

        except Exception as e:
            print(f"  ✗ ERROR - {str(e)}")
            detailed_results.append(
                {
                    "number": i,
                    "topic": test_query.topic,
                    "question": test_query.question,
                    "keywords": test_query.keywords,
                    "status": "ERROR",
                    "matching_keyword": "",
                    "results": [],
                    "result_count": 0,
                    "error": str(e),
                }
            )

    return score, detailed_results


def print_summary_table(score: int, detailed_results: list[dict]):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    for result in detailed_results:
        status_symbol = "✓" if result["status"] == "PASS" else "✗"
        keyword_info = (
            f" → Found: '{result['matching_keyword']}'"
            if result["matching_keyword"]
            else ""
        )

        print(f"\nQ{result['number']}: {status_symbol} {result['status']}")
        print(f"  Topic: {result['topic']}")
        print(f"  Question: {result['question']}")
        print(f"  Results: {result['result_count']} chunks retrieved{keyword_info}")

    # Final score
    percentage = (score / len(TEST_QUERIES)) * 100
    threshold = 7
    overall_status = "PASS" if score >= threshold else "FAIL"

    print("\n" + "=" * 80)
    print(f"FINAL SCORE: {score}/{len(TEST_QUERIES)} ({percentage:.0f}%)")
    print(f"THRESHOLD: {threshold}/{len(TEST_QUERIES)} (70%)")
    print(f"RESULT: {overall_status}")
    print("=" * 80 + "\n")

    return score >= threshold


def save_evidence_report(score: int, detailed_results: list[dict]):
    """Save detailed report to evidence file."""
    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report_file = evidence_dir / "task-7-rag-quality.txt"

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RAG QUALITY TEST REPORT - Tunic Knowledge Base\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        percentage = (score / len(TEST_QUERIES)) * 100
        f.write(f"FINAL SCORE: {score}/{len(TEST_QUERIES)} ({percentage:.0f}%)\n")
        f.write(f"THRESHOLD: 7/10 (70%)\n")
        f.write(f"STATUS: {'PASS' if score >= 7 else 'FAIL'}\n\n")

        # Detailed results
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for result in detailed_results:
            f.write(f"Q{result['number']}: {result['status']}\n")
            f.write(f"Topic: {result['topic']}\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Expected Keywords: {', '.join(result['keywords'])}\n")

            if result["status"] == "PASS":
                f.write(f"Matching Keyword: {result['matching_keyword']}\n")
            elif result["status"] == "ERROR":
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")

            f.write(f"Results Retrieved: {result['result_count']}\n")

            if result["results"]:
                f.write("\nRetrieved Content:\n")
                f.write("-" * 80 + "\n")
                for i, res in enumerate(result["results"], 1):
                    f.write(f"\n[Result {i}]\n{res}\n")
                f.write("-" * 80 + "\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"✓ Evidence report saved to: {report_file}")


def test_rag_quality_threshold():
    """
    Pytest test function: At least 7/10 queries must return relevant results.
    """
    score, detailed_results = run_quality_test()

    # Print summary
    passed = print_summary_table(score, detailed_results)

    # Save evidence
    save_evidence_report(score, detailed_results)

    # Assert threshold
    assert score >= 7, (
        f"RAG quality test failed: {score}/10 queries returned relevant results. "
        f"Expected at least 7/10 (70%)."
    )


def main():
    """Standalone execution."""
    try:
        score, detailed_results = run_quality_test()
        passed = print_summary_table(score, detailed_results)
        save_evidence_report(score, detailed_results)

        # Exit with appropriate code
        sys.exit(0 if passed else 1)

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
