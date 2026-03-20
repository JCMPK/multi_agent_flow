"""Example 3: Evaluator-Optimizer Loop

Run: python examples/03_evaluator_optimizer.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patterns.evaluator_optimizer import run_essay_writer


async def main() -> None:
    topic = "The impact of asynchronous programming on modern software architecture"
    print(f"Topic: {topic}")
    print("Running evaluator-optimizer loop (score threshold: 8, max iterations: 3)...\n")

    result = await run_essay_writer(topic, score_threshold=8, max_iterations=3)

    print("=" * 60)
    print("ITERATION HISTORY")
    print("=" * 60)
    print(f"{'Iter':<6} {'Score':<7} Feedback")
    print("-" * 60)
    for entry in result["history"]:
        print(f"{entry['iteration']:<6} {entry['score']:<7} {entry['feedback']}")

    print(f"\nCompleted in {result['iterations']} iteration(s)")
    print(f"Best score: {result['score']}/10")
    print()
    print("=" * 60)
    print("FINAL ESSAY")
    print("=" * 60)
    print(result["essay"])


if __name__ == "__main__":
    asyncio.run(main())
