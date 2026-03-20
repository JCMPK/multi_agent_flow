"""Example 2: Orchestrator + Parallel Subagents

Run: python examples/02_orchestrator.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patterns.orchestrator import run_code_review

SAMPLE_CODE = '''
import os

def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    password = "hardcoded_secret_123"
    result = db.execute(query)
    return result[0]

def process_items(items):
    total = 0
    for i in range(len(items) + 1):  # off-by-one?
        total += items[i]
    return total
'''


async def main() -> None:
    print("Code under review:")
    print(SAMPLE_CODE)
    print("Running 3 subagents in parallel (style, security, logic)...\n")

    result = await run_code_review(SAMPLE_CODE)

    print("=" * 60)
    print("STYLE REVIEW")
    print("=" * 60)
    print(result["style"])

    print("\n" + "=" * 60)
    print("SECURITY REVIEW")
    print("=" * 60)
    print(result["security"])

    print("\n" + "=" * 60)
    print("LOGIC REVIEW")
    print("=" * 60)
    print(result["logic"])

    print("\n" + "=" * 60)
    print("SYNTHESIS (orchestrator)")
    print("=" * 60)
    print(result["synthesis"])
    print(f"\nTotal API calls: {result['model_calls']}")


if __name__ == "__main__":
    asyncio.run(main())
