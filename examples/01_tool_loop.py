"""Example 1: Tool-Use Agent Loop

Run: python examples/01_tool_loop.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patterns.tool_loop import ResearchAgent


async def main() -> None:
    agent = ResearchAgent()
    question = (
        "What is Anthropic and what is asyncio? Also, what is 17 * 42?"
    )
    print(f"Question: {question}\n")
    print("Running agent (may make multiple tool calls)...\n")

    answer = await agent.run(question)

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print()
    print(f"Total messages in history: {len(agent.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
