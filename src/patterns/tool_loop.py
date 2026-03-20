"""Pattern 1: Tool-Use Agent Loop.

ResearchAgent wires BaseAgent to the mock search/calculate/summarize tools.
"""

from __future__ import annotations

import anthropic

from src.agent import BaseAgent
from src.tools import TOOL_REGISTRY, TOOL_SCHEMAS

SYSTEM_PROMPT = (
    "You are a research assistant. Use the available tools to answer questions "
    "thoroughly. Search for relevant information, calculate if needed, and "
    "summarize findings before giving your final answer."
)


class ResearchAgent(BaseAgent):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_iterations: int = 10,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            tool_registry=TOOL_REGISTRY,
            model=model,
            max_iterations=max_iterations,
            client=client,
        )
