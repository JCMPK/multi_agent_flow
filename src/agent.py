"""BaseAgent: a reusable tool-calling loop built on the Anthropic async API.

Design notes
------------
- `messages` is kept as instance state so callers can inspect history.
- Tool results for one assistant turn are batched into a *single* user message
  (the API requires this — sending multiple user messages in a row is rejected).
- `max_iterations` is a hard safety cap; raises RuntimeError when exceeded.
- The client is injected so tests can pass a mock without monkeypatching.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import anthropic


class BaseAgent:
    def __init__(
        self,
        system: str,
        tools: list[dict],
        tool_registry: dict[str, Callable],
        model: str = "claude-sonnet-4-6",
        max_iterations: int = 10,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.system = system
        self.tools = tools
        self.tool_registry = tool_registry
        self.model = model
        self.max_iterations = max_iterations
        self.client = client or anthropic.AsyncAnthropic()
        self.messages: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, user_message: str) -> str:
        """Append *user_message* and run the tool-call loop until end_turn."""
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
            )

            # Append the assistant turn exactly as returned (list of content blocks).
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return self._extract_text(response.content)

            if response.stop_reason == "tool_use":
                tool_results = await self._execute_tool_calls(response.content)
                # All results for this turn go into ONE user message (API requirement).
                self.messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — treat as terminal.
            return self._extract_text(response.content)

        raise RuntimeError(
            f"BaseAgent exceeded max_iterations={self.max_iterations} without end_turn."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_tool_calls(self, content: list) -> list[dict]:
        """Run all tool_use blocks in *content* and return tool_result blocks."""
        tasks = [
            self._call_tool(block)
            for block in content
            if getattr(block, "type", None) == "tool_use"
        ]
        return await asyncio.gather(*tasks)

    async def _call_tool(self, block) -> dict:
        name: str = block.name
        inputs: dict = block.input
        handler = self.tool_registry.get(name)
        if handler is None:
            result = f"Error: unknown tool '{name}'"
        else:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**inputs)
                else:
                    result = handler(**inputs)
            except Exception as exc:
                result = f"Error executing {name}: {exc}"
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(result),
        }

    @staticmethod
    def _extract_text(content: list) -> str:
        """Return the concatenated text from all TextBlock items in *content*."""
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
