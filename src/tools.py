"""Mock tools for the research agent demo.

Each function is synchronous and returns a plain string — BaseAgent
wraps the return value in the required tool_result block format.
"""

import json
import math


def search(query: str) -> str:
    """Simulated web search — returns canned snippets keyed on keywords."""
    db = {
        "python": "Python is a high-level, interpreted programming language known for readability.",
        "asyncio": "asyncio is Python's built-in library for writing concurrent code using async/await.",
        "anthropic": "Anthropic is an AI safety company that created the Claude family of models.",
        "claude": "Claude is Anthropic's AI assistant, available via API with tool-use capabilities.",
        "agent": "An AI agent is a system that perceives its environment and takes actions autonomously.",
    }
    query_lower = query.lower()
    for keyword, snippet in db.items():
        if keyword in query_lower:
            return snippet
    return f"No results found for '{query}'."


def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    # Only allow digits, operators, spaces, parentheses, and dots.
    allowed = set("0123456789+-*/.() ")
    if not set(expression).issubset(allowed):
        return f"Error: expression contains disallowed characters: {expression!r}"
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def summarize(text: str) -> str:
    """Return a mock summary (first sentence + word count)."""
    first_sentence = text.split(".")[0].strip()
    word_count = len(text.split())
    return f"Summary ({word_count} words): {first_sentence}."


# Registry used by BaseAgent to dispatch tool calls by name.
TOOL_REGISTRY: dict[str, callable] = {
    "search": search,
    "calculate": calculate,
    "summarize": summarize,
}

# JSON schema definitions sent to the Anthropic API.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search",
        "description": "Search the web for information about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a simple arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression using +, -, *, /, ().",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "summarize",
        "description": "Summarize a piece of text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize."},
            },
            "required": ["text"],
        },
    },
]
