"""Tests for all three multi-agent patterns.

No real API calls are made — an AsyncAnthropic mock is injected via constructor.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake API response objects
# ---------------------------------------------------------------------------


def _text_block(text: str):
    """Minimal object that looks like a TextBlock."""
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_id: str, inputs: dict):
    """Minimal object that looks like a ToolUseBlock."""
    return SimpleNamespace(type="tool_use", name=name, id=tool_id, input=inputs)


def make_text_response(text: str):
    """Fake messages.create() response that ends a conversation."""
    return SimpleNamespace(
        content=[_text_block(text)],
        stop_reason="end_turn",
    )


def make_tool_use_response(name: str, tool_id: str, inputs: dict):
    """Fake messages.create() response that requests a tool call."""
    return SimpleNamespace(
        content=[_tool_use_block(name, tool_id, inputs)],
        stop_reason="tool_use",
    )


def _mock_client(*responses):
    """
    Build a mock AsyncAnthropic whose messages.create returns *responses*
    in sequence. Each element of *responses* may itself be a list of responses
    (used to simulate asyncio.gather).
    """
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=list(responses))
    return client


# ===========================================================================
# Pattern 1: Tool-Use Agent Loop
# ===========================================================================


class TestToolLoop:
    async def test_single_tool_call_then_end_turn(self):
        """Agent makes one tool call then returns text."""
        from src.patterns.tool_loop import ResearchAgent

        tool_resp = make_tool_use_response(
            "search", "tid_1", {"query": "anthropic"}
        )
        text_resp = make_text_response("Anthropic is an AI safety company.")

        agent = ResearchAgent(client=_mock_client(tool_resp, text_resp))
        result = await agent.run("Tell me about Anthropic.")

        assert "Anthropic" in result
        # messages: [user, assistant(tool_use), user(tool_result), assistant(text)]
        assert len(agent.messages) == 4

    async def test_tool_result_batched_in_single_user_message(self):
        """Multiple tool calls in one turn → batched into ONE user message."""
        from src.patterns.tool_loop import ResearchAgent

        # First response has two tool_use blocks.
        multi_tool_content = [
            _tool_use_block("search", "t1", {"query": "python"}),
            _tool_use_block("calculate", "t2", {"expression": "2+2"}),
        ]
        first_resp = SimpleNamespace(content=multi_tool_content, stop_reason="tool_use")
        second_resp = make_text_response("Done.")

        agent = ResearchAgent(client=_mock_client(first_resp, second_resp))
        await agent.run("Search python and calculate 2+2.")

        # The tool_result user message should contain both results.
        tool_result_msg = agent.messages[2]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert len(tool_result_msg["content"]) == 2

    async def test_max_iterations_raises(self):
        """RuntimeError when tool_use loops exceed max_iterations."""
        from src.patterns.tool_loop import ResearchAgent

        # Always returns tool_use → never reaches end_turn.
        infinite_tool = make_tool_use_response("search", "t1", {"query": "loop"})
        client = _mock_client(*([infinite_tool] * 5))

        agent = ResearchAgent(max_iterations=3, client=client)
        with pytest.raises(RuntimeError, match="max_iterations"):
            await agent.run("Loop forever.")

    async def test_unknown_tool_returns_error_string(self):
        """Calling an unregistered tool name returns an error, not an exception."""
        from src.agent import BaseAgent

        tool_resp = make_tool_use_response("nonexistent", "t1", {})
        text_resp = make_text_response("OK.")

        agent = BaseAgent(
            system="test",
            tools=[],
            tool_registry={},
            client=_mock_client(tool_resp, text_resp),
        )
        await agent.run("test")

        # The tool_result message content should contain the error string.
        tool_result_content = agent.messages[2]["content"]
        assert any("unknown tool" in item["content"] for item in tool_result_content)


# ===========================================================================
# Pattern 2: Orchestrator + Parallel Subagents
# ===========================================================================


class TestOrchestrator:
    async def test_four_api_calls_total(self):
        """Exactly 4 calls: 3 subagents + 1 synthesis."""
        from src.patterns.orchestrator import run_code_review

        subagent_resp = make_text_response("Looks fine.")
        synthesis_resp = make_text_response("Overall: good code.")

        client = _mock_client(
            subagent_resp,   # style
            subagent_resp,   # security
            subagent_resp,   # logic
            synthesis_resp,  # orchestrator
        )
        result = await run_code_review("def foo(): pass", client=client)

        assert client.messages.create.call_count == 4
        assert result["model_calls"] == 4

    async def test_result_keys_present(self):
        """Return dict contains all expected keys."""
        from src.patterns.orchestrator import run_code_review

        resp = make_text_response("Review text.")
        client = _mock_client(resp, resp, resp, resp)
        result = await run_code_review("x = 1", client=client)

        assert set(result.keys()) >= {"style", "security", "logic", "synthesis", "model_calls"}

    async def test_subagent_text_propagated(self):
        """Each subagent's text lands in the right key."""
        from src.patterns.orchestrator import run_code_review

        # asyncio.gather preserves order: style, security, logic
        style_resp = make_text_response("Style: ok")
        security_resp = make_text_response("Security: ok")
        logic_resp = make_text_response("Logic: ok")
        synthesis_resp = make_text_response("All good.")

        client = _mock_client(style_resp, security_resp, logic_resp, synthesis_resp)
        result = await run_code_review("pass", client=client)

        assert result["style"] == "Style: ok"
        assert result["security"] == "Security: ok"
        assert result["logic"] == "Logic: ok"
        assert result["synthesis"] == "All good."


# ===========================================================================
# Pattern 3: Evaluator-Optimizer Loop
# ===========================================================================


class TestEvaluatorOptimizer:
    @staticmethod
    def _eval_resp(score: int, feedback: str = "Good."):
        payload = json.dumps({"score": score, "feedback": feedback})
        return make_text_response(payload)

    async def test_early_exit_on_high_score(self):
        """Loop exits after iteration 1 when score >= threshold."""
        from src.patterns.evaluator_optimizer import run_essay_writer

        essay_resp = make_text_response("A great essay.")
        eval_resp = self._eval_resp(9, "Excellent!")

        client = _mock_client(essay_resp, eval_resp)
        result = await run_essay_writer(
            "test topic", score_threshold=8, max_iterations=3, client=client
        )

        assert result["iterations"] == 1
        assert result["score"] == 9
        assert client.messages.create.call_count == 2  # 1 generate + 1 evaluate

    async def test_full_three_iteration_run(self):
        """Runs all 3 iterations and returns best (not last) score."""
        from src.patterns.evaluator_optimizer import run_essay_writer

        essay_resp = make_text_response("An essay.")
        responses = [
            essay_resp, self._eval_resp(5, "Needs work."),
            essay_resp, self._eval_resp(7, "Better."),
            essay_resp, self._eval_resp(6, "Slightly worse."),
        ]
        client = _mock_client(*responses)
        result = await run_essay_writer(
            "test topic", score_threshold=8, max_iterations=3, client=client
        )

        assert result["iterations"] == 3
        assert result["score"] == 7  # best, not last (which was 6)
        assert len(result["history"]) == 3

    async def test_history_records_all_iterations(self):
        """history contains one entry per iteration with score and feedback."""
        from src.patterns.evaluator_optimizer import run_essay_writer

        essay_resp = make_text_response("Essay.")
        responses = [
            essay_resp, self._eval_resp(4, "Weak."),
            essay_resp, self._eval_resp(9, "Great!"),
        ]
        client = _mock_client(*responses)
        result = await run_essay_writer(
            "test topic", score_threshold=8, max_iterations=3, client=client
        )

        assert len(result["history"]) == 2
        assert result["history"][0] == {"iteration": 1, "score": 4, "feedback": "Weak."}
        assert result["history"][1] == {"iteration": 2, "score": 9, "feedback": "Great!"}

    async def test_feedback_passed_to_generator(self):
        """After a low score, the next generate call receives the feedback."""
        from src.patterns.evaluator_optimizer import run_essay_writer

        essay_resp = make_text_response("Essay draft.")
        responses = [
            essay_resp,
            self._eval_resp(5, "Add more examples."),
            essay_resp,
            self._eval_resp(9, "Much better!"),
        ]
        client = _mock_client(*responses)
        await run_essay_writer(
            "test topic", score_threshold=8, max_iterations=3, client=client
        )

        # The third call (second generate) should include the feedback in its prompt.
        third_call_kwargs = client.messages.create.call_args_list[2]
        messages_arg = third_call_kwargs.kwargs.get(
            "messages", third_call_kwargs.args[0] if third_call_kwargs.args else []
        )
        user_content = messages_arg[0]["content"] if messages_arg else ""
        assert "Add more examples" in user_content
