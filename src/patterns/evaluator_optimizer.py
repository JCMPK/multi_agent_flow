"""Pattern 3: Evaluator-Optimizer Loop.

An essay generator produces a draft; an evaluator scores it 1-10;
if score < threshold the generator receives the feedback and tries again.
Loop exits on: score >= threshold | max_iterations | returns best seen.
"""

from __future__ import annotations

import re

import anthropic

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

_GENERATOR_SYSTEM = (
    "You are an expert essay writer. Write clear, well-structured, engaging essays. "
    "When given feedback, incorporate it to improve your next draft."
)

_EVALUATOR_SYSTEM = (
    "You are a strict essay evaluator. Score essays on a scale of 1 to 10 based on "
    "clarity, structure, argumentation, and engagement. "
    "Respond ONLY with JSON in this format: "
    '{"score": <integer 1-10>, "feedback": "<one sentence of the most important improvement>"}'
)


# ---------------------------------------------------------------------------
# Internal one-shot helpers
# ---------------------------------------------------------------------------

async def _generate(
    client: anthropic.AsyncAnthropic,
    topic: str,
    feedback: str | None,
) -> str:
    user_msg = f"Write a short essay (3-4 paragraphs) on: {topic}"
    if feedback:
        user_msg += f"\n\nPrevious feedback to address: {feedback}"
    response = await client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        system=_GENERATOR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(getattr(b, "text", "") for b in response.content)


async def _evaluate(
    client: anthropic.AsyncAnthropic,
    essay: str,
) -> tuple[int, str]:
    response = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=256,
        system=_EVALUATOR_SYSTEM,
        messages=[{"role": "user", "content": f"Essay:\n\n{essay}"}],
    )
    raw = "".join(getattr(b, "text", "") for b in response.content)

    # Parse the JSON the evaluator was asked to return.
    try:
        import json
        data = json.loads(raw)
        score = int(data["score"])
        feedback = str(data["feedback"])
    except Exception:
        # Fallback: extract first integer found.
        match = re.search(r"\b([1-9]|10)\b", raw)
        score = int(match.group()) if match else 5
        feedback = raw.strip()

    return score, feedback


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_essay_writer(
    topic: str,
    score_threshold: int = 8,
    max_iterations: int = 3,
    client: anthropic.AsyncAnthropic | None = None,
) -> dict:
    """
    Run the generator-evaluator loop.

    Returns:
        essay        — best essay produced
        score        — its score
        iterations   — how many loops ran
        history      — list of {"iteration": n, "score": s, "feedback": f}
    """
    _client = client or anthropic.AsyncAnthropic()

    best_essay: str = ""
    best_score: int = 0
    feedback: str | None = None
    history: list[dict] = []

    for i in range(1, max_iterations + 1):
        essay = await _generate(_client, topic, feedback)
        score, feedback = await _evaluate(_client, essay)

        history.append({"iteration": i, "score": score, "feedback": feedback})

        if score > best_score:
            best_score = score
            best_essay = essay

        if score >= score_threshold:
            return {
                "essay": best_essay,
                "score": best_score,
                "iterations": i,
                "history": history,
            }

    # max_iterations reached — return the best we found.
    return {
        "essay": best_essay,
        "score": best_score,
        "iterations": max_iterations,
        "history": history,
    }
