"""Pattern 2: Orchestrator + Parallel Subagents.

Three haiku subagents review code from different angles concurrently;
a sonnet orchestrator synthesises their findings into one report.
"""

from __future__ import annotations

import asyncio

import anthropic

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Subagent prompts
# ---------------------------------------------------------------------------

_STYLE_PROMPT = (
    "You are a code style reviewer. Examine the code for naming conventions, "
    "readability, formatting, and PEP 8 compliance. Be concise."
)

_SECURITY_PROMPT = (
    "You are a security reviewer. Examine the code for common vulnerabilities: "
    "injection, insecure defaults, missing input validation, exposed secrets. Be concise."
)

_LOGIC_PROMPT = (
    "You are a logic reviewer. Examine the code for algorithmic correctness, "
    "edge cases, off-by-one errors, and incorrect assumptions. Be concise."
)

_SYNTHESIS_SYSTEM = (
    "You are a senior engineering lead. You receive three code review reports "
    "(style, security, logic) and synthesise them into a single, prioritised "
    "review with an overall assessment."
)


# ---------------------------------------------------------------------------
# One-shot subagent helper
# ---------------------------------------------------------------------------

async def _review(
    client: anthropic.AsyncAnthropic,
    system: str,
    code: str,
    label: str,
) -> tuple[str, str]:
    """Run a single one-shot review call. Returns (label, review_text)."""
    response = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": f"Review this code:\n\n```\n{code}\n```"}],
    )
    text = "".join(
        getattr(block, "text", "") for block in response.content
    )
    return label, text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_code_review(
    code: str,
    client: anthropic.AsyncAnthropic | None = None,
) -> dict:
    """
    Orchestrate a parallel code review.

    Returns a dict with keys:
        style, security, logic  — individual subagent findings
        synthesis               — orchestrator's combined report
        model_calls             — total API calls made (always 4)
    """
    _client = client or anthropic.AsyncAnthropic()

    # Phase 1: fan out to three subagents concurrently.
    results = await asyncio.gather(
        _review(_client, _STYLE_PROMPT, code, "style"),
        _review(_client, _SECURITY_PROMPT, code, "security"),
        _review(_client, _LOGIC_PROMPT, code, "logic"),
    )
    findings = dict(results)  # {"style": "...", "security": "...", "logic": "..."}

    # Phase 2: synthesise.
    synthesis_prompt = (
        f"**Style review:**\n{findings['style']}\n\n"
        f"**Security review:**\n{findings['security']}\n\n"
        f"**Logic review:**\n{findings['logic']}"
    )
    synthesis_response = await _client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        system=_SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    synthesis_text = "".join(
        getattr(block, "text", "") for block in synthesis_response.content
    )

    return {
        **findings,
        "synthesis": synthesis_text,
        "model_calls": 4,
    }
