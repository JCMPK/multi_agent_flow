# Multi-Agent Flow

A toy educational repository demonstrating three reusable multi-agent patterns using the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

## Patterns

| # | Pattern | Where | Toy Task |
|---|---|---|---|
| 1 | **Tool-Use Agent Loop** | `src/patterns/tool_loop.py` | Research assistant with mock search/calculate/summarize tools |
| 2 | **Orchestrator + Parallel Subagents** | `src/patterns/orchestrator.py` | Code reviewer: 3 subagents run concurrently, orchestrator synthesises |
| 3 | **Evaluator-Optimizer Loop** | `src/patterns/evaluator_optimizer.py` | Essay writer: generator → evaluator scores → loop with feedback |

## Project Layout

```
multi_agent_flow/
├── src/
│   ├── agent.py                    # BaseAgent: tool-call loop, message history
│   ├── tools.py                    # Mock tools: search(), calculate(), summarize()
│   └── patterns/
│       ├── tool_loop.py            # Pattern 1
│       ├── orchestrator.py         # Pattern 2
│       └── evaluator_optimizer.py  # Pattern 3
├── examples/
│   ├── 01_tool_loop.py
│   ├── 02_orchestrator.py
│   └── 03_evaluator_optimizer.py
└── tests/
    └── test_patterns.py
```

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Run tests (no API key needed — all mocked)
pytest tests/ -v

# Run live examples
python examples/01_tool_loop.py
python examples/02_orchestrator.py
python examples/03_evaluator_optimizer.py
```

## Key Design Decisions

### BaseAgent (`src/agent.py`)
- Tool results for one assistant turn are batched into **one user message** — the API requires this.
- `max_iterations` hard cap (default 10) prevents infinite loops.
- Client is injected via constructor (`client=None` defaults to `AsyncAnthropic()`) — makes testing trivial.

### Orchestrator (`src/patterns/orchestrator.py`)
- Uses `asyncio.gather()` so all three subagents run concurrently.
- Wall-clock time ≈ `max(t_style, t_security, t_logic)` instead of their sum.
- Haiku for subagents (cheap, focused tasks), Sonnet for orchestrator (synthesis).

### Evaluator-Optimizer (`src/patterns/evaluator_optimizer.py`)
- Tracks `best_essay`/`best_score` across iterations to guard against score regression.
- Returns `history` list showing per-iteration scores — the key learning artifact.
- Three exit conditions: score ≥ threshold → max_iterations hit → return best seen.

## Models Used

| Role | Model |
|---|---|
| Subagents / evaluator | `claude-haiku-4-5-20251001` |
| Orchestrator / generator | `claude-sonnet-4-6` |
