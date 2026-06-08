# Project Guide

## Purpose

This repository is a Week 4 learning project for building an LLM agent loop from
scratch with Python and the Anthropic API. The goal is to understand the
mechanics behind agent frameworks before using LangGraph in Week 5.

The finished Week 4 project should support:

- Multi-step tool use
- Multi-turn CLI conversations
- Tool validation and error recovery
- Tool retries with a maximum of three attempts
- A maximum agent-step limit
- Structured execution logs
- Pydantic models for internal agent data
- Async Anthropic API calls
- Token and cost tracking
- Focused pytest coverage

## Current State

The project currently has:

- Four tools: calculator, file reader, web search, and URL fetcher
- Pydantic input schemas
- A `Tool` wrapper and `ToolRegistry`
- An asynchronous multi-step `Agent` loop
- Multi-turn conversation history
- Structured tool-call, tool-result, and agent-step models
- Tool retries with exponential backoff
- A ten-step safety limit
- Structured execution logging
- Per-task token and estimated cost tracking
- Ruff, mypy, and pytest configuration
- Eight focused tests
- A thin CLI entry point

The Week 4 implementation is feature-complete. Remaining work is final
evaluation, documentation, and theory review.

## Structure

```text
main.py                 CLI entry point
agent/agent.py          Agent loop and conversation state
agent/setup.py          Tool construction and registration
agent/tool.py           Tool schema generation and execution
agent/tool_registry.py  Tool storage and dispatch
agent/tools.py          Tool implementations
agent/schemas.py        Pydantic models
agent/retry.py          Retry decorator
agent/token_tracker.py  Token and estimated cost tracking
tests/                  Automated tests
```

## Development Rules

- Keep changes minimal and scoped to the current learning task.
- Do not add abstractions until the current behavior creates a real need for
  them.
- Preserve existing working behavior unless the task explicitly changes it.
- Use the repository's existing patterns before introducing new ones.
- Keep all source code, comments, identifiers, logs, and user-facing CLI text
  in English.
- Use `snake_case` for Python variables, functions, methods, and modules.
- Use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.
- Add type annotations to all functions and methods.
- Use Pydantic for structured agent data required by the Week 4 plan.
- Use structured APIs and SDK types instead of ad hoc string parsing.
- Never commit `.env`, API keys, credentials, or other secrets.

## Learning Style

- When a task introduces a new concept, explain only the necessary foundation
  before implementation.
- Explain why the concept is needed in this project and show one small example.
- Avoid front-loading advanced details that belong to later tasks.
- For simple mechanical changes, implement them directly and verify them.
- Ask the learner to write code when the step teaches an important design or
  language concept.
- Continue explanations in Chinese unless the learner requests otherwise.
- Keep generated project code and documentation in English.

## Verification

After code changes, run the smallest relevant checks:

```bash
.venv/bin/python -m py_compile main.py agent/*.py
git diff --check
```

Run focused tests when they exist. Real Anthropic or web API calls should only
be used when local checks cannot verify the behavior.

## Git

- Use Conventional Commits:
  - `feat:` for new behavior
  - `fix:` for bug fixes
  - `refactor:` for behavior-preserving restructuring
  - `test:` for tests
  - `docs:` for documentation
  - `chore:` for tooling and maintenance
- Review the diff before committing.
- Keep unrelated changes out of the commit.
- Do not rewrite or discard user changes.

## Remaining Week 4 Work

The code implementation is closed unless final evaluation reveals a bug.

1. Run the final multi-tool arXiv research task.
2. Write the README with usage, architecture, limitations, and test commands.
3. Review the ReAct paper and Anthropic's agent-building guidance.
4. Prepare explanations of loop termination, recovery, Pydantic, async,
   decorators, context managers, and ReAct versus plain tool calling.

## Pricing Scope

Estimated cost currently supports Claude Haiku 4.5 standard input and output
tokens at $1/MTok and $5/MTok. Pricing was verified against the official
[Anthropic pricing documentation](https://platform.claude.com/docs/en/about-claude/pricing)
on 2026-06-08.

Prompt caching and server-side tool fees are not included because this project
does not enable those features. Update `MODEL_PRICING` before changing models.
