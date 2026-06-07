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
- A synchronous multi-step `Agent` loop
- Multi-turn conversation history
- Tool-call and tool-result models
- A ten-step safety limit
- Basic execution logging
- A thin CLI entry point

The current learning task is to finish Pydantic-based internal structures,
starting with `ToolCall`, `ToolResult`, and then `AgentStep`.

## Structure

```text
main.py                 CLI entry point
agent/agent.py          Agent loop and conversation state
agent/setup.py          Tool construction and registration
agent/tool.py           Tool schema generation and execution
agent/tool_registry.py  Tool storage and dispatch
agent/tools.py          Tool implementations
agent/schemas.py        Pydantic models
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

Complete these tasks in dependency order:

1. Finish Pydantic models for internal agent data.
2. Add retry behavior through a `@retry` decorator.
3. Add a `TokenTracker` context manager.
4. Convert the Agent loop to `AsyncAnthropic`.
5. Configure Ruff, mypy, and pytest in `pyproject.toml`.
6. Add at least five focused tests.
7. Run the final multi-tool arXiv research task.
8. Document usage, architecture, limitations, and learning outcomes.
