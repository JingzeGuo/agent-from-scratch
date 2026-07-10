# Project Guide

## Purpose

This repository contains a local terminal coding agent built with Python,
Pydantic, and LLM tool calling. The agent operates inside the current working
directory, uses structured tools to inspect and modify files, can run bounded
verification commands, and records session state for resumable workflows.

The project is intended to be a practical, maintainable coding-agent
implementation with explicit controller behavior, provider adapters, tool
validation, session persistence, context management, memory, observability, and
cost tracking.

## Capabilities

The agent supports:

- Multi-step tool use with a maximum step limit
- Multi-turn CLI conversations
- Anthropic, DeepSeek, and OpenAI-compatible provider adapters
- Pydantic schemas for tool inputs and internal agent data
- Tool validation and model-visible error observations
- Transient tool retries with exponential backoff
- Workspace-confined file reads, searches, edits, writes, diffs, and commands
- Command safety policy with approval flow for broad commands
- Read-only sub-agent delegation for focused repository exploration
- Context compaction with structured checkpoints
- Session checkpoints, resume, rename, and JSONL trace events
- Project and global memory retrieval plus run reflection
- Token and estimated cost tracking
- Focused pytest coverage, Ruff, and mypy checks

## Structure

```text
main.py                          CLI parsing, startup wiring, provider setup
agent/cli_commands.py            Slash commands, checkpoint, memory, trace commands
agent/agent.py                   Agent controller, run loop, scheduling, termination
agent/provider.py                Anthropic and OpenAI-compatible provider adapters
agent/setup.py                   Built-in tool registry construction
agent/tool.py                    Tool schema generation, validation, execution, retry
agent/tool_registry.py           Tool storage, dispatch, file tracking, diffs
agent/tools.py                   Tool implementations
agent/schemas.py                 Pydantic models for tools, runs, sessions, traces
agent/context.py                 Context compaction and checkpoint construction
agent/session.py                 Session snapshots, pending actions, JSONL events
agent/memory.py                  Project/global memory stores, retrieval, reflection
agent/security.py                Command policy and trace redaction
agent/token_tracker.py           Token and estimated cost tracking
agent/verification.py            Verification evidence extraction
agent/workspace.py               Workspace path normalization and escape rejection
scripts/                         Deterministic evaluation scripts
tests/                           Automated tests
```

## Development Rules

- Keep changes minimal and scoped to the requested behavior.
- Preserve existing working behavior unless the task explicitly changes it.
- Use the repository's existing patterns before introducing new abstractions.
- Add abstractions only when they remove real duplication or clarify ownership.
- Keep all source code, comments, identifiers, logs, and user-facing CLI text in
  English.
- Use `snake_case` for Python variables, functions, methods, and modules.
- Use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.
- Add type annotations to all functions and methods.
- Use Pydantic for structured agent data and tool input schemas.
- Use structured APIs and SDK types instead of ad hoc string parsing.
- Keep generated output bounded before returning it to the model.
- Never commit `.env`, API keys, credentials, access tokens, or other secrets.

## Agent Design Rules

- Keep the core controller loop in `agent/agent.py` explicit and easy to audit.
- Treat tools as the action space and tool results as observations.
- Return validation and execution failures to the model as recoverable
  observations when possible.
- Preserve workspace confinement for all file and command operations.
- Keep `sub_agent` read-only and bounded; it must not edit files, run commands,
  use network tools, or recursively spawn another sub-agent.
- Keep command execution policy in controller-level code rather than relying on
  shell behavior.
- Keep provider-specific message conversion in provider adapters.
- Keep CLI command handling outside the core agent controller.
- Do not add live API calls to tests when a fake provider or deterministic
  script can verify the behavior.

## Verification

After code changes, run the smallest relevant checks:

```bash
.venv/bin/python -m py_compile main.py agent/*.py scripts/*.py
git diff --check
```

Run focused tests for touched behavior. Common focused commands:

```bash
.venv/bin/python -m pytest tests/test_agent.py -q
.venv/bin/python -m pytest tests/test_main.py -q
.venv/bin/python -m pytest tests/test_tools.py -q
```

Before committing broad changes, run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/mypy .
```

Real provider and web API calls should only be used when local checks cannot
verify the behavior.

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
- Do not record or commit files under `docs/`; the repository intentionally
  keeps `docs/` ignored.

## Pricing Scope

Estimated cost currently supports Claude Haiku 4.5 standard input and output
tokens at $1/MTok and $5/MTok. Pricing was verified against the official
[Anthropic pricing documentation](https://platform.claude.com/docs/en/about-claude/pricing)
on 2026-06-08.

Prompt caching and server-side tool fees are not included because this project
does not enable those features. Update `MODEL_PRICING` before changing models.
