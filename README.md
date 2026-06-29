# agent-from-scratch

A small terminal coding agent built from scratch with Python, Pydantic, and
LLM tool calling.

This project is both a usable command-line agent and a white-box learning
project for understanding how agent frameworks work internally. It keeps the
main agent concepts explicit: state, policy, action space, observations,
controller loop, termination, recovery, context, persistence, and evaluation.

## Features

- Streaming terminal conversation
- One-shot task mode
- Anthropic, DeepSeek, and OpenAI-compatible provider adapters
- Multi-step tool use with a maximum step limit
- Multi-turn conversation state
- Pydantic schemas for tool inputs and internal run data
- Tool validation errors returned as model observations
- Transient tool retries with exponential backoff
- Workspace-confined file and command tools
- File reading, globbing, regex search, exact edit, file write, and diff tools
- Bounded command execution with a command safety policy
- Session checkpoints, listing, resume, and rename
- Context compaction reporting
- Structured JSONL trace events with secret redaction
- Token and estimated cost tracking
- Optional web search and URL fetching
- Read-only sub-agent delegation for bounded repository exploration

## Requirements

- Python 3.12 or newer
- An API key for one supported model provider
- Optional: `TAVILY_API_KEY` for web search

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd agent-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

After installation, the console script is available as:

```bash
agent --help
```

If the virtual environment is not activated, run the installed script directly:

```bash
.venv/bin/agent
```

For app-style local installation, use `pipx` from the project root:

```bash
pipx install -e .
```

That exposes `agent` on your shell `PATH` while keeping the app in an isolated
Python environment.

## Configuration

Copy `.env.example` to `.env` and fill in real values, or export the variables
in your shell.

```bash
cp .env.example .env
```

Supported variables:

| Variable | Description |
| --- | --- |
| `AGENT_PROVIDER` | Provider to use: `anthropic`, `deepseek`, or `openai` |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model, default `claude-haiku-4-5` |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_MODEL` | DeepSeek model, default `deepseek-v4-flash` |
| `DEEPSEEK_BASE_URL` | DeepSeek Chat Completions base URL, default `https://api.deepseek.com` |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | OpenAI model, default `gpt-4o-mini` |
| `OPENAI_BASE_URL` | OpenAI API base URL, default `https://api.openai.com/v1` |
| `TAVILY_API_KEY` | Optional key for the web search tool |
| `AGENT_TRACE_REDACT_PATTERNS` | Optional newline-separated regex patterns redacted from trace text |

The CLI loads `.env` from the directory where you start the process. For a
coding task in another repository, either export environment variables globally
or place that repository's agent configuration in its own `.env` file.

Never commit `.env`, API keys, access tokens, or credentials.

## Quick Start

Start an interactive session in the repository you want the agent to work on:

```bash
cd /path/to/target-repository
agent
```

The current working directory becomes the agent workspace. File and command
tools are confined to that workspace.

Run one task and exit:

```bash
agent "Read the tests and summarize what behavior they cover."
```

Resume a saved session by id or session name:

```bash
agent --resume session-20260629-104200-123456
agent --resume docs-review
```

Pass an API key directly when needed:

```bash
agent --api-key "$ANTHROPIC_API_KEY"
```

Show the installed version without loading provider configuration:

```bash
agent --version
```

## CLI Commands

Interactive sessions support slash commands:

| Command | Description |
| --- | --- |
| `/help` | Show available commands |
| `/model` | Show the current provider and model |
| `/model <provider> [model]` | Switch provider and optionally model |
| `/tokens` | Show input tokens, output tokens, total tokens, and estimated cost |
| `/status` | Show session, workspace, provider, model, files, and token state |
| `/reset` | Clear the current conversation context |
| `/save` | Save a session checkpoint |
| `/diff` | Show all file changes from this session |
| `/diff <path>` | Show changes for one file |
| `/compact` | Show context compaction metrics |
| `/trace` | Print structured JSONL trace events |
| `/trace <path>` | Export trace events to a workspace-relative file |
| `/rename <session-name>` | Rename the current session |
| `/sessions` | List saved sessions |
| `/exit` | Exit the application |

## Tools

The default registry exposes these tools to the model:

| Tool | Kind | Purpose |
| --- | --- | --- |
| `calculator` | pure | Safely evaluate a mathematical expression |
| `read_file` | read-only | Read a workspace file with line offset and limit |
| `glob_files` | read-only | Find workspace files with a glob pattern |
| `search_text` | read-only | Search workspace files with a Python regular expression |
| `edit_file` | write | Replace one exact text match and return a unified diff |
| `write_file` | write | Create or intentionally overwrite a file and return a unified diff |
| `get_diff` | read-only | Return diffs for files changed in the session |
| `run_command` | command | Run a bounded command inside the workspace |
| `sub_agent` | delegated | Delegate bounded read-only repository exploration |
| `fetch_url` | network | Fetch URL content |
| `search_web` | network | Search the web when `TAVILY_API_KEY` is configured |

All tool inputs are validated with Pydantic before execution. Validation
failures are returned to the model as error observations so the agent can
recover by choosing corrected arguments.

## Python API

The current library API is intentionally small and close to the learning
project internals.

Create a registry with the built-in tools:

```python
from pathlib import Path

from agent.setup import create_registry

registry = create_registry(Path.cwd())
```

Register a custom tool:

```python
from pydantic import BaseModel, Field

from agent.tool import Tool
from agent.tool_registry import ToolRegistry


class EchoInput(BaseModel):
    text: str = Field(description="Text to echo.")


def echo(text: str) -> str:
    return text


registry = ToolRegistry()
registry.register(
    Tool(
        name="echo",
        description="Return the provided text.",
        input_schema=EchoInput,
        fn=echo,
        kind="pure",
    )
)
```

Custom tools should keep inputs structured, return bounded text observations,
and use the narrowest accurate `kind`: `pure`, `read_only`, `write`, `command`,
`network`, or `delegated`.

## Architecture

High-level flow:

```text
CLI
  -> Agent controller
       -> Provider adapter
       -> Context builder
       -> Tool registry
            -> repository tools
            -> edit/write tools
            -> command tool
            -> web tools
            -> sub-agent tool
       -> Session store
       -> Trace writer
       -> Token and cost tracker
```

Core modules:

| Path | Responsibility |
| --- | --- |
| `main.py` | CLI parsing, interactive loop, slash commands, session wiring |
| `agent/agent.py` | Agent controller, run loop, tool scheduling, recovery, termination |
| `agent/provider.py` | Anthropic and OpenAI-compatible provider adapters |
| `agent/setup.py` | Built-in tool registry construction |
| `agent/tool.py` | Tool wrapper, schema conversion, validation, retry |
| `agent/tool_registry.py` | Tool storage, dispatch, changed-file tracking, diffs |
| `agent/tools.py` | Tool implementations |
| `agent/schemas.py` | Pydantic models for tools, runs, sessions, traces, context |
| `agent/workspace.py` | Workspace path normalization and escape rejection |
| `agent/security.py` | Command policy and trace redaction |
| `agent/session.py` | Session snapshots, pending actions, JSONL trace events |
| `agent/context.py` | Context compaction and checkpoint construction |
| `agent/token_tracker.py` | Token and estimated cost tracking |
| `agent/verification.py` | Verification evidence extraction |

## Sessions, Checkpoints, and Traces

Session data is stored inside the active workspace:

```text
.agents/sessions/
  <session-id>.json
  pending/<session-id>.json
  events/<session-id>.jsonl
```

The CLI automatically checkpoints after each completed task. Use `/save` to
checkpoint manually and `/sessions` to list saved sessions.

Trace events are append-only JSONL facts such as session start, model request,
tool start, tool finish, run finish, checkpoint save, and compaction report.
Use `/trace` to inspect them or `/trace traces/current.jsonl` to export them
inside the workspace.

Trace text is redacted for common secret-like patterns. Add
`AGENT_TRACE_REDACT_PATTERNS` for project-specific redaction.

## Safety Model

The agent is designed for local coding tasks, so safety is part of the product
contract.

- The default workspace is `Path.cwd()` when the CLI starts.
- File paths are resolved through the workspace resolver.
- `..` traversal and symlink escapes outside the workspace are rejected.
- File write tools track changed files and keep original file contents for
  session diffs.
- Commands run inside the workspace with timeout and output limits.
- Shell operators and command substitution are blocked by policy.
- Dangerous commands such as `rm`, `sudo`, `chmod`, `chown`, `dd`, `mount`,
  `shutdown`, and `git reset --hard` are blocked.
- Broad commands such as `git`, `pip`, `python`, `uv`, `curl`, and `wget`
  require approval unless they match the safe command policy.
- One-shot mode denies command approvals automatically.

This is a controller-level safety policy, not an operating-system sandbox.
Run the agent only in repositories where you are comfortable reviewing and
approving local changes.

## Development Checks

Run the smallest checks after documentation-only changes:

```bash
git diff --check
```

Run local static checks:

```bash
.venv/bin/python -m py_compile main.py agent/*.py scripts/*.py
.venv/bin/ruff check .
.venv/bin/mypy .
```

Run tests:

```bash
.venv/bin/pytest -q
```

Run deterministic evaluations:

```bash
.venv/bin/python scripts/evaluate_tool_selection.py
.venv/bin/python scripts/evaluate_coding_tasks.py
```

Real API calls should be reserved for behavior experiments that cannot be
verified with local fake providers or deterministic tests.

## Packaging Checks

Development install:

```bash
pip install -e .
agent --help
```

Build a wheel and source distribution:

```bash
python -m pip install build
python -m build
```

Install the app from a local checkout with `pipx`:

```bash
pipx install -e .
agent --help
```

If `pipx` was not previously configured, run:

```bash
pipx ensurepath
```

Then restart the shell.

## Release Checklist

- Package name, version, and console script are final for the release.
- `pip install -e .` exposes `agent`.
- `pipx install -e .` exposes `agent` outside the source checkout.
- `agent --help` and `agent --version` work without provider configuration.
- Help output works from a different working directory.
- The active workspace is the directory where `agent` is started.
- README covers installation, configuration, CLI commands, tools, sessions,
  traces, safety, development checks, and limitations.
- `.env`, API keys, tokens, credentials, and local-only secrets are not
  committed.
- `git diff --check` passes.
- `py_compile`, Ruff, mypy, and pytest pass.
- Packaging build succeeds.
- Known limitations are documented.

## Current Limitations

- This is a learning project, not a hardened sandbox.
- Model behavior is nondeterministic with live providers.
- Token cost estimates currently cover the configured pricing model used by
  the project and may need updates when changing models.
- Web search requires Tavily configuration.
- Provider support depends on streaming tool-call compatibility.
- The public Python API is intentionally small and may evolve as the project
  moves from Week 4 learning code toward a fuller coding-agent product.
