# agent-from-scratch

A terminal coding agent built with Python, Pydantic, and LLM tool calling.

The agent runs inside a local workspace, plans and executes multi-step coding
tasks, uses structured tools for repository inspection and edits, records
session state, and tracks token usage and estimated cost.

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
- Project and global memory stores for durable agent context
- Local hybrid memory retrieval with BM25-like lexical scoring and TF-IDF cosine
- Automatic run reflection into session, topic, profile, and cross-project memories
- Structured JSONL trace events with secret redaction
- Token and estimated cost tracking
- Optional web search and URL fetching
- Read-only sub-agent delegation for bounded repository exploration
- Optional stdio MCP server tool loading

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
| `AGENT_MEMORY_ENABLED` | Enable memory retrieval and reflection, default `true` |
| `AGENT_MEMORY_GLOBAL_DIR` | Global memory directory, default `~/.agent-from-scratch/memory` |
| `AGENT_MEMORY_MAX_RESULTS` | Maximum retrieved memory records per model request, default `5` |
| `AGENT_MEMORY_MAX_CONTEXT_CHARS` | Maximum memory context characters inserted into a request, default `4000` |
| `AGENT_MCP_CONFIG` | Optional path to an MCP server config. If unset, `.agents/mcp.json` is loaded when present |

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
| `/memory status` | Show memory paths and record counts |
| `/memory search <query>` | Search project and global memory |
| `/memory show <id>` | Show one memory record |
| `/memory reflect` | Reflect on the latest completed run and save memory |
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
| `mcp_<server>__<tool>` | mcp | Call tools discovered from configured stdio MCP servers |

All tool inputs are validated with Pydantic before execution. Validation
failures are returned to the model as error observations so the agent can
recover by choosing corrected arguments.

### MCP Tools

This project implements a minimal stdio MCP client, not the full MCP
specification. It currently supports local stdio servers, `initialize`,
`tools/list`, and `tools/call`. It does not yet support HTTP transport,
resources, prompts, sampling, auth, progress notifications, or tool-list change
notifications.

The agent can load tools from configured stdio MCP servers at startup. By
default, it looks for `.agents/mcp.json` in the active workspace. Set
`AGENT_MCP_CONFIG` to use a different JSON file, or set it to an empty value to
disable MCP config loading explicitly.

The config uses the common `mcpServers` shape with explicit trust and approval
settings:

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "TOKEN": "..."
      },
      "cwd": "tools",
      "trust": "untrusted",
      "approval": "auto",
      "allowedTools": ["list_items", "get_item"],
      "blockedTools": ["delete_item"],
      "readOnlyTools": ["list_items", "get_item"],
      "allowExternalCwd": false
    }
  }
}
```

Each configured server is launched without a shell, initialized over stdio, and
queried with `tools/list`. Exposed tools are registered as
`mcp_<server>__<tool>` and tool calls are forwarded with `tools/call`.

Tool registration follows `blockedTools > allowedTools > discovered tools`.
`blockedTools` are never registered. If `allowedTools` is omitted, every
discovered tool that is not blocked is registered. If `allowedTools` is present,
only listed tools are registered.

By default, MCP servers use `trust: "untrusted"` and `approval: "auto"`.
Approval modes are:

- `always`: every MCP tool call requires user approval.
- `auto`: tools listed in `readOnlyTools` run without approval; other MCP tools
  require approval.
- `never`: MCP tool calls run without approval. Use this only for servers you
  trust.

`readOnlyTools` is a user configuration claim used only for approval decisions;
it does not affect registration and is not inferred from server-provided tool
descriptions.

MCP server working directories are workspace-confined by default. If `cwd` is
omitted, the server starts in the workspace root. Relative `cwd` values are
resolved under the workspace and rejected if they escape it. Absolute `cwd`
values are rejected unless `allowExternalCwd` is set to `true`.

`sub_agent` is a delegated tool rather than a normal file or command helper.
When the main controller executes it, the tool creates an isolated child agent
with a fresh conversation, the same provider adapter, and a read-only registry
containing `calculator`, `read_file`, `glob_files`, `search_text`, and
`get_diff`. The child agent cannot edit files, run commands, use network tools,
or recursively spawn another sub-agent.

## Python API

The current library API is intentionally small and centered on constructing tool
registries and agent instances.

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
       -> Memory system
       -> Tool registry
            -> repository tools
            -> edit/write tools
            -> command tool
            -> web tools
            -> MCP server tools
            -> sub-agent tool
       -> Session store
       -> Trace writer
       -> Token and cost tracker
```

Core modules:

| Path | Responsibility |
| --- | --- |
| `main.py` | CLI parsing, startup wiring, provider setup, interactive loop |
| `agent/cli_commands.py` | Slash commands, session checkpoint commands, memory and trace commands |
| `agent/agent.py` | Agent controller, run loop, tool scheduling, recovery, termination |
| `agent/provider.py` | Anthropic and OpenAI-compatible provider adapters |
| `agent/setup.py` | Built-in tool registry construction |
| `agent/tool.py` | Tool wrapper, schema generation, validation, retry |
| `agent/tool_registry.py` | Tool storage, dispatch, changed-file tracking, diffs |
| `agent/tools.py` | Tool implementations |
| `agent/schemas.py` | Pydantic models for tools, runs, sessions, traces, context |
| `agent/workspace.py` | Workspace path normalization and escape rejection |
| `agent/security.py` | Command policy and trace redaction |
| `agent/session.py` | Session snapshots, pending actions, JSONL trace events |
| `agent/context.py` | Context compaction and checkpoint construction |
| `agent/memory.py` | Project/global memory stores, retrieval, and run reflection |
| `agent/token_tracker.py` | Token and estimated cost tracking |
| `agent/verification.py` | Verification evidence extraction |

## Operating Model

The core loop is in `agent/agent.py`: the controller keeps conversation and
step state, sends context plus tool definitions to the model, receives either
final text or tool calls, executes actions through the registry, appends tool
observations, and stops on completion, protocol error, or the maximum step
limit.

The recovery story is tool-centered: Pydantic validates inputs before execution,
validation and runtime failures are returned as observations, transient tool
failures retry up to three attempts, command approval handles risky actions,
and focused verification evidence is extracted from command results.

The advanced layers are optional extensions around the loop. Context compaction
keeps long sessions usable, session checkpoints make runs resumable, trace
events make behavior inspectable, memory adds durable context, and `sub_agent`
is a controlled read-only delegation path for narrow repository exploration.

## Memory

Memory is separate from session checkpoints. Checkpoints preserve resumable
controller state; memory preserves durable context that may help future tasks.

Project memory lives inside the active workspace:

```text
.agents/memory/
  profile.md
  index.json
  sessions/
  topics/
  reflections/
```

Global memory defaults to `~/.agent-from-scratch/memory` and uses the same
layout. Project memory is for repository-specific facts, session notes,
debugging history, and reflections. Global memory is reserved for stable user
preferences and cross-project notes that should still matter in another
repository.

On each task, the agent searches project and global memory using local hybrid
retrieval: BM25-like lexical scoring plus TF-IDF cosine scoring over local
tokens. Retrieved memory is inserted as supporting context after the structured
checkpoint and does not override system or project rules.

After a run finishes, the agent asks the configured model to propose concise
memory candidates. The controller filters candidates before persistence, redacts
secret-like text, rejects project-specific global memories, and stores accepted
records as Markdown plus `index.json` entries. If memory reflection fails, the
main task result is still preserved.

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
- External MCP tools are mapped into the same approval system and are not
  automatically trusted.
- MCP server `cwd` is workspace-confined by default; external absolute working
  directories require explicit `allowExternalCwd`.
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
.venv/bin/agent eval
.venv/bin/python scripts/evaluate_tool_selection.py
.venv/bin/python scripts/evaluate_coding_tasks.py
```

The `agent eval` command runs the default deterministic coding-task suite of
13 local tasks. It reports pass rate, average steps, average token cost, average
tool calls, and failure counts for compile errors, test failures, max step
exits, and unsafe blocked commands.

Run selected cases or a live-provider smoke case:

```bash
.venv/bin/agent eval small_bug_fix targeted_refactor
.venv/bin/agent eval --real-model repository_search
```

Run SWE-bench-style instances from a local JSONL export:

```bash
.venv/bin/agent eval --swe-bench swe-bench-lite.jsonl --swe-bench-limit 3
```

SWE-bench mode checks out each instance's `repo` at `base_commit`, asks the
agent to produce a patch, writes predictions to
`.agents/evals/swe-bench-predictions.jsonl`, and runs best-effort local pytest
targets from `FAIL_TO_PASS` and `PASS_TO_PASS` when present. The predictions
file uses `instance_id`, `model_name_or_path`, and `model_patch` fields so it
can be passed to the official SWE-bench harness for Docker-based scoring.

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

- This is not a hardened sandbox.
- Model behavior is nondeterministic with live providers.
- Token cost estimates currently cover the configured pricing model used by
  the project and may need updates when changing models.
- Web search requires Tavily configuration.
- MCP support is limited to local stdio tool discovery and calls; it is not a
  full MCP client.
- Provider support depends on streaming tool-call compatibility.
- The public Python API is intentionally small and may evolve.
