# agent-from-scratch

A local terminal coding agent built with Python, Pydantic, and DeepSeek Chat
Completions. The controller keeps the agent loop explicit: the model chooses
from structured tools, tool observations return to the model, and the loop ends
on completion, protocol failure, or a bounded step limit.

The agent is designed for practical repository work. It confines file and
command operations to the current workspace, validates every tool input,
records resumable sessions and JSONL traces, compacts long conversations into
structured checkpoints, and tracks token use and estimated cost.

## Requirements and setup

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A DeepSeek API key for interactive use or live-model evaluation
- A Tavily API key only when using `search_web`

Install the project and development dependencies:

```bash
uv sync --dev
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`, then start the agent from the repository you
want it to operate on:

```bash
cd /path/to/target-repository
/path/to/agent-from-scratch/.venv/bin/agent
```

The current directory becomes the workspace root.

## Configuration

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` | Required provider API key; `--api-key` overrides it |
| `DEEPSEEK_MODEL` | Model name; defaults to `deepseek-v4-flash` |
| `DEEPSEEK_BASE_URL` | API base URL; defaults to `https://api.deepseek.com` |
| `TAVILY_API_KEY` | Required only by `search_web` |
| `AGENT_STATE_DIR` | Session and trace directory; defaults to `<workspace>/.agents` |
| `AGENT_DEBUG` | Set to `1`, `true`, `yes`, or `on` to print provider tracebacks |
| `AGENT_TRACE_REDACT_PATTERNS` | Optional newline-separated regular expressions redacted from traces |

Relative `AGENT_STATE_DIR` values are resolved from the workspace root.

Estimated cost is available for the models listed in
`agent/token_tracker.py`. The estimate covers configured input and output token
prices only.

## CLI

Startup options:

```text
agent [--api-key KEY] [--resume SESSION_ID_OR_NAME]
agent --help
agent --version
agent eval [evaluation options]
```

Interactive commands:

| Command | Behavior |
| --- | --- |
| `/help` | Show commands |
| `/tokens` | Show input/output tokens and estimated cost |
| `/status` | Show provider, workspace, session, and controller state |
| `/reset` | Clear conversation messages, steps, and approval cache |
| `/save` | Save a session checkpoint |
| `/diff [path]` | Show session changes, optionally for one file |
| `/compact` | Report current context-compaction metrics |
| `/trace [path]` | Print trace events or export them inside the workspace |
| `/rename <name>` | Rename and save the current session |
| `/sessions` | List saved sessions |
| `/paste` | Enter multiline input; finish with `/send` or cancel with `/cancel` |
| `/exit` | Exit the application |

Completed interactive turns are checkpointed automatically. Resume by session
ID or name:

```bash
agent --resume session-20260813-120000-000000
```

## Tools

The built-in `Tool` and `ToolRegistry` classes are the complete tool
abstraction. The default registry contains:

| Tool | Behavior |
| --- | --- |
| `read_file` | Read a bounded line range from a workspace text file |
| `glob_files` | Find workspace files matching a bounded glob |
| `search_text` | Search workspace files with a regular expression |
| `edit_file` | Replace one exact unique match and return a unified diff |
| `write_file` | Create a file or intentionally overwrite a complete file |
| `get_diff` | Return unified diffs for files changed in the session |
| `run_command` | Run a bounded command in the workspace |
| `sub_agent` | Run a bounded, isolated, read-only repository exploration |
| `fetch_url` | Fetch a known URL with bounded output |
| `search_web` | Search with Tavily and return bounded structured results |

The `read_only_explorer` profile used by `sub_agent` exposes only `read_file`,
`glob_files`, and `search_text`, with a maximum of eight steps. It cannot edit,
run commands, access the network, or delegate recursively.

## Controller behavior

For each user task, `Agent.run`:

1. Adds the user message to conversation state.
2. Builds bounded model context, adding a structured checkpoint when prior
   steps exist.
3. Streams a normalized provider response.
4. Validates and schedules requested tools.
5. Returns tool results as observations and continues the loop.
6. Records verification evidence and a termination reason.

Multiple calls run concurrently only when every requested tool is in the
controller's read-only set. Mutating or order-sensitive calls run serially.
The default maximum is 40 model steps per task.

The controller requires an existing file to be read before `edit_file` may
change it or `write_file` may overwrite it. Tool validation and execution errors
are returned to the model as recoverable observations.

## Command safety

`run_command` parses arguments without a shell, rejects shell operators and
command substitution, blocks destructive commands, and confines `cwd` to the
workspace. Focused commands such as `pytest`, `mypy`, `ruff`, `py_compile`, and
read-only Git inspection run automatically. Broader commands require interactive
approval.

Command output is bounded and includes exit code, timeout state, duration,
stdout, and stderr. Approval is a controller decision rather than a property of
the shell.

## Sessions, traces, and context

By default, runtime state is stored under `.agents/`:

```text
.agents/
  sessions/                 resumable JSON snapshots
    events/                 append-only JSONL traces
    pending/                in-flight tool markers
  evals/                    generated evaluation output
```

Snapshots preserve messages, steps, completed runs, file tracking, and token
totals. Pending-action markers let resume report a tool call that started before
the last completed checkpoint.

Trace events cover model requests and responses, scheduling, approvals, tool
execution, child runs, compaction, checkpoints, and run outcomes. Common
secret-like values and configured redaction patterns are removed before events
are written.

Long conversations use deterministic context compaction. Older large tool
results are shortened, while a structured checkpoint retains the goal, files,
edits, decisions, commands, errors, pending action, and latest verification.

## Architecture

| File | Responsibility |
| --- | --- |
| `main.py` | CLI parsing, provider setup, sessions, and startup wiring |
| `agent/agent.py` | Explicit controller loop, scheduling, approvals, traces, and termination |
| `agent/provider.py` | DeepSeek transport and provider-neutral response normalization |
| `agent/setup.py` | Default tool registry and read-only child profile |
| `agent/tool.py` | Tool schemas, validation, execution, and retry boundary |
| `agent/tool_registry.py` | Dispatch, workspace action tracking, and diffs |
| `agent/tools.py` | Built-in tool implementations |
| `agent/context.py` | Bounded context and structured checkpoints |
| `agent/session.py` | Snapshots, pending actions, and JSONL events |
| `agent/schemas.py` | Provider-neutral controller and session models |
| `agent/security.py` | Command policy and trace redaction |
| `agent/token_tracker.py` | Token totals and estimated cost |
| `agent/verification.py` | Verification evidence and task-success inference |
| `scripts/evaluate_coding_tasks.py` | Deterministic, live-model, and patch-generation evaluation |

## Evaluation

The evaluation runner has three modes and intentionally avoids a large grading
framework.

### Deterministic local tasks

The default suite uses a scripted provider and temporary local repositories. It
covers repository search, a focused bug fix, and recovery from a failed edit.
Each result reports success, verification status, termination reason, steps,
tool calls, tokens, estimated cost, latency, and a failure reason when needed.

```bash
.venv/bin/agent eval
.venv/bin/agent eval --list
.venv/bin/agent eval small_bug_fix
```

### Optional live-model tasks

Live provider calls are opt-in. With no case name, this mode runs only the
read-only `repository_search` case.

```bash
.venv/bin/agent eval --real-model
.venv/bin/agent eval --real-model small_bug_fix --max-steps 20
```

### SWE-bench-compatible patch generation

Pass a JSON or JSONL export containing `instance_id`, `repo`, `base_commit`, and
`problem_statement`. The runner clones each selected repository, checks out the
base commit, runs the live agent, collects the Git diff, and writes JSONL
predictions with exactly these standard fields:

- `instance_id`
- `model_name_or_path`
- `model_patch`

```bash
.venv/bin/agent eval \
  --swe-bench instances.jsonl \
  --instance-id owner__repo-123 \
  --swe-bench-predictions predictions.jsonl
```

Use `--swe-bench-limit N` for a prefix of the selected instances and repeat
`--instance-id` to select several IDs. This mode generates patches only. Use the
official SWE-bench harness for environment construction and scoring.

## Development checks

Run the normal local checks with the project environment:

```bash
.venv/bin/python -m py_compile main.py agent/*.py scripts/*.py
.venv/bin/ruff check .
.venv/bin/mypy .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate_coding_tasks.py
```

Tests use fake providers and temporary workspaces; they do not make live API
calls.
