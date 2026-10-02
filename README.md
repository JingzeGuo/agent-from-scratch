# agent-from-scratch

A local terminal coding agent built with Python, Pydantic, and DeepSeek Chat
Completions. The controller keeps the agent loop explicit: the model chooses
from structured tools, tool observations return to the model, and the loop ends
on completion, protocol failure, or a bounded step limit.

The agent is designed for practical repository work. It confines file
operations and command working directories to the current workspace, validates
every tool input, records resumable sessions and JSONL traces, compacts long
conversations into a single continuation state under context pressure, and tracks token use and estimated
cost.

In an audited run on a fixed random 50-instance subset of SWE-bench Lite, the
agent resolved **29/50 instances (58%)** with DeepSeek V4 Flash. Patches were
scored by the official SWE-bench harness under the isolated evaluation profile
described below.

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
`agent/state/token_tracker.py`. The estimate covers configured input and output token
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
2. Preserves raw context until token pressure requires consolidating an old prefix.
3. Streams a normalized provider response.
4. Validates and schedules requested tools.
5. Returns tool results as observations and continues the loop.
6. Records the run steps and termination reason.

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
    pending/                uncheckpointed tool-action markers
  evals/                    generated evaluation output
```

Snapshots preserve messages, steps, completed runs, file tracking, and token
totals. A pending-action marker records the most recent tool call that started
after the last completed checkpoint. It remains until the interactive turn is
checkpointed, even if the tool has already finished, so resume reports that the
workspace may be ahead of the saved conversation state.

Trace events cover model requests and responses, scheduling, approvals, tool
execution, child runs, compaction, checkpoints, and run outcomes. Common
secret-like values and configured redaction patterns are removed before events
are written.

Working context is an optional `ConsolidatedState` followed by the raw recent
history. Task completion does not summarize anything. Below the soft threshold,
even multi-task sessions remain raw. At the threshold, the context builder folds
the oldest completed tasks first, stopping as soon as the state plus remaining raw
history fits below it. Only when necessary does it fold an active task's old
prefix, preserving the latest messages and complete tool exchanges. Later folds
merge the previous state with newly aged history into one replacement state.

`agent/state/consolidation.py` uses the existing model adapter to construct validated
JSON with findings, decisions, changes, unresolved work, verification, relevant
context, and the current task's objective. Invalid or oversized output shares one
repair attempt before fallback; state size includes the objective and message framing. Failed
or insufficient consolidation falls back to hard collapse with an explicit loss
warning. If even the latest indivisible exchange cannot fit, the request fails
with `ContextBudgetExceeded` instead of sending an oversized/broken tool sequence.
Consolidation calls contribute to session token/cost totals.

Configure `Agent(..., context_config=ContextConfig(...))` in Python. Defaults:

| Setting | Default |
| --- | --- |
| Usable input budget (reserve model output separately) | 32,000 tokens |
| Soft / emergency threshold | 65% / 90% |
| Consolidated state hard limit | 2,048 tokens |
| State budget reserved when choosing a fold boundary | 1,024 tokens |
| Active task's minimum raw suffix | 8 messages, expanded to a tool boundary |
| Pathological single tool-result limit | 16,000 tokens |
| Retained head + tail of a pathological result | 2,000 tokens plus truncation marker |

Token pressure includes system instructions, tool definitions, and runtime step
instructions. Counting uses `tiktoken`'s `cl100k_base` BPE as an estimate, not
DeepSeek's exact tokenizer; `ContextBuilder` accepts an alternative token counter.
The tokenizer downloads and caches its vocabulary on first use (offline hosts
must prepopulate the cache, optionally via `TIKTOKEN_CACHE_DIR`). Normal-sized
tool results remain intact; pathological results retain their head/tail and tool
metadata regardless of age.

Snapshots still contain full raw messages and run history, plus the single
consolidated state, folded-prefix offset, and task boundaries. Old snapshots load
with an empty working state. `/reset` clears this state with the conversation.
Legacy character metrics and the `summary_included` trace field remain compatible
(the latter now means that a consolidated state is present).

## Architecture

`agent/` keeps the controller, provider adapter, shared schemas, and policies at
the top level. `agent/state/` owns context, sessions, and token accounting;
`agent/tooling/` owns tool implementations, registration, and retries.

| File | Responsibility |
| --- | --- |
| `main.py` | CLI parsing, provider setup, sessions, and startup wiring |
| `agent/agent.py` | Explicit controller loop, scheduling, approvals, traces, and termination |
| `agent/cli_commands.py` | Interactive slash commands and session controls |
| `agent/provider.py` | DeepSeek transport and provider-neutral response normalization |
| `agent/prompts.py` | System prompt and tool guidance |
| `agent/tooling/setup.py` | Default tool registry and read-only child profile |
| `agent/tooling/tool.py` | Tool schemas, validation, execution, and retry boundary |
| `agent/tooling/tool_registry.py` | Dispatch, workspace action tracking, and diffs |
| `agent/tooling/tools.py` | Built-in tool implementations |
| `agent/tooling/retry.py` | Retry policy for transient tool errors |
| `agent/state/context.py` | Token pressure, prefix selection, working context, emergency fallback |
| `agent/state/consolidation.py` | LLM-generated, schema-validated continuation state |
| `agent/state/session.py` | Snapshots, pending actions, and JSONL events |
| `agent/schemas.py` | Provider-neutral controller and session models |
| `agent/security.py` | Command policy and trace redaction |
| `agent/workspace.py` | Workspace-relative path resolution |
| `agent/state/token_tracker.py` | Token totals and estimated cost |
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

#### Audited result

The isolated evaluation profile was run against a fixed random 50-instance
subset of SWE-bench Lite and scored with the official harness on `linux/amd64`.
The denominator includes empty patches.

| Metric | Result |
| --- | ---: |
| Resolved | **29/50 (58%)** |
| Unresolved after harness execution | 17 |
| Empty patches | 4 |
| Submitted | 50 |
| Completed by the harness | 46 |
| Infrastructure failures | 0 |
| Evaluator errors | 0 |

One unresolved instance, `pytest-dev__pytest-5692`, was marked ambiguous
because the harness collected no tests. The run used DeepSeek V4 Flash, one
harness worker, a 3,600-second per-instance timeout, and run ID
`deepseek-v4-flash-audit-random-50-v2-amd64`. This is a subset result, not a
claim about the complete SWE-bench Lite test set or leaderboard.

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
  --swe-bench-predictions predictions.jsonl \
  --swe-bench-metrics predictions.metrics.jsonl \
  --swe-bench-trajectories predictions.trajectories
```

Use `--swe-bench-limit N` for a prefix of the selected instances and repeat
`--instance-id` to select several IDs. This mode generates patches only. Use the
official SWE-bench harness for environment construction and scoring.

SWE-bench runs use a dedicated patch-generation profile. The profile instructs
the model to treat missing-dependency or environment failures as terminal
verification blocks instead of triggering changes to the repository. It also
instructs the agent to reserve the last two model steps for diff review and the
final answer. Each completed instance is immediately appended to a metrics JSONL
file with termination, step, tool-call, token, and changed-file details. When
`--swe-bench-metrics` is omitted, the path defaults to
`<predictions-stem>.metrics.jsonl`. Evaluation activity is prefixed with the
instance ID and `main` or `sub` agent role.

The SWE-bench profile removes `search_web` and `fetch_url`, uses an anonymous
UUID workspace, fetches only the requested `base_commit`, removes the Git
remote, and blocks Agent access to Git commands and `.git` metadata. Structured
per-instance trajectory events record the Agent role, tool name, complete tool
input, and command while redacting configured secrets. When
`--swe-bench-trajectories` is omitted, trajectories default to
`<predictions-stem>.trajectories/events/<instance-id>.jsonl`.

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
