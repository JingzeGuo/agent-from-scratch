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

On a held-out random 100-instance subset of SWE-bench Lite, the agent resolved
**74/100 instances (74%)** with DeepSeek V4 Flash. Patches were generated inside
each instance's SWE-bench image and scored by the official SWE-bench harness, as
described below.

## Requirements and setup

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A DeepSeek API key for interactive use or live-model evaluation
- A Tavily API key only when using `search_web`
- On Linux, `bwrap` for sandboxed commands (for example `apt install bubblewrap`)

Install the project and development dependencies:

```bash
uv sync --dev
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. `uv sync` generates the `myagent` command in
`.venv/bin/`; to run it from any directory, link it onto your `PATH` from this
repository's root:

```bash
ln -s "$PWD/.venv/bin/myagent" ~/.local/bin/myagent
```

Then start the agent from the repository you want it to operate on:

```bash
cd /path/to/target-repository
myagent
```

The current directory becomes the workspace root. `python -m agent` is equivalent
to the `myagent` command.

## Configuration

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` | Required provider API key; `--api-key` overrides it |
| `DEEPSEEK_MODEL` | Model name; defaults to `deepseek-v4-flash` |
| `DEEPSEEK_BASE_URL` | API base URL; defaults to `https://api.deepseek.com` |
| `TAVILY_API_KEY` | Required only by `search_web` |
| `AGENT_STATE_DIR` | Session and trace directory; defaults to `<workspace>/.agents` |
| `AGENT_CONTEXT_TOKENS` | Usable input budget before context consolidation; defaults to `128000` |
| `AGENT_DEBUG` | Set to `1`, `true`, `yes`, or `on` to print provider tracebacks |
| `AGENT_TRACE_REDACT_PATTERNS` | Optional newline-separated regular expressions redacted from traces |

Relative `AGENT_STATE_DIR` values are resolved from the workspace root.

Estimated cost is available for the models listed in
`agent/state/token_tracker.py`. The estimate covers configured input and output token
prices only.

## CLI

Startup options:

```text
myagent [--api-key KEY] [--resume SESSION_ID_OR_NAME]
myagent --help
myagent --version
myagent eval [evaluation options]
```

Interactive commands:

| Command | Behavior |
| --- | --- |
| `/help` | Show commands |
| `/tokens` | Show input/output tokens and estimated cost |
| `/status` | Show provider, workspace, session, and controller state |
| `/reset` | Clear conversation messages, working context, and approval cache |
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
myagent --resume session-20260813-120000-000000
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

`run_command` runs commands with `bash -c` (or `/bin/sh -c` when bash is
missing) and `pipefail` enabled, so a failure anywhere in a pipeline such as
`pytest | tail` fails the whole command. Commands run inside an OS sandbox:
`sandbox-exec` on macOS
and [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) on Linux.
Both apply the same policy:

- Commands may read any file but write only the workspace and temporary
  directories.
- `.git/` and `.agents/` stay read-only, so commands cannot plant Git hooks or
  alter the agent's memory and sessions. `write_file` and `edit_file` enforce
  the same rule.
- Network access is unavailable.
- Environment variables named like credentials (`*_API_KEY`, `*_TOKEN`, ...)
  are removed.
- A timeout kills the whole process group.

Sandboxed commands run without approval. When no sandbox is available (other
platforms, `bwrap` missing, or user namespaces blocked, as inside many
containers), every command requires interactive approval and evaluation runs
cannot execute commands.

The sandbox does not protect files inside the workspace: a command can still
delete or overwrite uncommitted work there. Commit or stash before long tasks.

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
  memory.sqlite3            repository-local long-term memory
```

Snapshots preserve messages, completed runs with their steps, file tracking, and
token totals. A pending-action marker records the most recent tool call that
started after the last completed checkpoint. It remains until the interactive
turn is checkpointed, even if the tool has already finished, so resume reports that the
workspace may be ahead of the saved conversation state.

Trace events cover model requests and responses, scheduling, approvals, tool
execution, child runs, checkpoints, and run outcomes. Common
secret-like values and configured redaction patterns are removed before events
are written.

Working context is an optional `ConsolidatedState` followed by the raw recent
history. Task completion does not summarize anything. Below the soft threshold,
even multi-task sessions remain raw. At the threshold, the context builder first
replaces old tool results (oldest first, skipping the most recent and small ones)
with placeholders naming the tool and its arguments, until the context drops below
the clear target. This needs no model call and only ever advances, so the request
prefix stays stable for prompt caching. If that is not enough, it folds
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

Configure `Agent(..., context_config=ContextConfig(...))` in Python, or set
`AGENT_CONTEXT_TOKENS` to change only the input budget. Defaults:

| Setting | Default |
| --- | --- |
| Usable input budget (reserve model output separately) | 128,000 tokens |
| Soft / emergency threshold | 65% / 90% |
| Consolidated state hard limit | 2,048 tokens |
| State budget reserved when choosing a fold boundary | 1,024 tokens |
| Active task's minimum raw suffix | 8 messages, expanded to a tool boundary |
| Pathological single tool-result limit | 16,000 tokens |
| Retained head + tail of a pathological result | 2,000 tokens plus truncation marker |
| Tool-result clear target | 50% |
| Most recent tool results never cleared | 6 |
| Smallest tool result worth clearing | 200 tokens |

Token pressure includes system instructions, long-term memory, tool definitions,
and runtime step instructions. Counting uses `tiktoken`'s `cl100k_base` BPE as an estimate, not
DeepSeek's exact tokenizer; `ContextBuilder` accepts an alternative token counter.
The tokenizer downloads and caches its vocabulary on first use (offline hosts
must prepopulate the cache, optionally via `TIKTOKEN_CACHE_DIR`). Normal-sized
tool results remain intact; pathological results retain their head/tail and tool
metadata regardless of age.

Snapshots still contain full raw messages and run history, plus the single
consolidated state, folded-prefix offset, and task boundaries. `/reset` clears
the working state with the conversation and retains completed run history.
Context building returns the messages directly; token counting is used for
budget decisions without generating unused diagnostic reports.

## Long-term memory (V1)

The interactive CLI uses one SQLite store at `.agents/memory.sqlite3`, or
`$AGENT_STATE_DIR/memory.sqlite3`. Use a separate state directory for each
repository. Facts and historical experiences share the same table; user
preferences are facts with user scope. Both scopes remain local to this store
in V1, with no cross-repository sharing.

At each task start, the agent loads active core memories and performs one FTS5
lexical search, with file/path overlap and near-duplicate filtering. Defaults
are at most four retrieved memories, 1,500 tokens of core memory and 1,800 tokens
of retrieved memory. Whole records and section metadata must fit; oversized
records are skipped. Core admission is conservative, with overflow routed to
retrieval. Historical memories are explicitly subordinate to current user
instructions and repository state.

`TaskMemoryContext` is temporary request context in the system prompt. It is
never appended to `Agent.messages`, folded into `ConsolidatedState`, or stored
in session snapshots. Same-session continuity continues to use working memory.
`/reset` clears the temporary context but leaves persistent memories intact.

Task completion captures an immutable snapshot of that run's steps (including
verification results), outcome, observed file edits, evidence references, and
injected memory IDs/content. It schedules background formation and immediately
returns. The next task never waits for formation. The existing provider extracts
zero to three evidenced CORE or RETRIEVAL candidates, then chooses ADD, MERGE,
SUPERSEDE, or NOOP against similar active memories. MERGE updates an existing
record; SUPERSEDE atomically retains the old fact and links its replacement. Experiences retain
their historical meaning. MERGE and SUPERSEDE inherit the target's access mode,
subject to the core token budget; only ADD uses the candidate's routing decision.
Echo checks reject mere reuse without new information.

Formation uses only the captured snapshot and store, with no later reads of live
agent history. Token usage contributes to the existing token tracker. Exceptions
are logged at the background-job boundary. CLI input yields the event loop, and
normal shutdown drains outstanding jobs. Jobs are in-process and are not resumed
after a crash; malformed LLM output fails that job without delaying user tasks.
File-edit metadata covers successful `edit_file`/`write_file` calls; command-driven
changes and their verification remain available in the captured tool trajectory.

For Python callers, memory is opt-in (evaluations and child agents do not enable
it implicitly):

```python
from pathlib import Path
from agent.agent import Agent
from agent.memory.store import MemoryConfig, MemoryStore

store = MemoryStore(Path(".agents/memory.sqlite3"))  # injectable for tests
agent = Agent(provider, registry, memory_store=store,
              memory_config=MemoryConfig(top_k=4))
# Keep the same event loop alive across tasks:
await agent.run("First task")
await agent.run("Second task")  # does not await First task's formation
await agent.wait_for_memory_jobs()  # shutdown/test helper only
store.close()
```

## Architecture

`agent/` keeps the controller, provider adapter, shared schemas, and policies at
the top level. Each subpackage owns one subsystem: `agent/cli/` the interactive
entry point, `agent/tooling/` tool schemas, implementations, and registration,
`agent/state/` working context, sessions, and token accounting, `agent/memory/`
long-term memory, and `agent/evaluation/` the evaluation harnesses.

| File | Responsibility |
| --- | --- |
| `agent/agent.py` | Explicit controller loop, scheduling, approvals, traces, and termination |
| `agent/provider.py` | DeepSeek transport and provider-neutral response normalization |
| `agent/prompts.py` | System prompt and tool guidance |
| `agent/schemas.py` | Provider-neutral controller, provider, and session models |
| `agent/sandbox.py` | macOS and Linux command sandbox |
| `agent/security.py` | Command approval policy and trace redaction |
| `agent/workspace.py` | Workspace-relative path resolution |
| `agent/cli/main.py` | CLI parsing, provider setup, sessions, and startup wiring |
| `agent/cli/commands.py` | Interactive slash commands, approval prompts, and checkpoints |
| `agent/tooling/catalog.py` | Built-in tool catalog and registry factory |
| `agent/tooling/inputs.py` | Validated input schemas for built-in tools |
| `agent/tooling/tool.py` | Tool definition, validation, execution, and retry boundary |
| `agent/tooling/tool_registry.py` | Dispatch, workspace isolation, action tracking, and diffs |
| `agent/tooling/files.py` | Glob, search, read, edit, write, and unified diffs |
| `agent/tooling/command.py` | Bounded, shell-free command execution |
| `agent/tooling/web.py` | URL fetching and web search |
| `agent/tooling/sub_agent.py` | Read-only child agent profiles and delegation |
| `agent/tooling/retry.py` | Retry policy for transient tool errors |
| `agent/state/context.py` | Token pressure, prefix selection, working context, emergency fallback |
| `agent/state/consolidation.py` | LLM-generated, schema-validated continuation state |
| `agent/state/session.py` | Snapshots, pending actions, and JSONL events |
| `agent/state/token_tracker.py` | Token totals and estimated cost |
| `agent/memory/models.py` | Memory records and detached task snapshots |
| `agent/memory/store.py` | SQLite memory store, lexical retrieval, and task memory budgets |
| `agent/memory/formation.py` | Immutable task capture, LLM candidate extraction, and lifecycle updates |
| `agent/evaluation/coding_tasks.py` | Deterministic, live-model, and patch-generation evaluation |

## Evaluation

The evaluation runner has three modes and intentionally avoids a large grading
framework.

### Deterministic local tasks

The default suite uses a scripted provider and temporary local repositories. It
covers repository search, a focused bug fix, and recovery from a failed edit.
Each result reports success, verification status, termination reason, steps,
tool calls, tokens, estimated cost, latency, and a failure reason when needed.

```bash
.venv/bin/myagent eval
.venv/bin/myagent eval --list
.venv/bin/myagent eval small_bug_fix
```

### Optional live-model tasks

Live provider calls are opt-in. With no case name, this mode runs only the
read-only `repository_search` case.

```bash
.venv/bin/myagent eval --real-model
.venv/bin/myagent eval --real-model small_bug_fix --max-steps 20
```

### SWE-bench-compatible patch generation

#### Result

Patches were generated at commit `1734386` (2026-10-22) with DeepSeek V4 Flash,
`--swe-bench-docker`, at most 40 steps, and the default 128k context budget, then
scored once with the official harness on `linux/amd64`.

The 100 test instances were drawn with a fixed seed from the 250 SWE-bench Lite
instances outside a separate 50-instance development set; all agent changes
were made against the development set only. The seed and instance IDs are in
`swe_bench_lite_test_100.meta.json`.

| Metric | Result |
| --- | ---: |
| Resolved | **74/100 (74%)** |
| Unresolved after harness execution | 26 |
| Empty patches | 0 |
| Infrastructure failures / evaluator errors | 0 / 0 |
| Estimated model cost | $9.74 |
| Mean model steps per instance | 29.4 |

This is a single-run subset result (roughly ±9 points at 95% confidence), not a
claim about the complete SWE-bench Lite test set or leaderboard. The model may
have seen these public issues during training.

To score predictions, pre-pull the `linux/amd64` evaluation images (the harness
otherwise pulls for the host platform and fails on Apple Silicon). The SWE-bench
Lite export already includes the harness `image` and `eval_script` fields, so the
same file can be passed to both the agent and the harness:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name swe_bench_lite_test_100.jsonl \
  --predictions_path predictions.jsonl \
  --max_workers 1 --timeout 3600 \
  --run_id <run-id> \
  --report_dir official-reports
```

Pass a JSON or JSONL export containing `instance_id`, `repo`, `base_commit`, and
`problem_statement`. The runner clones each selected repository, checks out the
base commit, runs the live agent, collects the Git diff, and writes JSONL
predictions with exactly these standard fields:

- `instance_id`
- `model_name_or_path`
- `model_patch`

```bash
.venv/bin/myagent eval \
  --swe-bench instances.jsonl \
  --instance-id owner__repo-123 \
  --swe-bench-predictions predictions.jsonl \
  --swe-bench-metrics predictions.metrics.jsonl \
  --swe-bench-trajectories predictions.trajectories
```

With `--swe-bench-docker`, each instance runs in its prebuilt
`swebench/sweb.eval.x86_64.*` image instead of a fresh clone, so the repository's
own tests can run. The image's `/testbed` is copied to the host workspace, its Git
history is replaced by a single baseline commit, and it is mounted back into a
container without network access. File tools edit the host copy; `run_command`
runs in the container.

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
.venv/bin/python -m compileall -q agent
.venv/bin/ruff check .
.venv/bin/mypy .
.venv/bin/python -m pytest -q
.venv/bin/myagent eval
```

Tests use fake providers and temporary workspaces; they do not make live API
calls.
