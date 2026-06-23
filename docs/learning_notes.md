# Agent Learning Notes

## Day 1: Agent Anatomy and Trajectory

Date: 2026-06-09

### Core Question

What makes this program an agent instead of a chatbot with functions?

### Key Concepts

- **Objective**: The user's task that defines the desired outcome.
- **State**: The information available for the next decision, including the
  objective, conversation history, tool calls, and observations.
- **Policy**: The LLM component that selects the next action from the current
  state.
- **Action space**: The available tools and the option to return a final
  answer.
- **Action**: A tool call or final response selected by the policy.
- **Environment**: The external system in which an action is executed.
- **Observation**: The result or error returned after an action.
- **Controller**: The agent loop that calls the model, executes actions,
  updates state, and applies termination rules.
- **Trajectory**: The sequence of states, actions, and observations from the
  initial objective to termination.

### Calculator Trajectory

Task: Calculate `(17 * 8) + 23`.

```text
Initial state
-> calculator(expression="(17 * 8) + 23")
-> observation: "159"
-> updated state containing the tool result
-> final answer: "The result is 159."
-> termination: completed
```

### Multi-Tool Trajectory

Task: Search for the official Python `asyncio.TaskGroup` documentation, read
the page, and summarize the problem it solves.

```text
Initial state
-> search_web(query="site:docs.python.org asyncio TaskGroup")
-> observation: official documentation search result and URL
-> fetch_url(url="<official documentation URL>")
-> observation: documentation page content
-> final answer: summary based on the retrieved content
-> termination: completed
```

The search result is only a candidate source. Reading the page is a separate
action and observation.

### Model and Controller Responsibilities

The model decides:

- whether to use a tool
- which tool to use
- which arguments to provide
- what to do after receiving an observation
- when and how to produce the final answer

The controller decides:

- which state and tools are sent to the model
- how tool calls are executed
- how observations are added to the state
- when the model is called again
- when runtime safety conditions stop the loop
- how the trajectory is recorded

### ReAct and Agent State

Classic ReAct describes model behavior as:

```text
Thought -> Action -> Observation
```

Agent-system analysis describes the complete loop as:

```text
State -> Policy decision -> Action -> Observation -> Next state
```

The model must make a decision, but the system does not need to expose or
store private chain-of-thought. Public plans or decision summaries may be
recorded when available.

### Implementation Result

`AgentStep` represents one model step and its tool interactions.

`AgentRun` represents one user task:

```text
AgentRun
├── objective
├── steps
└── termination
```

The supported termination outcomes are:

- `completed`
- `max_steps`
- `unexpected_stop`

`AgentRun.steps` contains only the steps from the current task, while
`Agent.steps` continues to collect steps across the complete CLI session.

### Failure Cases

A successful tool execution does not prove task success. The model may:

- select the wrong tool
- provide incorrect arguments
- misunderstand a correct observation
- omit required actions
- produce an incorrect final answer

The provider returning `end_turn` proves only that generation ended normally.
It does not prove that the user's objective was satisfied correctly.

### Interview Answers

**Why is an LLM not an agent?**

An LLM is the policy or decision component. An agent also requires an
objective, state, action space, environment, observations, a controller,
memory or context, and termination rules.

**Why does tool success not imply task success?**

Tool success proves only that one action executed successfully. Task success
also depends on selecting the right action, using correct arguments,
interpreting the observation correctly, and producing an answer that
satisfies the objective.

**What is the difference between an AgentStep, AgentRun, and session?**

- An `AgentStep` is one decision step.
- An `AgentRun` is the complete trajectory for one user task.
- A session is the complete multi-turn conversation containing multiple runs.

### Verification

- Structured run outcomes are covered by focused tests.
- Multiple runs keep independent `AgentRun.steps`.
- The full test suite passes.
- Ruff, mypy, compilation, and diff checks pass.

## Day 2: Tool Calling as an Action Space

Date: 2026-06-09

### Core Question

How do tool names, descriptions, schemas, and granularity influence the
model's decisions?

### Key Concepts

- The registered tools and final-answer option form the agent's action space.
- Tool names and descriptions primarily help the model select an action.
- Input schemas guide argument generation and validate argument structure.
- A tool-selection error chooses the wrong action.
- A tool-argument error chooses the right tool with invalid or unsuitable
  arguments.
- Pydantic validates the selected tool's arguments. It does not determine
  whether the model selected the correct tool for the task.
- Bounded tool results protect the context window from oversized
  observations.

### Fixed Evaluation Cases

The deterministic baseline contains:

- calculation -> `calculator`
- local file -> `read_file`
- known URL -> `fetch_url`
- unknown web information -> `search_web`
- research task -> `search_web`, followed by an observation-dependent
  `fetch_url`
- correct calculator selection with the wrong parameter name -> argument
  validation error

### Description Experiment

The same five selection tasks were run once with the current clear
descriptions and once with intentionally vague descriptions. The schemas and
tool names remained unchanged.

```text
Clear descriptions
selection:          5/5
schema valid:       5/5
exact arguments:    3/5

Vague descriptions
selection:          5/5
schema valid:       5/5
exact arguments:    3/5
```

The two strict argument mismatches were valid search requests with different
query wording or `max_results` values. They were not schema failures.

### Result

This small task set did not demonstrate a measurable selection benefit from
the clearer top-level descriptions. The tasks were simple, and the tool names
plus detailed field schemas still supplied enough affordance.

No production descriptions or schemas were changed because the experiment did
not provide evidence for a specific improvement.

The experiment also shows that exact argument matching is stricter than task
success. Evaluation should distinguish:

- correct tool selection
- schema-valid arguments
- exact baseline arguments
- semantic task success

### Failure Cases

- A valid `path` string may still contain a URL, so schema validation cannot
  prevent every semantic tool-selection error.
- A valid search query may differ from the baseline while still satisfying
  the task.
- Adding more similar tools can make selection harder by creating overlapping
  affordances.
- Combining many responsibilities into one broad tool can hide intermediate
  decisions and reduce trajectory observability.

## Day 3: Retry Versus Agent Recovery

Date: 2026-06-10

### Core Question

When should the system repeat an action, and when should the model choose a
different action?

### Key Concepts

- Retry repeats the same action with the same arguments.
- Recovery makes a new policy decision after receiving an error observation.
- Retry is appropriate for transient infrastructure failures.
- Recovery is appropriate for permanent, semantic, or configuration failures.
- An error result is part of agent state and can change the next action.
- Retry budgets limit latency and repeated external work.

### Error Classification

The runtime retries:

- `TimeoutError`
- `ConnectionError`
- `httpx.TransportError`
- HTTP `408`
- HTTP `429`
- HTTP `5xx`

The runtime does not retry:

- `FileNotFoundError`
- `ValueError`
- ordinary `RuntimeError`
- HTTP `404` and other non-transient `4xx` responses
- Pydantic validation errors

Classification uses exception types and HTTP status codes rather than error
message strings.

### Baseline Experiment

Before the change:

```text
Timeout:           3 attempts, then success
Missing file:      3 attempts, then error
Invalid arguments: 0 tool executions, validation error
```

The missing-file behavior was incorrect because waiting does not make the same
file path valid.

After the change:

```text
Transient timeout: retry within the three-attempt budget
Missing file:      fail after one execution
Invalid arguments: fail before tool execution
```

### Recovery Trajectories

Invalid calculator arguments:

```text
calculator(number="1 + 1")
-> validation error observation
-> calculator(expression="1 + 1")
-> observation: "2"
-> final answer
```

Missing local file:

```text
read_file(path="missing.txt")
-> FileNotFoundError observation
-> search_web(query="requested information")
-> successful observation
-> final answer
```

Focused tests verify that each error is returned as
`ToolResult(is_error=True)` and appears in the message state sent to the next
model call.

### Interview Answers

**What is the difference between retry and recovery?**

Retry is infrastructure-controlled repetition of the same action and
arguments after a transient failure. Recovery is a new model decision based
on an error observation, such as correcting arguments or selecting another
tool.

**Why should invalid arguments not be retried three times?**

The arguments remain invalid and the environment has not changed, so the same
action will fail again. Repeating it only adds latency and resource cost. The
validation error should be returned to the model immediately.

**How does an error observation change the policy decision?**

The error becomes part of the next state. The model can use its type and
details to choose a corrected action, such as adding a missing field or
switching from a missing local file to a web search.

### Verification

- Transient and permanent errors have focused classification tests.
- Parameter correction is covered by a complete recovery trajectory.
- Missing-file fallback is covered by a complete recovery trajectory.
- Error observations are verified in the next model request.
- The full suite passes with Ruff, mypy, compilation, and diff checks.

## Day 4: Product Contract, Workspace, and Termination

Date: 2026-06-11

### Core Question

What does a coding agent own, and what does it mean for a coding task to stop?

### Workspace Boundary

The repository workspace is the part of the filesystem that the coding agent
is authorized to inspect and modify. Relative paths are resolved from an
explicit workspace root rather than the process's accidental current
directory.

The shared path resolver:

1. resolves the workspace root
2. joins relative paths to that root
3. normalizes `.` and `..`
4. resolves symbolic links
5. rejects a final path outside the workspace

The decision is based on the resolved path, not the original path string.
Therefore, `../project/README.md` may be allowed when it resolves inside the
workspace, while an apparently internal symlink is rejected when its target
is outside.

The `read_file` tool now uses this resolver. A boundary violation becomes a
structured error observation that the model can use for recovery.

### Four Result Layers

A coding run has four separate result layers:

- **Provider stop reason**: why the model provider stopped one generation,
  such as `end_turn`, `tool_use`, or `max_tokens`.
- **Runtime termination**: why the controller stopped the complete run, such
  as `completed`, `max_steps`, or `unexpected_stop`.
- **Verification evidence**: an observed check result such as `not_run`,
  `passed`, `failed`, or `error`, together with its command and output when
  applicable.
- **Task success**: whether the complete user objective is known to have been
  satisfied.

These values must not be inferred from each other without sufficient
evidence. For example:

```text
provider stop reason: end_turn
runtime termination: completed
verification: failed
task success: false
```

This combination is valid because the model and controller may stop normally
even though a test failed.

`AgentRun` now preserves the final provider stop reason and keeps
verification evidence and task success separate. Until the product has a
command tool or external evaluator, normal runs use:

```text
verification: not_run
task success: unknown
```

### Acceptance Scenarios

Verification must match the task's acceptance conditions rather than always
mean running `pytest`.

For a Python code fix:

- the intended behavior is changed
- the relevant tests are executed
- the test exit code is zero
- the changes remain inside the workspace

For a README-only task:

- the requested documentation is present
- the diff contains the README change
- the diff contains no source-code changes

For a code-location task:

- the correct file path is returned
- the symbol's line number is returned
- search or file-read evidence supports the answer
- no files are modified

These scenarios are product and evaluation specifications. Future command,
diff, and evaluation capabilities will produce the evidence needed to apply
them.

### Interview Answers

**Why does a coding agent need a workspace boundary?**

The filesystem is larger than the environment the user authorized for the
task. A workspace boundary prevents tools from reading or modifying unrelated
projects, credentials, system files, or paths reached through symbolic links.
The boundary must be enforced by tools, not only requested in a system prompt.

**What is the difference between provider stop reason and runtime
termination?**

The provider stop reason describes one model generation. Runtime termination
describes the controller's complete run. A final `tool_use` response can be
followed by runtime termination `max_steps`, while `max_tokens` without a tool
result produces `unexpected_stop`.

**Why does `end_turn` not prove a bug is fixed?**

`end_turn` only means the model considered its current generation complete.
The model may not have run verification, may have misunderstood an
observation, or may claim completion after a failed test. Task success
requires independent evidence matched to the user's acceptance conditions.

**Why does passed verification not always prove task success?**

A check may cover only part of the objective. A focused test can pass while a
required documentation update is missing, or while another acceptance
condition remains unchecked. Verification records observed facts; task
success evaluates the complete objective.

### Verification

- Workspace-relative and internal absolute paths are accepted.
- Parent-path and symbolic-link workspace escapes are rejected.
- `read_file` enforces the shared workspace boundary.
- Completed, maximum-step, and unexpected-stop runs preserve their final
  provider stop reason.
- A completed run can contain failed verification evidence.
- The full suite passes with Ruff, mypy, compilation, and diff checks.

## Day 5: Repository Inspection Tools

Date: 2026-06-22

### Core Question

How should a coding agent gather enough repository context without reading
everything?

### Key Concepts

- Repository inspection is context selection. The agent must choose which
  files, symbols, and line ranges are relevant to the next decision.
- Path search answers "which files might matter?"
- Content search answers "where does this symbol or text appear?"
- Line-oriented reading answers "what local context is around this location?"
- Observations must be bounded so that the model receives useful evidence
  without flooding the context window.
- Reading more is not always better. Large observations can hide the relevant
  information inside noise.

### Tool Responsibilities

`glob_files` searches file paths:

```text
glob_files(pattern="tests/test_*.py")
-> tests/test_agent.py
-> tests/test_tools.py
```

It is useful when the task names a file shape, extension, or directory.

`search_text` searches file contents:

```text
search_text(pattern="async def run", file_pattern="agent/*.py")
-> agent/agent.py:44:     async def run(...)
```

It is useful when the task names a function, class, symbol, error message, or
code fragment.

`read_file` reads a bounded line range:

```text
read_file(path="agent/agent.py", offset=35, limit=45)
-> numbered lines around Agent.run
```

It is useful after the agent has a candidate path and location.

### Exploration Policy

A typical repository-inspection trajectory is:

```text
glob_files or search_text
-> identify candidate files and line numbers
-> read_file with offset and limit
-> expand the range only if the context is insufficient
```

The agent should prefer:

- `search_text` when it knows a symbol or code fragment
- `glob_files` when it knows a path pattern
- `read_file` when it knows the concrete file and approximate location

### Bounded Observations

The repository tools enforce bounds:

- `read_file` uses `offset` and `limit`, returns line numbers, and caps reads
  at 500 lines.
- `glob_files` uses `max_results` and reports truncation.
- `search_text` uses `max_matches` and reports truncation.
- Repository tools skip noisy directories such as `.git`, `.venv`,
  `node_modules`, `build`, and `dist`.
- Workspace boundaries still apply to all local file tools.

The goal is not to guarantee a complete semantic unit in one read. The goal
is to give the model enough local evidence for the next step while keeping
context manageable.

### Experiment

Locate `resolve_workspace_path`:

```text
glob_files(pattern="agent/*.py", max_results=20)
-> agent/__init__.py
-> agent/agent.py
-> ...
-> agent/workspace.py

search_text(
    pattern="def resolve_workspace_path",
    file_pattern="agent/*.py",
)
-> agent/workspace.py:4: def resolve_workspace_path(...)

read_file(path="agent/workspace.py", offset=1, limit=20)
-> lines 1-15 containing the resolver implementation
```

Locate `Agent.run`:

```text
search_text(
    pattern="class Agent|async def run",
    file_pattern="agent/*.py",
)
-> agent/agent.py:15: class Agent:
-> agent/agent.py:44:     async def run(...)
-> agent/schemas.py:113: class AgentStep(...)
-> agent/schemas.py:128: class AgentRun(...)

read_file(path="agent/agent.py", offset=35, limit=45)
-> bounded local context around Agent.run
```

Broad path search:

```text
glob_files(pattern="**/*.py", max_results=8)
-> returns 8 paths
-> [truncated after 8 files]
```

This demonstrates why broad listing is a map, not sufficient context.

### Implementation Result

The repository inspection tools now include:

- `read_file(path, offset=1, limit=200)`
- `glob_files(pattern, max_results=50)`
- `search_text(pattern, file_pattern="**/*", max_matches=50)`

The tools remain in `agent/tools.py` for now. Splitting each tool into a
separate file would add import and registration overhead before the codebase
has enough tool complexity to justify it.

### Interview Answers

**Why is repository search part of context selection?**

The model cannot reason over files it has not seen, but it also cannot
efficiently use an entire repository dumped into context. Repository search
selects the relevant subset of the environment for the next policy decision.

**What is the difference between glob and grep-style search?**

Glob searches paths and filenames. Grep-style content search looks inside
files and returns matching lines. Glob can find candidate files such as
`tests/test_*.py`; content search can locate a symbol such as
`async def run`.

**Why must tool observations be bounded?**

Unbounded observations can exceed the context window, increase cost, and bury
important evidence inside noise. Bounded observations let the controller keep
state useful and let the model request more context only when needed.

**Why include line numbers in `read_file` and `search_text` output?**

Line numbers make observations actionable. They let the model ask for nearby
context, discuss precise locations with the user, and later connect search
results to targeted edits.

### Verification

- Line-range reads return numbered output.
- Invalid line ranges are rejected by Pydantic.
- Glob patterns return relative paths and skip noisy directories.
- Glob and text search enforce result limits and truncation messages.
- Regex errors become explicit tool error observations.
- Workspace escape attempts are rejected.
- The full suite passes with Ruff, mypy, compilation, and diff checks.

## Day 6: Safe File Editing and Diff Tracking

Date: 2026-06-23

### Core Question

How can the agent change code in a way that is precise, reviewable, and
recoverable?

### Key Concepts

- File writes are environment mutations, not just text output.
- `edit_file` is a targeted mutation: it replaces one exact, unique text
  match.
- `write_file` is a stronger mutation: it creates a whole file or
  intentionally overwrites an existing file.
- Read-before-edit is an agent policy. The agent must observe a file before
  modifying it.
- A unified diff is the observation that makes file mutation reviewable.
- Session diff requires state beyond the latest tool result. The registry
  tracks changed files and the original content before the first mutation.

### Tool Responsibilities

`edit_file` replaces one exact match:

```text
read_file(path="agent/tools.py")
-> observation with current file content

edit_file(
    path="agent/tools.py",
    old_text="old exact text",
    new_text="new exact text",
)
-> unified diff for that edit
```

The replacement is rejected when the text appears zero times or more than
once. This prevents stale context and ambiguous edits from silently changing
the wrong location.

`write_file` creates a new file by default:

```text
write_file(path="docs/note.md", content="# Note\n")
-> creates parent directories and returns a diff
```

When the target file already exists, `write_file` rejects the action unless
`overwrite=true`. Overwriting an existing file also requires a prior
successful `read_file` for that path.

`get_diff` returns the session-level review surface:

```text
get_diff()
-> unified diffs for files changed during this session

get_diff(path="docs/note.md")
-> unified diff for one changed file
```

### Implementation Result

The file mutation slice now includes:

- `EditFileInput`
- `WriteFileInput`
- `GetDiffInput`
- `edit_file`
- `write_file`
- `get_diff`
- `ToolRegistry.read_files`
- `ToolRegistry.changed_files`
- `ToolRegistry.original_file_contents`
- CLI `/diff`

`ToolRegistry` enforces the session-aware policy:

- successful `read_file` records a resolved path in `read_files`
- `edit_file` requires the file to be in `read_files`
- `write_file(overwrite=true)` requires the existing file to be in
  `read_files`
- successful `edit_file` and `write_file` record the file in `changed_files`
- the first successful mutation snapshots the original file content

The snapshot is taken before the first mutation only. Therefore, multiple
edits to the same file still produce a session diff from the original file to
the current file, not merely from the previous edit to the latest edit.

### Failure Cases

- Editing before reading the file is rejected.
- Exact replacement with zero matches is rejected.
- Exact replacement with duplicate matches is rejected.
- Creating a file outside the workspace is rejected.
- Overwriting an existing file without `overwrite=true` is rejected.
- Overwriting an existing file before reading it is rejected.
- Requesting a diff for an unchanged path is rejected.

### Interview Answers

**Why is exact replacement safer than line-number editing?**

Line numbers can become stale when files change. Exact replacement requires
the observed text to still exist and to identify one location. If the text is
missing or ambiguous, the tool returns an error observation instead of
guessing.

**Why do writes need stronger policy than reads?**

Reads only observe the environment. Writes mutate it and can overwrite user
work, create incorrect files, or make unrelated changes. Therefore writes
need workspace boundaries, read-before-edit checks, explicit overwrite
intent, and reviewable diff observations.

**How do edit observations affect the next decision?**

The diff becomes part of the agent state. The model can inspect what changed,
decide whether another edit is needed, run verification, or report the
change to the user. The diff also separates tool execution success from task
success.

### Verification

- Unique, missing, and duplicate exact replacements are covered by focused
  tests.
- New-file creation, parent-directory creation, overwrite policy, and
  workspace escape rejection are covered by focused tests.
- Read-before-edit and changed-file tracking are covered by focused tests.
- Session diff, path-filtered diff, new-file diff, and multiple-edit
  original snapshots are covered by focused tests.
- CLI `/diff` is covered by focused tests.
- The full suite passes with Ruff, mypy, compilation, and diff checks.
