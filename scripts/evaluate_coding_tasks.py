# ruff: noqa: I001
import argparse
import ast
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.agent import Agent
from agent.provider import ProviderAdapter, create_provider_adapter, load_provider_config
from agent.schemas import (
    AgentRun,
    ProviderCapabilities,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agent.security import CommandPolicyResult
from agent.setup import create_registry
from agent.tool_registry import ToolRegistry

PYTHON = sys.executable
EvaluationMode = Literal["scripted", "real_model", "swe_bench"]
FailureReason = Literal[
    "compile_error",
    "test_failure",
    "max_step",
    "unsafe_command_blocked",
]
FAILURE_REASONS: tuple[FailureReason, ...] = (
    "compile_error",
    "test_failure",
    "max_step",
    "unsafe_command_blocked",
)


class ScriptedStep(BaseModel):
    text: str | None = None
    tool_call: ToolCall | None = None
    input_tokens: int = Field(default=20, ge=0)
    output_tokens: int = Field(default=8, ge=0)


class CodingTaskCase(BaseModel):
    name: str
    task: str
    acceptance_criteria: list[str]
    expected_evidence: list[str]
    scripted_steps: list[ScriptedStep]


class CodingTaskResult(BaseModel):
    name: str
    mode: EvaluationMode
    task_success: bool
    runtime_success: bool
    verification_success: bool
    recovery_success: bool
    tool_accuracy: bool
    steps: int
    tool_calls: int
    tools: list[str]
    commands: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    failures: list[str] = Field(default_factory=list)
    failure_reasons: list[FailureReason] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    total: int
    passed: int
    pass_rate: float
    average_steps: float
    average_token_cost: float
    average_tool_calls: float
    failure_reason_counts: dict[FailureReason, int]


class SweBenchInstance(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str = ""
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)


class ScriptedProviderAdapter:
    """Deterministic provider used for local coding-agent evaluations."""

    provider = "fake"
    model = "claude-haiku-4-5"
    capabilities = ProviderCapabilities()

    def __init__(self, steps: list[ScriptedStep]) -> None:
        self._steps = steps
        self._index = 0

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        del system, tools, messages
        if self._index >= len(self._steps):
            raise RuntimeError("Scripted provider ran out of responses.")

        step = self._steps[self._index]
        self._index += 1
        if step.tool_call is not None:
            content = [
                {
                    "type": "tool_use",
                    "id": step.tool_call.tool_use_id,
                    "name": step.tool_call.name,
                    "input": step.tool_call.input,
                }
            ]
            return ProviderResponse(
                message={"role": "assistant", "content": content},
                stop_reason="tool_use",
                tool_calls=[step.tool_call],
                usage=TokenUsage(
                    input_tokens=step.input_tokens,
                    output_tokens=step.output_tokens,
                ),
                native_metadata={
                    "provider": self.provider,
                    "script_index": self._index,
                },
            )

        text = step.text or ""
        if on_text_delta is not None and text:
            on_text_delta(text)
        return ProviderResponse(
            message={
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
            stop_reason="end_turn",
            text=[text] if text else [],
            usage=TokenUsage(
                input_tokens=step.input_tokens,
                output_tokens=step.output_tokens,
            ),
            native_metadata={"provider": self.provider, "script_index": self._index},
        )

    def tool_result_message(self, tool_results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": result.type,
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in tool_results
            ],
        }


Oracle = Callable[[Path, AgentRun, Agent], list[str]]
FixtureBuilder = Callable[[Path], None]


def tool_call(name: str, input_data: dict[str, Any], tool_use_id: str) -> ScriptedStep:
    return ScriptedStep(
        tool_call=ToolCall(name=name, input=input_data, tool_use_id=tool_use_id)
    )


def final_text(text: str) -> ScriptedStep:
    return ScriptedStep(text=text)


def build_cases() -> dict[str, tuple[CodingTaskCase, FixtureBuilder, Oracle]]:
    return {
        "repository_search": (
            CodingTaskCase(
                name="repository_search",
                task=(
                    "Find where token usage is tracked and explain which file owns "
                    "estimated cost calculation."
                ),
                acceptance_criteria=[
                    "The agent searches repository content.",
                    "The agent reads the relevant file.",
                    "The final answer names the token tracker owner.",
                    "No file edits are made.",
                ],
                expected_evidence=[
                    "search_text was used.",
                    "read_file was used on agent/token_tracker.py.",
                    "changed files list is empty.",
                ],
                scripted_steps=[
                    tool_call(
                        "search_text",
                        {"pattern": "estimated_cost", "file_pattern": "**/*.py"},
                        "toolu_search_cost",
                    ),
                    tool_call(
                        "read_file",
                        {"path": "agent/token_tracker.py", "offset": 1, "limit": 120},
                        "toolu_read_token_tracker",
                    ),
                    final_text(
                        "Estimated cost is owned by agent/token_tracker.py in "
                        "TokenTracker."
                    ),
                ],
            ),
            fixture_repository_search,
            oracle_repository_search,
        ),
        "small_bug_fix": (
            CodingTaskCase(
                name="small_bug_fix",
                task=(
                    "Fix a bug where add(2, 3) incorrectly returns 6 instead of 5. "
                    "After editing, run the focused test that verifies the behavior. "
                    "Do not create new files."
                ),
                acceptance_criteria=[
                    "The source file is read before editing.",
                    "The add function returns a + b.",
                    "The focused pytest command passes.",
                    "Only the intended source file changes.",
                ],
                expected_evidence=[
                    "read_file was used before edit_file.",
                    "edit_file returned a successful diff.",
                    "run_command reported exit_code: 0.",
                ],
                scripted_steps=[
                    tool_call(
                        "read_file",
                        {"path": "calculator.py"},
                        "toolu_read_calculator",
                    ),
                    tool_call(
                        "edit_file",
                        {
                            "path": "calculator.py",
                            "old_text": "def add(a: int, b: int) -> int:\n    return a * b\n",
                            "new_text": "def add(a: int, b: int) -> int:\n    return a + b\n",
                        },
                        "toolu_fix_add",
                    ),
                    tool_call(
                        "run_command",
                        {"command": f"{PYTHON} -m pytest tests/test_calculator.py"},
                        "toolu_pytest_add",
                    ),
                    final_text("The add bug is fixed and the focused test passes."),
                ],
            ),
            fixture_small_bug_fix,
            oracle_small_bug_fix,
        ),
        "targeted_refactor": (
            CodingTaskCase(
                name="targeted_refactor",
                task=(
                    "Rename format_user_name to format_display_name and update "
                    "local references. After editing, run the focused test that "
                    "verifies the behavior. Do not create new files."
                ),
                acceptance_criteria=[
                    "The old symbol is searched.",
                    "The definition and call sites are updated.",
                    "The old symbol no longer appears in Python source.",
                    "The focused pytest command passes.",
                ],
                expected_evidence=[
                    "search_text found old references.",
                    "edit_file updated the module.",
                    "run_command reported exit_code: 0.",
                ],
                scripted_steps=[
                    tool_call(
                        "search_text",
                        {"pattern": "format_user_name", "file_pattern": "**/*.py"},
                        "toolu_search_old_name",
                    ),
                    tool_call(
                        "read_file",
                        {"path": "users.py"},
                        "toolu_read_users",
                    ),
                    tool_call(
                        "edit_file",
                        {
                            "path": "users.py",
                            "old_text": (
                                "def format_user_name(first: str, last: str) -> str:\n"
                                '    return f"{first} {last}"\n\n\n'
                                "def greeting(first: str, last: str) -> str:\n"
                                '    return f"Hello, {format_user_name(first, last)}!"\n'
                            ),
                            "new_text": (
                                "def format_display_name(first: str, last: str) -> str:\n"
                                '    return f"{first} {last}"\n\n\n'
                                "def greeting(first: str, last: str) -> str:\n"
                                '    return f"Hello, {format_display_name(first, last)}!"\n'
                            ),
                        },
                        "toolu_refactor_name",
                    ),
                    tool_call(
                        "run_command",
                        {"command": f"{PYTHON} -m pytest tests/test_users.py"},
                        "toolu_pytest_users",
                    ),
                    final_text("The rename is complete and tests pass."),
                ],
            ),
            fixture_targeted_refactor,
            oracle_targeted_refactor,
        ),
        "failed_edit_recovery": (
            CodingTaskCase(
                name="failed_edit_recovery",
                task=(
                    "Fix a README typo by changing instalation to installation, "
                    "recovering from a failed edit if needed. Check the final diff."
                ),
                acceptance_criteria=[
                    "One edit attempt fails.",
                    "The agent rereads or searches before retrying.",
                    "A later edit succeeds.",
                    "The README contains the corrected word.",
                ],
                expected_evidence=[
                    "trace contains a failed edit_file result.",
                    "read_file occurs after the failed edit.",
                    "trace contains a later successful edit_file result.",
                ],
                scripted_steps=[
                    tool_call(
                        "read_file",
                        {"path": "README.md"},
                        "toolu_read_readme",
                    ),
                    tool_call(
                        "edit_file",
                        {
                            "path": "README.md",
                            "old_text": "installation guide",
                            "new_text": "installation guide",
                        },
                        "toolu_bad_readme_edit",
                    ),
                    tool_call(
                        "read_file",
                        {"path": "README.md"},
                        "toolu_reread_readme",
                    ),
                    tool_call(
                        "edit_file",
                        {
                            "path": "README.md",
                            "old_text": "Follow the instalation guide.\n",
                            "new_text": "Follow the installation guide.\n",
                        },
                        "toolu_good_readme_edit",
                    ),
                    final_text("Recovered from the failed edit and fixed the typo."),
                ],
            ),
            fixture_failed_edit_recovery,
            oracle_failed_edit_recovery,
        ),
        "failed_test_recovery": (
            CodingTaskCase(
                name="failed_test_recovery",
                task="Fix module.py after a failed verification command.",
                acceptance_criteria=[
                    "The first verification command fails.",
                    "The agent edits after observing the failure.",
                    "The later verification command passes.",
                    "The final file is syntactically valid.",
                ],
                expected_evidence=[
                    "run_command first reports exit_code: 1.",
                    "edit_file repairs module.py.",
                    "run_command later reports exit_code: 0.",
                ],
                scripted_steps=[
                    tool_call(
                        "read_file",
                        {"path": "module.py"},
                        "toolu_read_module",
                    ),
                    tool_call(
                        "run_command",
                        {"command": f"{PYTHON} -m py_compile module.py"},
                        "toolu_failed_compile",
                    ),
                    tool_call(
                        "edit_file",
                        {
                            "path": "module.py",
                            "old_text": "def answer()\n    return 1\n",
                            "new_text": "def answer() -> int:\n    return 1\n",
                        },
                        "toolu_fix_syntax",
                    ),
                    tool_call(
                        "run_command",
                        {"command": f"{PYTHON} -m py_compile module.py"},
                        "toolu_passing_compile",
                    ),
                    final_text("The syntax error is fixed and verification passes."),
                ],
            ),
            fixture_failed_test_recovery,
            oracle_failed_test_recovery,
        ),
    }


def fixture_repository_search(workspace: Path) -> None:
    package = workspace / "agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "token_tracker.py").write_text(
        "\n".join(
            [
                "class TokenTracker:",
                "    def __init__(self) -> None:",
                "        self.input_tokens = 0",
                "        self.output_tokens = 0",
                "        self._estimated_cost = 0.0",
                "",
                "    @property",
                "    def estimated_cost(self) -> float:",
                "        return self._estimated_cost",
                "",
            ]
        ),
        encoding="utf-8",
    )


def fixture_small_bug_fix(workspace: Path) -> None:
    (workspace / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a * b\n",
        encoding="utf-8",
    )
    tests = workspace / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "from calculator import add\n\n\n"
        "def test_adds_two_numbers() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )


def fixture_targeted_refactor(workspace: Path) -> None:
    (workspace / "users.py").write_text(
        "def format_user_name(first: str, last: str) -> str:\n"
        '    return f"{first} {last}"\n\n\n'
        "def greeting(first: str, last: str) -> str:\n"
        '    return f"Hello, {format_user_name(first, last)}!"\n',
        encoding="utf-8",
    )
    tests = workspace / "tests"
    tests.mkdir()
    (tests / "test_users.py").write_text(
        "from users import greeting\n\n\n"
        "def test_greeting_uses_display_name() -> None:\n"
        "    assert greeting('Ada', 'Lovelace') == 'Hello, Ada Lovelace!'\n",
        encoding="utf-8",
    )


def fixture_failed_edit_recovery(workspace: Path) -> None:
    (workspace / "README.md").write_text(
        "# Demo\n\nFollow the instalation guide.\n",
        encoding="utf-8",
    )


def fixture_failed_test_recovery(workspace: Path) -> None:
    (workspace / "module.py").write_text(
        "def answer()\n    return 1\n",
        encoding="utf-8",
    )


def oracle_repository_search(workspace: Path, run: AgentRun, agent: Agent) -> list[str]:
    failures: list[str] = []
    calls = tool_names(run)
    final = final_answer(run)
    if not any(name in calls for name in {"search_text", "glob_files"}):
        failures.append("expected repository search or globbing")
    if "read_file" not in calls:
        failures.append("expected read_file on the relevant file")
    if "TokenTracker" not in final or "agent/token_tracker.py" not in final:
        failures.append("final answer did not identify TokenTracker owner")
    if agent.registry.changed_files:
        failures.append("repository search should not change files")
    if not (workspace / "agent" / "token_tracker.py").exists():
        failures.append("fixture file disappeared")
    return failures


def oracle_small_bug_fix(workspace: Path, run: AgentRun, agent: Agent) -> list[str]:
    failures: list[str] = []
    content = (workspace / "calculator.py").read_text(encoding="utf-8")
    if "return a + b" not in content:
        failures.append("calculator.py does not add the inputs")
    if "return a * b" in content:
        failures.append("calculator.py still multiplies the inputs")
    if not oracle_command_passed(
        workspace,
        [PYTHON, "-m", "pytest", "tests/test_calculator.py"],
    ):
        failures.append("external pytest oracle did not pass")
    if relative_changed_files(workspace, agent.registry) != ["calculator.py"]:
        failures.append("changed files were not limited to calculator.py")
    return failures


def oracle_targeted_refactor(workspace: Path, run: AgentRun, agent: Agent) -> list[str]:
    del agent
    failures: list[str] = []
    content = (workspace / "users.py").read_text(encoding="utf-8")
    if "format_display_name" not in content:
        failures.append("new symbol is missing")
    if "format_user_name" in content:
        failures.append("old symbol still appears in users.py")
    if not oracle_command_passed(workspace, [PYTHON, "-m", "pytest", "tests/test_users.py"]):
        failures.append("external pytest oracle did not pass")
    return failures


def oracle_failed_edit_recovery(
    workspace: Path,
    run: AgentRun,
    agent: Agent,
) -> list[str]:
    del agent
    failures: list[str] = []
    results = [
        (tool_call.name, tool_result.is_error)
        for step in run.steps
        for tool_call, tool_result in zip(step.tool_calls, step.tool_results)
    ]
    if ("edit_file", False) not in results:
        failures.append("missing successful edit_file observation")
    content = (workspace / "README.md").read_text(encoding="utf-8")
    if "installation" not in content or "instalation" in content:
        failures.append("README typo was not corrected")
    return failures


def oracle_failed_test_recovery(
    workspace: Path,
    run: AgentRun,
    agent: Agent,
) -> list[str]:
    del agent
    failures: list[str] = []
    if not oracle_command_passed(workspace, [PYTHON, "-m", "py_compile", "module.py"]):
        failures.append("external py_compile oracle did not pass")
    return failures


def oracle_command_passed(workspace: Path, command: list[str]) -> bool:
    completed = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return completed.returncode == 0


async def evaluate_case(
    case: CodingTaskCase,
    fixture_builder: FixtureBuilder,
    oracle: Oracle,
    *,
    mode: EvaluationMode,
    provider_adapter: ProviderAdapter | None,
    keep_workspace: bool,
    max_steps: int | None = None,
) -> CodingTaskResult:
    workspace = Path(tempfile.mkdtemp(prefix=f"agent-eval-{case.name}-"))
    original_path = os.environ.get("PATH", "")
    started = perf_counter()
    try:
        fixture_builder(workspace)
        add_python_shim(workspace)
        os.environ["PATH"] = (
            f"{(workspace / '.venv' / 'bin').as_posix()}{os.pathsep}{original_path}"
        )
        registry = create_registry(workspace)
        provider = provider_adapter or ScriptedProviderAdapter(case.scripted_steps)
        agent = Agent(
            provider_adapter=provider,
            registry=registry,
            stream_output=False,
            max_steps=max_steps or max(10, len(case.scripted_steps) + 1),
            approval_callback=approve_eval_diagnostic_command,
        )
        run = await agent.run(case.task)
        latency_ms = (perf_counter() - started) * 1000
        failures = oracle(workspace, run, agent)
        task_success = not failures
        failure_reasons = (
            [] if task_success else classify_failure_reasons(run, failures)
        )
        return CodingTaskResult(
            name=case.name,
            mode=mode,
            task_success=task_success,
            runtime_success=run.termination == "completed",
            verification_success=run.verification.status == "passed",
            recovery_success=recovered_from_error(run),
            tool_accuracy=tool_accuracy(case, run, mode),
            steps=len(run.steps),
            tool_calls=sum(len(step.tool_calls) for step in run.steps),
            tools=tool_names(run),
            commands=command_summaries(run),
            latency_ms=latency_ms,
            input_tokens=agent.token_tracker.input_tokens,
            output_tokens=agent.token_tracker.output_tokens,
            estimated_cost=agent.token_tracker.estimated_cost,
            failures=failures,
            failure_reasons=failure_reasons,
        )
    finally:
        os.environ["PATH"] = original_path
        if keep_workspace:
            print(f"Kept workspace for {case.name}: {workspace}")
        else:
            shutil.rmtree(workspace)


def add_python_shim(workspace: Path) -> None:
    bin_dir = workspace / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for executable in ("python", "python3"):
        python_path = bin_dir / executable
        try:
            python_path.symlink_to(PYTHON)
        except OSError:
            python_path.write_text(
                "#!/bin/sh\n"
                f"exec {PYTHON} \"$@\"\n",
                encoding="utf-8",
            )
            python_path.chmod(0o755)


def approve_eval_diagnostic_command(
    tool_call: ToolCall,
    policy: CommandPolicyResult,
) -> bool:
    del tool_call
    args = policy.args
    if not args:
        return False
    executable = Path(args[0]).name
    if executable in {"python", "python3"}:
        return len(args) >= 2 and args[1] in {"-c", "--version"}
    return executable in {"which", "ls"}


def recovered_from_error(run: AgentRun) -> bool:
    saw_failure_observation = False
    for step in run.steps:
        for tool_call, tool_result in zip(step.tool_calls, step.tool_results):
            if observation_failed(tool_call, tool_result):
                saw_failure_observation = True
            elif saw_failure_observation:
                return True
    return False


def observation_failed(tool_call: ToolCall, tool_result: ToolResult) -> bool:
    if tool_result.is_error:
        return True
    if tool_call.name != "run_command":
        return False
    return "exit_code: 0" not in tool_result.content


def tool_accuracy(case: CodingTaskCase, run: AgentRun, mode: EvaluationMode) -> bool:
    if mode == "scripted":
        return tool_sequence_matches_script(case, run)
    return expected_tools_were_used(case, run)


def tool_sequence_matches_script(case: CodingTaskCase, run: AgentRun) -> bool:
    expected = [
        step.tool_call.name
        for step in case.scripted_steps
        if step.tool_call is not None
    ]
    return tool_names(run) == expected


def expected_tools_were_used(case: CodingTaskCase, run: AgentRun) -> bool:
    expected = {
        step.tool_call.name
        for step in case.scripted_steps
        if step.tool_call is not None
    }
    actual = set(tool_names(run))
    return expected.issubset(actual)


def tool_names(run: AgentRun) -> list[str]:
    return [tool_call.name for step in run.steps for tool_call in step.tool_calls]


def command_summaries(run: AgentRun) -> list[str]:
    summaries: list[str] = []
    for step in run.steps:
        for tool_call, tool_result in zip(step.tool_calls, step.tool_results):
            if tool_call.name != "run_command":
                continue
            command = tool_call.input.get("command")
            command_text = command if isinstance(command, str) else "[unknown]"
            exit_code = field_value(tool_result.content, "exit_code") or "error"
            timed_out = field_value(tool_result.content, "timed_out") or "unknown"
            summaries.append(
                f"{command_text} -> exit_code={exit_code}, timed_out={timed_out}"
            )
    return summaries


def field_value(output: str, field: str) -> str | None:
    prefix = f"{field}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return None


def classify_failure_reasons(
    run: AgentRun,
    failures: list[str],
) -> list[FailureReason]:
    reasons: set[FailureReason] = set()
    if run.termination == "max_steps":
        reasons.add("max_step")

    for tool_call, tool_result in iter_tool_results(run):
        if tool_call.name != "run_command":
            continue
        if command_was_blocked(tool_result):
            reasons.add("unsafe_command_blocked")
        if not command_failed(tool_result):
            continue
        command = command_text(tool_call)
        output = tool_result.content
        if command_is_compile(command) or output_has_compile_error(output):
            reasons.add("compile_error")
        elif command_is_test(command):
            reasons.add("test_failure")

    for failure in failures:
        normalized = failure.lower()
        if any(marker in normalized for marker in ("py_compile", "syntax", "compile")):
            reasons.add("compile_error")
        if any(marker in normalized for marker in ("pytest", "test failure")):
            reasons.add("test_failure")

    return [reason for reason in FAILURE_REASONS if reason in reasons]


def iter_tool_results(run: AgentRun) -> list[tuple[ToolCall, ToolResult]]:
    return [
        (tool_call, tool_result)
        for step in run.steps
        for tool_call, tool_result in zip(step.tool_calls, step.tool_results)
    ]


def command_text(tool_call: ToolCall) -> str:
    command = tool_call.input.get("command")
    if isinstance(command, str):
        return command
    return ""


def command_failed(tool_result: ToolResult) -> bool:
    if tool_result.is_error:
        return True
    timed_out = field_value(tool_result.content, "timed_out")
    if timed_out == "true":
        return True
    exit_code = field_value(tool_result.content, "exit_code")
    return exit_code not in {None, "0"}


def command_was_blocked(tool_result: ToolResult) -> bool:
    if not tool_result.is_error:
        return False
    normalized = tool_result.content.lower()
    return any(
        marker in normalized
        for marker in (
            "approval denied",
            "requires approval",
            "blocked dangerous command",
            "shell operators are not supported",
            "command substitution is not supported",
        )
    )


def command_is_compile(command: str) -> bool:
    return "py_compile" in command


def command_is_test(command: str) -> bool:
    return "pytest" in command or "unittest" in command


def output_has_compile_error(output: str) -> bool:
    normalized = output.lower()
    return any(
        marker in normalized
        for marker in ("syntaxerror", "indentationerror", "taberror")
    )


def final_answer(run: AgentRun) -> str:
    for step in reversed(run.steps):
        if step.text:
            return "\n".join(step.text)
    return ""


def relative_changed_files(workspace: Path, registry: ToolRegistry) -> list[str]:
    root = workspace.resolve()
    return sorted(
        path.resolve().relative_to(root).as_posix() for path in registry.changed_files
    )


def create_real_provider_adapter(api_key: str | None = None) -> ProviderAdapter:
    config = load_provider_config(api_key=api_key)
    return create_provider_adapter(config)


async def evaluate_cases(
    selected_names: list[str],
    *,
    mode: EvaluationMode,
    keep_workspaces: bool,
    api_key: str | None = None,
    max_steps: int | None = None,
) -> list[CodingTaskResult]:
    cases = build_cases()
    results: list[CodingTaskResult] = []
    for name in selected_names:
        case, fixture_builder, oracle = cases[name]
        provider_adapter = (
            create_real_provider_adapter(api_key) if mode == "real_model" else None
        )
        results.append(
            await evaluate_case(
                case,
                fixture_builder,
                oracle,
                mode=mode,
                provider_adapter=provider_adapter,
                keep_workspace=keep_workspaces,
                max_steps=max_steps,
            )
        )
    return results


def print_results(results: list[CodingTaskResult]) -> None:
    mode = results[0].mode if results else "scripted"
    if mode == "real_model":
        title = "Coding-agent real-model evaluation"
    elif mode == "swe_bench":
        title = "Coding-agent SWE-bench evaluation"
    else:
        title = "Coding-agent deterministic evaluation"
    print(title)
    print("=" * len(title))
    for result in results:
        status = "PASS" if result.task_success else "FAIL"
        print(f"{status} {result.name}")
        print(
            "  "
            f"runtime={result.runtime_success} "
            f"verification={result.verification_success} "
            f"recovery={result.recovery_success} "
            f"tool_accuracy={result.tool_accuracy}"
        )
        print(
            "  "
            f"steps={result.steps} "
            f"tool_calls={result.tool_calls} "
            f"latency_ms={result.latency_ms:.1f} "
            f"tokens={result.input_tokens + result.output_tokens} "
            f"cost=${result.estimated_cost:.6f}"
        )
        print(f"  tools={', '.join(result.tools) if result.tools else '[none]'}")
        for command in result.commands:
            print(f"  command={command}")
        if result.failure_reasons:
            print(f"  failure_reasons={', '.join(result.failure_reasons)}")
        for failure in result.failures:
            print(f"  - {failure}")
    summary = summarize_results(results)
    print("\nSummary")
    print(
        "  "
        f"pass_rate={summary.passed}/{summary.total} "
        f"({summary.pass_rate:.1%})"
    )
    print(f"  average_steps={summary.average_steps:.2f}")
    print(f"  average_token_cost=${summary.average_token_cost:.6f}")
    print(f"  average_tool_calls={summary.average_tool_calls:.2f}")
    print("  failure_reasons:")
    for reason in FAILURE_REASONS:
        print(f"    {reason}={summary.failure_reason_counts[reason]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local coding-agent evaluation cases."
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="Case names to run. Defaults to all implemented cases.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases and exit.",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Keep temporary workspaces for debugging.",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        help=(
            "Call the configured live provider instead of the scripted fake provider. "
            "Defaults to the read-only repository_search case when no case is named."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the agent step limit for each evaluated task.",
    )
    parser.add_argument(
        "--swe-bench",
        type=Path,
        default=None,
        help="Run SWE-bench-style instances from a JSON or JSONL file.",
    )
    parser.add_argument(
        "--swe-bench-limit",
        type=int,
        default=None,
        help="Maximum number of SWE-bench instances to run.",
    )
    parser.add_argument(
        "--swe-bench-cache",
        type=Path,
        default=Path(".agents/evals/swe-bench/repos"),
        help="Repository clone cache used for SWE-bench instances.",
    )
    parser.add_argument(
        "--swe-bench-predictions",
        type=Path,
        default=None,
        help="JSONL predictions path with instance_id, model_name_or_path, model_patch.",
    )
    parser.add_argument(
        "--skip-swe-bench-tests",
        action="store_true",
        help="Generate SWE-bench predictions without best-effort local pytest runs.",
    )
    return parser.parse_args(argv)


def load_swe_bench_instances(
    path: Path,
    *,
    limit: int | None = None,
) -> list[SweBenchInstance]:
    records = load_json_records(path)
    if limit is not None:
        records = records[:limit]
    instances = [parse_swe_bench_instance(record) for record in records]
    if not instances:
        raise SystemExit(f"No SWE-bench instances found in {path}")
    return instances


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"SWE-bench file not found: {path}")
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("instances"), list):
            records = payload["instances"]
        else:
            raise SystemExit(
                "SWE-bench JSON must be a list or an object with an instances list."
            )
    return [cast(dict[str, Any], record) for record in records]


def parse_swe_bench_instance(record: dict[str, Any]) -> SweBenchInstance:
    data = {
        "instance_id": record.get("instance_id"),
        "repo": record.get("repo"),
        "base_commit": record.get("base_commit"),
        "problem_statement": record.get("problem_statement"),
        "test_patch": record.get("test_patch") or "",
        "fail_to_pass": normalize_swe_bench_test_ids(
            record.get("FAIL_TO_PASS", record.get("fail_to_pass", []))
        ),
        "pass_to_pass": normalize_swe_bench_test_ids(
            record.get("PASS_TO_PASS", record.get("pass_to_pass", []))
        ),
    }
    return SweBenchInstance.model_validate(data)


def normalize_swe_bench_test_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return [line.strip() for line in stripped.splitlines() if line.strip()]
    return [str(value)]


async def evaluate_swe_bench_instances(
    instances: list[SweBenchInstance],
    *,
    api_key: str | None,
    keep_workspaces: bool,
    cache_root: Path,
    predictions_path: Path,
    run_tests: bool,
    max_steps: int | None,
) -> list[CodingTaskResult]:
    provider_adapter = create_real_provider_adapter(api_key)
    prepare_predictions_file(predictions_path)
    results: list[CodingTaskResult] = []
    for instance in instances:
        results.append(
            await evaluate_swe_bench_instance(
                instance,
                provider_adapter=provider_adapter,
                keep_workspace=keep_workspaces,
                cache_root=cache_root,
                predictions_path=predictions_path,
                run_tests=run_tests,
                max_steps=max_steps or 30,
            )
        )
    print(f"\nSWE-bench predictions: {predictions_path}")
    return results


async def evaluate_swe_bench_instance(
    instance: SweBenchInstance,
    *,
    provider_adapter: ProviderAdapter,
    keep_workspace: bool,
    cache_root: Path,
    predictions_path: Path,
    run_tests: bool,
    max_steps: int,
) -> CodingTaskResult:
    workspace = Path(tempfile.mkdtemp(prefix=f"agent-swe-{instance.instance_id}-"))
    started = perf_counter()
    repo_workspace = workspace / "repo"
    failures: list[str] = []
    commands: list[str] = []
    model_patch = ""
    run: AgentRun | None = None
    agent: Agent | None = None
    try:
        materialize_swe_bench_workspace(instance, repo_workspace, cache_root)
        registry = create_registry(repo_workspace)
        agent = Agent(
            provider_adapter=provider_adapter,
            registry=registry,
            stream_output=False,
            max_steps=max_steps,
            approval_callback=deny_eval_command_approval,
        )
        run = await agent.run(swe_bench_prompt(instance))
        model_patch = git_diff(repo_workspace)
        write_swe_bench_prediction(predictions_path, instance, provider_adapter, model_patch)

        if run.termination != "completed":
            failures.append(f"agent terminated with {run.termination}")
        if not model_patch.strip():
            failures.append("no model patch generated")
        if run_tests:
            test_success, test_command = run_swe_bench_tests(instance, repo_workspace)
            commands.append(test_command)
            if not test_success:
                failures.append("SWE-bench pytest command did not pass")

        latency_ms = (perf_counter() - started) * 1000
        assert run is not None
        assert agent is not None
        task_success = not failures
        failure_reasons = (
            [] if task_success else classify_failure_reasons(run, failures)
        )
        return CodingTaskResult(
            name=instance.instance_id,
            mode="swe_bench",
            task_success=task_success,
            runtime_success=run.termination == "completed",
            verification_success=run_tests and not any(
                failure == "SWE-bench pytest command did not pass"
                for failure in failures
            ),
            recovery_success=recovered_from_error(run),
            tool_accuracy=bool(model_patch.strip()),
            steps=len(run.steps),
            tool_calls=sum(len(step.tool_calls) for step in run.steps),
            tools=tool_names(run),
            commands=[*command_summaries(run), *commands],
            latency_ms=latency_ms,
            input_tokens=agent.token_tracker.input_tokens,
            output_tokens=agent.token_tracker.output_tokens,
            estimated_cost=agent.token_tracker.estimated_cost,
            failures=failures,
            failure_reasons=failure_reasons,
        )
    finally:
        if keep_workspace:
            print(f"Kept SWE-bench workspace for {instance.instance_id}: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


def prepare_predictions_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def write_swe_bench_prediction(
    path: Path,
    instance: SweBenchInstance,
    provider_adapter: ProviderAdapter,
    model_patch: str,
) -> None:
    record = {
        "instance_id": instance.instance_id,
        "model_name_or_path": f"{provider_adapter.provider}/{provider_adapter.model}",
        "model_patch": model_patch,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def swe_bench_prompt(instance: SweBenchInstance) -> str:
    return "\n".join(
        [
            "Fix this SWE-bench issue in the current repository.",
            "Make the minimal source changes needed for the described behavior.",
            "Run focused verification when practical.",
            "",
            "Problem statement:",
            instance.problem_statement,
        ]
    )


def deny_eval_command_approval(
    tool_call: ToolCall,
    policy: CommandPolicyResult,
) -> bool:
    del tool_call, policy
    return False


def materialize_swe_bench_workspace(
    instance: SweBenchInstance,
    repo_workspace: Path,
    cache_root: Path,
) -> None:
    cached_repo = ensure_swe_bench_repo_cache(instance, cache_root)
    run_subprocess(["git", "clone", cached_repo.as_posix(), repo_workspace.as_posix()])
    run_subprocess(["git", "checkout", instance.base_commit], cwd=repo_workspace)


def ensure_swe_bench_repo_cache(
    instance: SweBenchInstance,
    cache_root: Path,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_repo = cache_root / instance.repo.replace("/", "__")
    if cached_repo.exists():
        return cached_repo
    source = f"https://github.com/{instance.repo}.git"
    run_subprocess(["git", "clone", source, cached_repo.as_posix()])
    return cached_repo


def git_diff(repo_workspace: Path) -> str:
    run_subprocess(["git", "add", "-N", "."], cwd=repo_workspace)
    completed = run_subprocess(
        ["git", "diff", "--binary"],
        cwd=repo_workspace,
        check=False,
    )
    return completed.stdout


def run_swe_bench_tests(
    instance: SweBenchInstance,
    repo_workspace: Path,
) -> tuple[bool, str]:
    if instance.test_patch:
        completed = run_subprocess(
            ["git", "apply"],
            cwd=repo_workspace,
            input_text=instance.test_patch,
            check=False,
        )
        if completed.returncode != 0:
            summary = (
                "git apply <test_patch> -> "
                f"exit_code={completed.returncode}, timed_out=false"
            )
            return False, summary

    tests = [*instance.fail_to_pass, *instance.pass_to_pass]
    if not tests:
        return True, "[No SWE-bench test ids supplied]"
    command = [PYTHON, "-m", "pytest", *tests]
    completed = run_subprocess(command, cwd=repo_workspace, check=False, timeout=120)
    command_text = " ".join(command)
    return (
        completed.returncode == 0,
        f"{command_text} -> exit_code={completed.returncode}, timed_out=false",
    )


def run_subprocess(
    args: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    check: bool = True,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        command = " ".join(args)
        raise RuntimeError(
            f"Command failed: {command}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed


def default_case_names(
    mode: EvaluationMode,
    cases: dict[str, tuple[CodingTaskCase, FixtureBuilder, Oracle]],
) -> list[str]:
    if mode == "real_model":
        return ["repository_search"]
    return list(cases)


def summarize_results(results: list[CodingTaskResult]) -> EvaluationSummary:
    total = len(results)
    passed = sum(result.task_success for result in results)
    failure_reason_counts = {
        reason: sum(reason in result.failure_reasons for result in results)
        for reason in FAILURE_REASONS
    }
    return EvaluationSummary(
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        average_steps=average_float(result.steps for result in results),
        average_token_cost=average_float(result.estimated_cost for result in results),
        average_tool_calls=average_float(result.tool_calls for result in results),
        failure_reason_counts=failure_reason_counts,
    )


def average_float(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def main() -> None:
    raise SystemExit(asyncio.run(run_eval_cli()))


async def run_eval_cli(
    argv: Sequence[str] | None = None,
    *,
    api_key: str | None = None,
) -> int:
    args = parse_args(argv)
    cases = build_cases()
    if args.list:
        for name, (case, _, _) in cases.items():
            print(f"{name}: {case.task}")
        return 0

    if args.swe_bench is not None:
        load_dotenv()
        if args.cases:
            joined = ", ".join(args.cases)
            raise SystemExit(
                f"Do not pass built-in case names with --swe-bench: {joined}"
            )
        instances = load_swe_bench_instances(
            args.swe_bench,
            limit=args.swe_bench_limit,
        )
        predictions_path = args.swe_bench_predictions or Path(
            ".agents/evals/swe-bench-predictions.jsonl"
        )
        results = await evaluate_swe_bench_instances(
            instances,
            api_key=api_key,
            keep_workspaces=args.keep_workspaces,
            cache_root=args.swe_bench_cache,
            predictions_path=predictions_path,
            run_tests=not args.skip_swe_bench_tests,
            max_steps=args.max_steps,
        )
        print_results(results)
        if not all(result.task_success for result in results):
            return 1
        return 0

    mode: EvaluationMode = "real_model" if args.real_model else "scripted"
    if mode == "real_model":
        load_dotenv()

    selected_names = args.cases or default_case_names(mode, cases)
    unknown = sorted(set(selected_names) - set(cases))
    if unknown:
        joined = ", ".join(unknown)
        raise SystemExit(f"Unknown evaluation case(s): {joined}")

    results = await evaluate_cases(
        selected_names,
        mode=mode,
        keep_workspaces=args.keep_workspaces,
        api_key=api_key,
        max_steps=args.max_steps,
    )
    print_results(results)
    if not all(result.task_success for result in results):
        return 1
    return 0


if __name__ == "__main__":
    main()
