# ruff: noqa: E402
"""Small local, live-model, and SWE-bench-compatible agent evaluations."""

import argparse
import asyncio
import json
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
from agent.provider import DeepSeekProvider, ProviderAdapter, load_deepseek_config
from agent.schemas import (
    AgentRun,
    ProviderCapabilities,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agent.security import ToolApprovalPolicy
from agent.setup import create_registry

PYTHON = sys.executable
EvaluationMode = Literal["deterministic", "real_model", "swe_bench"]
SWE_BENCH_FINALIZATION_STEPS = 2
SWE_BENCH_SYSTEM_PROMPT_SUFFIX = """
## SWE-bench patch-generation profile

These profile instructions override the generic verification and recovery rules
below when they conflict.

Treat verification as best-effort because this checkout may not contain the
benchmark's dependency environment. After an edit, run one focused existing
verification command when practical. If it fails because dependencies are
missing, the interpreter or platform is incompatible, or the repository needs
external environment setup, treat that as an environment block, stop
verification, review the diff, and finish. Do not try to recover from an
environment block inside the repository.
Do not modify product code or repository configuration to repair the evaluation
environment.

Do not create dependency shims, any conftest.py, pytest.ini, temporary
verification scripts, scratch files, or paths beginning with _tmp. In
particular, never add distutils, pytz, or asgiref replacements. Do not install
dependencies. Use the existing repository files only, except for changes that
directly implement the reported issue and focused regression tests that belong
in the project's existing test suite.
"""


class ScriptedStep(BaseModel):
    """One deterministic model response."""

    tool_call: ToolCall | None = None
    text: str | None = None


class CodingTaskCase(BaseModel):
    """Data needed to materialize and verify one small coding task."""

    name: str
    task: str
    files: dict[str, str]
    scripted_steps: list[ScriptedStep]
    acceptance_criteria: list[str]
    expected_changed_files: list[str] = Field(default_factory=list)
    expected_file_contents: dict[str, str] = Field(default_factory=dict)
    expected_final_text: str | None = None
    verification_command: list[str] | None = None


class CodingTaskResult(BaseModel):
    """Compact metrics shared by all evaluation modes."""

    name: str
    mode: EvaluationMode
    success: bool
    verification_status: Literal["not_run", "passed", "failed", "error"]
    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    latency_ms: float
    failure_reason: str | None = None
    termination_reason: str
    final_stop_reason: str | None = None
    changed_files: list[str] = Field(default_factory=list)


class SweBenchRunMetrics(BaseModel):
    """Per-instance diagnostics written alongside standard predictions."""

    instance_id: str
    termination: str
    final_stop_reason: str | None = None
    steps: int
    tool_calls: int
    changed_files: list[str] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    latency_ms: float
    failure_reason: str | None = None


class EvaluationSummary(BaseModel):
    total: int
    successful: int
    success_rate: float
    average_steps: float
    average_tool_calls: float
    total_tokens: int
    total_estimated_cost: float
    average_latency_ms: float


class CaseVerification(BaseModel):
    status: Literal["not_run", "passed", "failed", "error"]
    failures: list[str] = Field(default_factory=list)


class SweBenchInstance(BaseModel):
    """Minimal instance fields needed to generate a prediction patch."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str


class ScriptedProviderAdapter:
    """Deterministic provider used only by the local evaluation suite."""

    provider = "scripted"
    model = "deepseek-v4-flash"
    capabilities = ProviderCapabilities()

    def __init__(self, steps: list[ScriptedStep]) -> None:
        self._responses = [self._response(step) for step in steps]

    async def stream_response(
        self,
        *,
        system: str,
        tools: list[ToolDefinition],
        messages: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        del system, tools, messages
        if not self._responses:
            raise RuntimeError("Scripted evaluation provider ran out of responses.")
        response = self._responses.pop(0)
        if on_text_delta is not None:
            for text in response.text:
                on_text_delta(text)
        return response

    def tool_result_message(self, tool_results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [result.model_dump() for result in tool_results],
        }

    def _response(self, step: ScriptedStep) -> ProviderResponse:
        if step.tool_call is not None:
            tool_call = step.tool_call
            message = {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_call.tool_use_id,
                        "name": tool_call.name,
                        "input": tool_call.input,
                    }
                ],
            }
            return ProviderResponse(
                message=message,
                stop_reason="tool_use",
                tool_calls=[tool_call],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        text = step.text or ""
        return ProviderResponse(
            message={"role": "assistant", "content": text},
            stop_reason="end_turn",
            text=[text],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


def tool_step(
    name: str,
    input_data: dict[str, Any],
    tool_use_id: str,
) -> ScriptedStep:
    return ScriptedStep(
        tool_call=ToolCall(
            name=name,
            input=input_data,
            tool_use_id=tool_use_id,
        )
    )


def final_step(text: str) -> ScriptedStep:
    return ScriptedStep(text=text)


def build_cases() -> dict[str, CodingTaskCase]:
    """Return a representative suite with search, editing, and recovery."""
    return {
        "repository_search": CodingTaskCase(
            name="repository_search",
            task=(
                "Find where estimated token cost is calculated. Name the file and "
                "class, and do not edit anything."
            ),
            files={
                "agent/token_tracker.py": (
                    "class TokenTracker:\n"
                    "    def estimated_cost(self) -> float:\n"
                    "        return 0.0\n"
                ),
            },
            scripted_steps=[
                tool_step(
                    "search_text",
                    {"pattern": "estimated_cost", "file_pattern": "**/*.py"},
                    "search-cost",
                ),
                tool_step(
                    "read_file",
                    {"path": "agent/token_tracker.py", "offset": 1, "limit": 80},
                    "read-tracker",
                ),
                final_step(
                    "Estimated cost is calculated by TokenTracker in "
                    "agent/token_tracker.py."
                ),
            ],
            acceptance_criteria=[
                "Search the repository before answering.",
                "Name agent/token_tracker.py and TokenTracker.",
                "Do not modify files.",
            ],
            expected_final_text="agent/token_tracker.py",
        ),
        "small_bug_fix": CodingTaskCase(
            name="small_bug_fix",
            task=(
                "Fix add(2, 3) so it returns 5, then run the focused test. "
                "Change only math_utils.py."
            ),
            files={
                "math_utils.py": (
                    "def add(a: int, b: int) -> int:\n"
                    "    return a - b\n"
                ),
                "tests/test_math_utils.py": (
                    "from math_utils import add\n\n\n"
                    "def test_add() -> None:\n"
                    "    assert add(2, 3) == 5\n"
                ),
            },
            scripted_steps=[
                tool_step("read_file", {"path": "math_utils.py"}, "read-math"),
                tool_step(
                    "edit_file",
                    {
                        "path": "math_utils.py",
                        "old_text": "    return a - b\n",
                        "new_text": "    return a + b\n",
                    },
                    "fix-add",
                ),
                tool_step(
                    "run_command",
                    {
                        "command": (
                            f"{PYTHON} -m pytest tests/test_math_utils.py -q"
                        )
                    },
                    "test-add",
                ),
                final_step("The add bug is fixed and the focused test passes."),
            ],
            acceptance_criteria=[
                "Read math_utils.py before editing it.",
                "Make add return a + b.",
                "Run tests/test_math_utils.py.",
                "Change only math_utils.py.",
            ],
            expected_changed_files=["math_utils.py"],
            expected_file_contents={
                "math_utils.py": (
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n"
                )
            },
            verification_command=[
                PYTHON,
                "-m",
                "pytest",
                "tests/test_math_utils.py",
                "-q",
            ],
        ),
        "failed_edit_recovery": CodingTaskCase(
            name="failed_edit_recovery",
            task=(
                "Change the default timeout from 10 to 30. Inspect the file if an "
                "edit fails, then compile the module."
            ),
            files={"settings.py": "DEFAULT_TIMEOUT = 10\n"},
            scripted_steps=[
                tool_step(
                    "edit_file",
                    {
                        "path": "settings.py",
                        "old_text": "DEFAULT_TIMEOUT = 20",
                        "new_text": "DEFAULT_TIMEOUT = 30",
                    },
                    "failed-edit",
                ),
                tool_step("read_file", {"path": "settings.py"}, "read-settings"),
                tool_step(
                    "edit_file",
                    {
                        "path": "settings.py",
                        "old_text": "DEFAULT_TIMEOUT = 10",
                        "new_text": "DEFAULT_TIMEOUT = 30",
                    },
                    "fix-timeout",
                ),
                tool_step(
                    "run_command",
                    {"command": f"{PYTHON} -m py_compile settings.py"},
                    "compile-settings",
                ),
                final_step("The timeout is now 30 and the module compiles."),
            ],
            acceptance_criteria=[
                "Recover from the failed edit by reading settings.py.",
                "Set DEFAULT_TIMEOUT to 30.",
                "Compile settings.py successfully.",
            ],
            expected_changed_files=["settings.py"],
            expected_file_contents={"settings.py": "DEFAULT_TIMEOUT = 30\n"},
            verification_command=[PYTHON, "-m", "py_compile", "settings.py"],
        ),
    }


def materialize_case(case: CodingTaskCase, workspace: Path) -> None:
    for relative_path, content in case.files.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def create_real_provider_adapter(api_key: str | None = None) -> ProviderAdapter:
    config = load_deepseek_config(api_key=api_key)
    return DeepSeekProvider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )


async def evaluate_case(
    case: CodingTaskCase,
    *,
    mode: Literal["deterministic", "real_model"],
    provider_adapter: ProviderAdapter | None = None,
    keep_workspace: bool = False,
    max_steps: int | None = None,
) -> CodingTaskResult:
    workspace = Path(tempfile.mkdtemp(prefix=f"agent-eval-{case.name}-"))
    started = perf_counter()
    try:
        materialize_case(case, workspace)
        adapter: ProviderAdapter
        if mode == "deterministic":
            adapter = cast(ProviderAdapter, ScriptedProviderAdapter(case.scripted_steps))
        elif provider_adapter is not None:
            adapter = provider_adapter
        else:
            raise ValueError("Real-model evaluation requires a provider adapter.")

        agent = Agent(
            provider_adapter=adapter,
            registry=create_registry(workspace),
            max_steps=max_steps or 20,
            stream_output=False,
            approval_callback=deny_broad_command,
        )
        run = await agent.run(evaluation_prompt(case, mode))
        verification = verify_case(case, workspace, run, agent)
        failures = list(verification.failures)
        success = run.termination == "completed" and not failures
        if run.termination != "completed":
            failures.insert(0, f"agent terminated with {run.termination}")
        return result_from_run(
            name=case.name,
            mode=mode,
            run=run,
            agent=agent,
            success=success,
            verification_status=verification.status,
            latency_ms=(perf_counter() - started) * 1000,
            failure_reason="; ".join(failures) or None,
        )
    finally:
        if keep_workspace:
            print(f"Kept evaluation workspace: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


def evaluation_prompt(
    case: CodingTaskCase,
    mode: Literal["deterministic", "real_model"],
) -> str:
    if mode == "deterministic":
        return case.task
    lines = [case.task, "", "Acceptance criteria:"]
    lines.extend(f"- {item}" for item in case.acceptance_criteria)
    if case.verification_command:
        lines.extend(
            [
                "",
                "Recommended verification command:",
                f"- {' '.join(case.verification_command)}",
            ]
        )
    return "\n".join(lines)


def verify_case(
    case: CodingTaskCase,
    workspace: Path,
    run: AgentRun,
    agent: Agent,
) -> CaseVerification:
    failures: list[str] = []
    changed_files = relative_changed_files(workspace, agent)
    if changed_files != case.expected_changed_files:
        failures.append(
            "changed files mismatch: "
            f"expected {case.expected_changed_files}, got {changed_files}"
        )
    for relative_path, expected in case.expected_file_contents.items():
        path = workspace / relative_path
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            failures.append(f"unexpected content in {relative_path}")
    if case.expected_final_text and case.expected_final_text not in final_answer(run):
        failures.append(f"final answer missing {case.expected_final_text!r}")

    if case.verification_command is None:
        return CaseVerification(
            status="failed" if failures else "not_run",
            failures=failures,
        )
    try:
        completed = run_subprocess(
            case.verification_command,
            cwd=workspace,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        failures.append(f"verification error: {error}")
        return CaseVerification(status="error", failures=failures)
    if completed.returncode != 0:
        failures.append(
            f"verification exited {completed.returncode}: "
            f"{completed.stdout}{completed.stderr}".strip()
        )
    return CaseVerification(
        status="failed" if failures else "passed",
        failures=failures,
    )


def relative_changed_files(workspace: Path, agent: Agent) -> list[str]:
    root = workspace.resolve()
    return sorted(
        path.resolve().relative_to(root).as_posix()
        for path in agent.registry.changed_files
    )


def final_answer(run: AgentRun) -> str:
    for step in reversed(run.steps):
        if step.text:
            return "\n".join(step.text)
    return ""


def deny_broad_command(
    tool_call: ToolCall,
    policy: ToolApprovalPolicy,
) -> bool:
    del tool_call, policy
    return False


async def evaluate_cases(
    selected_names: list[str],
    *,
    mode: Literal["deterministic", "real_model"],
    keep_workspaces: bool,
    api_key: str | None = None,
    max_steps: int | None = None,
) -> list[CodingTaskResult]:
    cases = build_cases()
    provider = create_real_provider_adapter(api_key) if mode == "real_model" else None
    results: list[CodingTaskResult] = []
    for name in selected_names:
        results.append(
            await evaluate_case(
                cases[name],
                mode=mode,
                provider_adapter=provider,
                keep_workspace=keep_workspaces,
                max_steps=max_steps,
            )
        )
    return results


def load_swe_bench_instances(
    path: Path,
    *,
    limit: int | None = None,
    instance_ids: set[str] | None = None,
) -> list[SweBenchInstance]:
    if not path.exists():
        raise SystemExit(f"SWE-bench file not found: {path}")
    records = load_json_records(path)
    instances = [SweBenchInstance.model_validate(record) for record in records]
    if instance_ids:
        instances = [
            instance
            for instance in instances
            if instance.instance_id in instance_ids
        ]
    if limit is not None:
        instances = instances[:limit]
    if not instances:
        raise SystemExit(f"No selected SWE-bench instances found in {path}")
    return instances


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            cast(dict[str, Any], json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [cast(dict[str, Any], record) for record in payload]
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return [cast(dict[str, Any], record) for record in payload["instances"]]
    raise SystemExit("SWE-bench JSON must be a list or contain an instances list.")


async def evaluate_swe_bench_instances(
    instances: list[SweBenchInstance],
    *,
    api_key: str | None,
    keep_workspaces: bool,
    predictions_path: Path,
    max_steps: int,
    metrics_path: Path | None = None,
) -> list[CodingTaskResult]:
    provider = create_real_provider_adapter(api_key)
    resolved_metrics_path = metrics_path or default_swe_bench_metrics_path(
        predictions_path
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("", encoding="utf-8")
    resolved_metrics_path.write_text("", encoding="utf-8")
    results: list[CodingTaskResult] = []
    for instance in instances:
        result, patch = await evaluate_swe_bench_instance(
            instance,
            provider_adapter=provider,
            keep_workspace=keep_workspaces,
            max_steps=max_steps,
        )
        append_prediction(predictions_path, instance, provider, patch)
        append_swe_bench_metrics(resolved_metrics_path, result)
        results.append(result)
        print_swe_bench_result(result, resolved_metrics_path)
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {resolved_metrics_path}")
    return results


async def evaluate_swe_bench_instance(
    instance: SweBenchInstance,
    *,
    provider_adapter: ProviderAdapter,
    keep_workspace: bool,
    max_steps: int,
) -> tuple[CodingTaskResult, str]:
    workspace = Path(tempfile.mkdtemp(prefix=f"agent-swe-{instance.instance_id}-"))
    repo_workspace = workspace / "repo"
    started = perf_counter()
    activity_prefix = f"[swe:{instance.instance_id}]"
    print(f"{activity_prefix}[main] Starting with max_steps={max_steps}.")
    try:
        clone_instance(instance, repo_workspace)
        agent = Agent(
            provider_adapter=provider_adapter,
            registry=create_registry(repo_workspace),
            max_steps=max_steps,
            stream_output=False,
            approval_callback=deny_broad_command,
            system_prompt_suffix=SWE_BENCH_SYSTEM_PROMPT_SUFFIX,
            reserve_finalization_steps=SWE_BENCH_FINALIZATION_STEPS,
            activity_prefix=activity_prefix,
            activity_label="main",
        )
        run = await agent.run(swe_bench_prompt(instance))
        patch = collect_patch(repo_workspace)
        changed_files = collect_patch_files(repo_workspace)
        success = run.termination == "completed" and bool(patch.strip())
        failure_reason = None
        if run.termination != "completed":
            failure_reason = f"agent terminated with {run.termination}"
        elif not patch.strip():
            failure_reason = "no patch generated"
        result = result_from_run(
            name=instance.instance_id,
            mode="swe_bench",
            run=run,
            agent=agent,
            success=success,
            verification_status="not_run",
            latency_ms=(perf_counter() - started) * 1000,
            failure_reason=failure_reason,
        )
        result.changed_files = changed_files
        return result, patch
    finally:
        if keep_workspace:
            print(f"Kept SWE-bench workspace: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


def clone_instance(instance: SweBenchInstance, destination: Path) -> None:
    source_path = Path(instance.repo).expanduser()
    source = (
        source_path.resolve().as_posix()
        if source_path.exists()
        else f"https://github.com/{instance.repo}.git"
    )
    run_subprocess(["git", "clone", "--quiet", source, destination.as_posix()])
    run_subprocess(
        ["git", "checkout", "--quiet", instance.base_commit],
        cwd=destination,
    )


def collect_patch(repo_workspace: Path) -> str:
    run_subprocess(["git", "add", "-N", "."], cwd=repo_workspace)
    completed = run_subprocess(
        ["git", "diff", "--binary"],
        cwd=repo_workspace,
        check=False,
    )
    return completed.stdout


def collect_patch_files(repo_workspace: Path) -> list[str]:
    completed = run_subprocess(
        ["git", "diff", "--name-only"],
        cwd=repo_workspace,
        check=False,
    )
    return [line for line in completed.stdout.splitlines() if line]


def default_swe_bench_metrics_path(predictions_path: Path) -> Path:
    suffix = predictions_path.suffix
    stem = (
        predictions_path.name.removesuffix(suffix) if suffix else predictions_path.name
    )
    return predictions_path.with_name(f"{stem}.metrics.jsonl")


def append_prediction(
    path: Path,
    instance: SweBenchInstance,
    provider_adapter: ProviderAdapter,
    model_patch: str,
) -> None:
    prediction = {
        "instance_id": instance.instance_id,
        "model_name_or_path": (
            f"{provider_adapter.provider}/{provider_adapter.model}"
        ),
        "model_patch": model_patch,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(prediction) + "\n")


def append_swe_bench_metrics(
    path: Path,
    result: CodingTaskResult,
) -> None:
    metrics = SweBenchRunMetrics(
        instance_id=result.name,
        termination=result.termination_reason,
        final_stop_reason=result.final_stop_reason,
        steps=result.steps,
        tool_calls=result.tool_calls,
        changed_files=result.changed_files,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost=result.estimated_cost,
        latency_ms=result.latency_ms,
        failure_reason=result.failure_reason,
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metrics.model_dump()) + "\n")


def print_swe_bench_result(
    result: CodingTaskResult,
    metrics_path: Path,
) -> None:
    print(
        f"[swe:{result.name}][main] Finished "
        f"termination={result.termination_reason} steps={result.steps} "
        f"tool_calls={result.tool_calls} "
        f"changed_files={len(result.changed_files)}."
    )
    print(f"[swe:{result.name}][metrics] Appended to {metrics_path}.")


def swe_bench_prompt(instance: SweBenchInstance) -> str:
    return "\n".join(
        [
            "Fix the issue below in the current repository.",
            "Make the smallest appropriate change and run focused verification when practical.",
            "Do not produce benchmark scoring; leave the working tree with the patch.",
            "",
            instance.problem_statement,
        ]
    )


def result_from_run(
    *,
    name: str,
    mode: EvaluationMode,
    run: AgentRun,
    agent: Agent,
    success: bool,
    verification_status: Literal["not_run", "passed", "failed", "error"],
    latency_ms: float,
    failure_reason: str | None,
) -> CodingTaskResult:
    return CodingTaskResult(
        name=name,
        mode=mode,
        success=success,
        verification_status=verification_status,
        steps=len(run.steps),
        tool_calls=sum(len(step.tool_calls) for step in run.steps),
        input_tokens=agent.token_tracker.input_tokens,
        output_tokens=agent.token_tracker.output_tokens,
        estimated_cost=agent.token_tracker.estimated_cost,
        latency_ms=latency_ms,
        failure_reason=failure_reason,
        termination_reason=run.termination,
        final_stop_reason=run.final_stop_reason,
    )


def summarize_results(results: list[CodingTaskResult]) -> EvaluationSummary:
    total = len(results)
    successful = sum(result.success for result in results)
    return EvaluationSummary(
        total=total,
        successful=successful,
        success_rate=successful / total if total else 0.0,
        average_steps=average(result.steps for result in results),
        average_tool_calls=average(result.tool_calls for result in results),
        total_tokens=sum(
            result.input_tokens + result.output_tokens for result in results
        ),
        total_estimated_cost=sum(result.estimated_cost for result in results),
        average_latency_ms=average(result.latency_ms for result in results),
    )


def average(values: Iterable[int | float]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def print_results(results: list[CodingTaskResult]) -> None:
    print("Coding-agent evaluation")
    print("=======================")
    for result in results:
        status = "PASS" if result.success else "FAIL"
        total_tokens = result.input_tokens + result.output_tokens
        print(
            f"{status} {result.name}: verification={result.verification_status} "
            f"termination={result.termination_reason}"
        )
        print(
            f"  steps={result.steps} tool_calls={result.tool_calls} "
            f"tokens={total_tokens} cost=${result.estimated_cost:.6f} "
            f"latency_ms={result.latency_ms:.1f}"
        )
        if result.failure_reason:
            print(f"  failure_reason={result.failure_reason}")
    summary = summarize_results(results)
    print("Summary")
    print(
        f"  success={summary.successful}/{summary.total} "
        f"({summary.success_rate:.1%})"
    )
    print(
        f"  average_steps={summary.average_steps:.2f} "
        f"average_tool_calls={summary.average_tool_calls:.2f}"
    )
    print(
        f"  total_tokens={summary.total_tokens} "
        f"total_cost=${summary.total_estimated_cost:.6f} "
        f"average_latency_ms={summary.average_latency_ms:.1f}"
    )


def run_subprocess(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Local case names to run.")
    parser.add_argument("--list", action="store_true", help="List local cases.")
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Use the configured live model. Local evaluation is deterministic by default.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument(
        "--swe-bench",
        type=Path,
        help="Generate patches for instances in a JSON or JSONL file.",
    )
    parser.add_argument("--swe-bench-limit", type=int)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Select an instance ID; may be repeated.",
    )
    parser.add_argument(
        "--swe-bench-predictions",
        type=Path,
        default=Path(".agents/evals/swe-bench-predictions.jsonl"),
    )
    parser.add_argument(
        "--swe-bench-metrics",
        type=Path,
        help=(
            "Write per-instance SWE-bench diagnostics as JSONL. Defaults to "
            "<predictions-stem>.metrics.jsonl."
        ),
    )
    return parser.parse_args(argv)


async def run_eval_cli(
    argv: Sequence[str] | None = None,
    *,
    api_key: str | None = None,
) -> int:
    args = parse_args(argv)
    cases = build_cases()
    if args.list:
        for case in cases.values():
            print(f"{case.name}: {case.task}")
        return 0

    if args.swe_bench is not None:
        if args.cases or args.real_model:
            raise SystemExit(
                "Use --swe-bench without local case names or --real-model."
            )
        load_dotenv()
        instances = load_swe_bench_instances(
            args.swe_bench,
            limit=args.swe_bench_limit,
            instance_ids=set(args.instance_id),
        )
        results = await evaluate_swe_bench_instances(
            instances,
            api_key=api_key,
            keep_workspaces=args.keep_workspaces,
            predictions_path=args.swe_bench_predictions,
            max_steps=args.max_steps or 30,
            metrics_path=args.swe_bench_metrics,
        )
    else:
        mode: Literal["deterministic", "real_model"] = (
            "real_model" if args.real_model else "deterministic"
        )
        if mode == "real_model":
            load_dotenv()
        selected_names = args.cases or (
            ["repository_search"] if mode == "real_model" else list(cases)
        )
        unknown = sorted(set(selected_names) - set(cases))
        if unknown:
            raise SystemExit(f"Unknown evaluation case(s): {', '.join(unknown)}")
        results = await evaluate_cases(
            selected_names,
            mode=mode,
            keep_workspaces=args.keep_workspaces,
            api_key=api_key,
            max_steps=args.max_steps,
        )

    print_results(results)
    return 0 if all(result.success for result in results) else 1


def main() -> None:
    raise SystemExit(asyncio.run(run_eval_cli()))


if __name__ == "__main__":
    main()
