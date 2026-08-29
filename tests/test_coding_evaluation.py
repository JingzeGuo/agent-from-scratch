import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.provider import DeepSeekProvider
from scripts.evaluate_coding_tasks import (
    SWE_BENCH_SYSTEM_PROMPT_SUFFIX,
    CodingTaskResult,
    ScriptedProviderAdapter,
    SweBenchInstance,
    append_prediction,
    append_swe_bench_metrics,
    build_cases,
    clone_instance,
    create_swe_bench_registry,
    create_swe_bench_workspace,
    default_swe_bench_metrics_path,
    default_swe_bench_trajectories_dir,
    evaluate_cases,
    evaluate_swe_bench_instance,
    evaluation_prompt,
    final_step,
    load_swe_bench_instances,
    parse_args,
    print_results,
    summarize_results,
    swe_bench_prompt,
    tool_step,
)


def test_build_cases_is_small_and_representative() -> None:
    cases = build_cases()

    assert list(cases) == [
        "repository_search",
        "small_bug_fix",
        "failed_edit_recovery",
    ]
    assert cases["small_bug_fix"].verification_command is not None
    assert cases["failed_edit_recovery"].scripted_steps[0].tool_call is not None


def test_real_model_prompt_includes_acceptance_and_verification() -> None:
    case = build_cases()["small_bug_fix"]

    prompt = evaluation_prompt(case, "real_model")

    assert "Acceptance criteria:" in prompt
    assert "Make add return a + b." in prompt
    assert "Recommended verification command:" in prompt
    assert "tests/test_math_utils.py" in prompt


def test_deterministic_prompt_is_the_plain_task() -> None:
    case = build_cases()["small_bug_fix"]

    assert evaluation_prompt(case, "deterministic") == case.task


def test_summarize_results_reports_core_metrics() -> None:
    results = [
        make_result("pass", True, steps=2, tool_calls=3, tokens=15, cost=0.02),
        make_result("fail", False, steps=4, tool_calls=5, tokens=25, cost=0.04),
    ]

    summary = summarize_results(results)

    assert summary.successful == 1
    assert summary.total == 2
    assert summary.success_rate == 0.5
    assert summary.average_steps == 3.0
    assert summary.average_tool_calls == 4.0
    assert summary.total_tokens == 40
    assert summary.total_estimated_cost == pytest.approx(0.06)


def test_print_results_includes_status_and_failure_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_results(
        [
            make_result("pass", True, steps=2, tool_calls=3, tokens=15, cost=0.02),
            make_result(
                "fail",
                False,
                steps=4,
                tool_calls=5,
                tokens=25,
                cost=0.04,
                failure_reason="focused test failed",
            ),
        ]
    )

    output = capsys.readouterr().out
    assert "PASS pass: verification=passed termination=completed" in output
    assert "FAIL fail: verification=failed termination=max_steps" in output
    assert "failure_reason=focused test failed" in output
    assert "success=1/2 (50.0%)" in output


def test_load_swe_bench_instances_selects_ids_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "instances.jsonl"
    records = [
        {
            "instance_id": "demo__repo-1",
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the first bug.",
        },
        {
            "instance_id": "demo__repo-2",
            "repo": "demo/repo",
            "base_commit": "def456",
            "problem_statement": "Fix the second bug.",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    instances = load_swe_bench_instances(
        path,
        instance_ids={"demo__repo-2"},
    )

    assert [instance.instance_id for instance in instances] == ["demo__repo-2"]


def test_swe_bench_instance_discards_gold_fields(tmp_path: Path) -> None:
    path = tmp_path / "instances.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "demo__repo-1",
                "repo": "demo/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug.",
                "patch": "GOLD SOURCE PATCH",
                "test_patch": "GOLD TEST PATCH",
                "FAIL_TO_PASS": ["secret_test"],
                "PASS_TO_PASS": ["existing_test"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [instance] = load_swe_bench_instances(path)

    assert instance.model_dump() == {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug.",
    }
    prompt = swe_bench_prompt(instance)
    assert "GOLD SOURCE PATCH" not in prompt
    assert "GOLD TEST PATCH" not in prompt
    assert "secret_test" not in prompt


def test_append_prediction_writes_standard_fields(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    provider = DeepSeekProvider(
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://api.deepseek.com",
    )
    instance = SweBenchInstance(
        instance_id="demo__repo-1",
        repo="demo/repo",
        base_commit="abc123",
        problem_statement="Fix the bug.",
    )

    append_prediction(path, instance, provider, "diff --git a/a.py b/a.py\n")

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "instance_id": "demo__repo-1",
        "model_name_or_path": "deepseek/deepseek-v4-flash",
        "model_patch": "diff --git a/a.py b/a.py\n",
    }


def test_swe_bench_profile_stops_on_environment_failures() -> None:
    instance = SweBenchInstance(
        instance_id="demo__repo-1",
        repo="demo/repo",
        base_commit="abc123",
        problem_statement="Fix the bug.",
    )

    assert "when practical" in swe_bench_prompt(instance)
    assert "treat that as an environment block" in (SWE_BENCH_SYSTEM_PROMPT_SUFFIX)
    assert "override the generic verification and recovery rules" in (
        SWE_BENCH_SYSTEM_PROMPT_SUFFIX
    )
    assert "Do not create dependency shims" in SWE_BENCH_SYSTEM_PROMPT_SUFFIX
    assert "conftest.py" in SWE_BENCH_SYSTEM_PROMPT_SUFFIX
    assert "paths beginning with _tmp" in SWE_BENCH_SYSTEM_PROMPT_SUFFIX
    assert "never add distutils, pytz, or asgiref" in (SWE_BENCH_SYSTEM_PROMPT_SUFFIX)
    assert "Do not inspect Git metadata" in SWE_BENCH_SYSTEM_PROMPT_SUFFIX
    assert "network source" in SWE_BENCH_SYSTEM_PROMPT_SUFFIX


def test_swe_bench_registry_removes_network_tools_and_blocks_git(
    tmp_path: Path,
) -> None:
    registry = create_swe_bench_registry(tmp_path)

    assert "search_web" not in registry.tools
    assert "fetch_url" not in registry.tools
    assert "run_command" in registry.tools
    assert registry.blocked_path_parts == frozenset({".git"})
    assert registry.blocked_command_names == frozenset({"git"})


def test_swe_bench_workspace_name_contains_only_random_identity() -> None:
    workspace = create_swe_bench_workspace()
    try:
        assert re.fullmatch(r"agent-swe-[0-9a-f]{32}-.+", workspace.name)
        assert "__" not in workspace.name
    finally:
        shutil.rmtree(workspace)


def test_clone_instance_fetches_only_base_commit_and_removes_remote(
    tmp_path: Path,
) -> None:
    source = initialize_git_repo(tmp_path / "source")
    (source / "value.txt").write_text("base\n", encoding="utf-8")
    git(source, "add", "value.txt")
    git(source, "commit", "--quiet", "-m", "base")
    base_commit = git(source, "rev-parse", "HEAD").stdout.strip()
    (source / "value.txt").write_text("future\n", encoding="utf-8")
    git(source, "commit", "--quiet", "-am", "future")
    future_commit = git(source, "rev-parse", "HEAD").stdout.strip()
    destination = tmp_path / "checkout"
    instance = SweBenchInstance(
        instance_id="demo__repo-1",
        repo=source.as_posix(),
        base_commit=base_commit,
        problem_statement="Fix the bug.",
    )

    clone_instance(instance, destination)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "base\n"
    assert git(destination, "remote").stdout == ""
    assert git(destination, "rev-list", "--all", "--count").stdout.strip() == "1"
    assert git(destination, "cat-file", "-e", future_commit, check=False).returncode != 0


def test_swe_bench_instance_records_tool_inputs_in_trajectory(
    tmp_path: Path,
) -> None:
    source = initialize_git_repo(tmp_path / "source")
    (source / "README.md").write_text("demo\n", encoding="utf-8")
    git(source, "add", "README.md")
    git(source, "commit", "--quiet", "-m", "base")
    instance = SweBenchInstance(
        instance_id="demo__repo-1",
        repo=source.as_posix(),
        base_commit=git(source, "rev-parse", "HEAD").stdout.strip(),
        problem_statement="Inspect the repository.",
    )
    provider = ScriptedProviderAdapter(
        [
            tool_step("read_file", {"path": "README.md"}, "read-readme"),
            tool_step("run_command", {"command": "git status"}, "blocked-git"),
            final_step("Done."),
        ]
    )
    trajectories_dir = tmp_path / "trajectories"

    asyncio.run(
        evaluate_swe_bench_instance(
            instance,
            provider_adapter=provider,
            keep_workspace=False,
            max_steps=4,
            trajectories_dir=trajectories_dir,
        )
    )

    events_path = trajectories_dir / "events" / "demo__repo-1.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    tool_started = next(
        event for event in events if event["event_type"] == "tool_started"
    )
    assert tool_started["agent_label"] == "main"
    assert tool_started["tool_name"] == "read_file"
    assert tool_started["tool_input"] == {"path": "README.md"}
    assert tool_started["command"] is None
    command_started = next(
        event
        for event in events
        if event["event_type"] == "tool_started"
        and event["tool_name"] == "run_command"
    )
    assert command_started["tool_input"] == {"command": "git status"}
    assert command_started["command"] == "git status"
    assert events[-1]["event_type"] == "run_finished"


def test_append_swe_bench_metrics_writes_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "predictions.metrics.jsonl"
    result = make_result(
        "demo__repo-1",
        False,
        steps=30,
        tool_calls=29,
        tokens=100,
        cost=0.05,
        failure_reason="agent terminated with max_steps",
    )
    result.changed_files = ["src/module.py", "tests/test_module.py"]
    result.final_stop_reason = "tool_use"

    append_swe_bench_metrics(path, result)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "instance_id": "demo__repo-1",
        "termination": "max_steps",
        "final_stop_reason": "tool_use",
        "steps": 30,
        "tool_calls": 29,
        "changed_files": ["src/module.py", "tests/test_module.py"],
        "input_tokens": 95,
        "output_tokens": 5,
        "estimated_cost": 0.05,
        "latency_ms": 1.0,
        "failure_reason": "agent terminated with max_steps",
    }


def test_default_swe_bench_metrics_path_uses_prediction_stem() -> None:
    predictions = Path(".agents/evals/lite-50.jsonl")

    assert default_swe_bench_metrics_path(predictions) == Path(
        ".agents/evals/lite-50.metrics.jsonl"
    )


def test_default_swe_bench_trajectories_dir_uses_prediction_stem() -> None:
    predictions = Path(".agents/evals/lite-50.jsonl")

    assert default_swe_bench_trajectories_dir(predictions) == Path(
        ".agents/evals/lite-50.trajectories"
    )


def test_parse_args_accepts_swe_bench_trajectory_directory() -> None:
    args = parse_args(
        [
            "--swe-bench",
            "instances.jsonl",
            "--swe-bench-trajectories",
            "traces",
        ]
    )

    assert args.swe_bench_trajectories == Path("traces")


def test_deterministic_suite_passes() -> None:
    results = asyncio.run(
        evaluate_cases(
            list(build_cases()),
            mode="deterministic",
            keep_workspaces=False,
        )
    )

    assert all(result.success for result in results)
    assert [result.verification_status for result in results] == [
        "not_run",
        "passed",
        "passed",
    ]
    assert all(result.steps > 0 for result in results)
    assert all(result.tool_calls > 0 for result in results)


def make_result(
    name: str,
    success: bool,
    *,
    steps: int,
    tool_calls: int,
    tokens: int,
    cost: float,
    failure_reason: str | None = None,
) -> CodingTaskResult:
    return CodingTaskResult(
        name=name,
        mode="deterministic",
        success=success,
        verification_status="passed" if success else "failed",
        steps=steps,
        tool_calls=tool_calls,
        input_tokens=tokens - 5,
        output_tokens=5,
        estimated_cost=cost,
        latency_ms=1.0,
        failure_reason=failure_reason,
        termination_reason="completed" if success else "max_steps",
    )


def initialize_git_repo(path: Path) -> Path:
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test User")
    return path


def git(
    cwd: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )
