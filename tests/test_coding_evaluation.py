import asyncio
import json
from pathlib import Path

import pytest

from agent.provider import DeepSeekProvider
from scripts.evaluate_coding_tasks import (
    CodingTaskResult,
    SweBenchInstance,
    append_prediction,
    build_cases,
    evaluate_cases,
    evaluation_prompt,
    load_swe_bench_instances,
    print_results,
    summarize_results,
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
