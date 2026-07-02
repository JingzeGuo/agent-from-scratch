import json
from pathlib import Path

import pytest

from agent.schemas import (
    AgentRun,
    AgentStep,
    ToolCall,
    ToolResult,
    VerificationEvidence,
)
from scripts.evaluate_coding_tasks import (
    CodingTaskResult,
    FailureReason,
    classify_failure_reasons,
    load_swe_bench_instances,
    print_results,
    summarize_results,
)


def test_classify_failure_reasons_detects_requested_categories() -> None:
    run = AgentRun(
        objective="Repair code",
        termination="max_steps",
        final_stop_reason="tool_use",
        verification=VerificationEvidence(status="failed"),
        steps=[
            AgentStep(
                step_number=1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        name="run_command",
                        input={"command": "python -m py_compile module.py"},
                        tool_use_id="toolu_compile",
                    )
                ],
                tool_results=[
                    ToolResult(
                        tool_use_id="toolu_compile",
                        content=(
                            "exit_code: 1\n"
                            "timed_out: false\n"
                            "stderr:\n"
                            "SyntaxError: expected ':'"
                        ),
                    )
                ],
            ),
            AgentStep(
                step_number=2,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        name="run_command",
                        input={"command": "rm -rf ."},
                        tool_use_id="toolu_rm",
                    )
                ],
                tool_results=[
                    ToolResult(
                        tool_use_id="toolu_rm",
                        content=(
                            "Tool 'run_command' raised ValueError: "
                            "Blocked dangerous command: rm"
                        ),
                        is_error=True,
                    )
                ],
            ),
        ],
    )

    reasons = classify_failure_reasons(
        run,
        ["external pytest oracle did not pass"],
    )

    assert reasons == [
        "compile_error",
        "test_failure",
        "max_step",
        "unsafe_command_blocked",
    ]


def test_summarize_results_reports_eval_metrics() -> None:
    results = [
        make_result("pass", True, steps=2, tool_calls=3, cost=0.02),
        make_result(
            "fail",
            False,
            steps=4,
            tool_calls=5,
            cost=0.04,
            failure_reasons=["test_failure"],
        ),
    ]

    summary = summarize_results(results)

    assert summary.passed == 1
    assert summary.total == 2
    assert summary.pass_rate == 0.5
    assert summary.average_steps == 3.0
    assert summary.average_token_cost == pytest.approx(0.03)
    assert summary.average_tool_calls == 4.0
    assert summary.failure_reason_counts["test_failure"] == 1


def test_print_results_includes_requested_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_results(
        [
            make_result("pass", True, steps=2, tool_calls=3, cost=0.02),
            make_result(
                "fail",
                False,
                steps=4,
                tool_calls=5,
                cost=0.04,
                failure_reasons=["unsafe_command_blocked"],
            ),
        ]
    )

    output = capsys.readouterr().out
    assert "pass_rate=1/2 (50.0%)" in output
    assert "average_steps=3.00" in output
    assert "average_token_cost=$0.030000" in output
    assert "average_tool_calls=4.00" in output
    assert "unsafe_command_blocked=1" in output


def test_load_swe_bench_instances_reads_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "swe.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "demo__repo-1",
                "repo": "demo/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug.",
                "test_patch": "diff --git a/tests/test_demo.py b/tests/test_demo.py",
                "FAIL_TO_PASS": "['tests/test_demo.py::test_bug']",
                "PASS_TO_PASS": ["tests/test_demo.py::test_existing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    instances = load_swe_bench_instances(path)

    assert len(instances) == 1
    assert instances[0].instance_id == "demo__repo-1"
    assert instances[0].fail_to_pass == ["tests/test_demo.py::test_bug"]
    assert instances[0].pass_to_pass == ["tests/test_demo.py::test_existing"]


def make_result(
    name: str,
    task_success: bool,
    *,
    steps: int,
    tool_calls: int,
    cost: float,
    failure_reasons: list[FailureReason] | None = None,
) -> CodingTaskResult:
    return CodingTaskResult(
        name=name,
        mode="scripted",
        task_success=task_success,
        runtime_success=task_success,
        verification_success=task_success,
        recovery_success=False,
        tool_accuracy=True,
        steps=steps,
        tool_calls=tool_calls,
        tools=[],
        commands=[],
        latency_ms=1.0,
        input_tokens=10,
        output_tokens=5,
        estimated_cost=cost,
        failure_reasons=failure_reasons or [],
    )
