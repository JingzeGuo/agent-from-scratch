# SWE-bench Lite failure analysis — test-100-1022-1755

**Model:** `deepseek/deepseek-v4-flash` · **Config:** Docker, 40 steps, 128k context · **Agent commit:** `1734386` · **Analyzed:** 2026-10-23

## Summary

- **74/100 resolved.** All 100 produced patches and were evaluated; zero infra or evaluator errors.
- Of the 26 failures, **13 are mainly spec/scope mismatches** (the issue allows or asks for something different from what gold tests require) and **13 are mainly agent faults** (symptom fix, missed edge case, or implementation bug). These are retrospective labels, not an adjusted score.
- **Biggest process problem:** local test commands reported success when they hadn't run. `pytest` was missing but `cmd | tail` returned exit code 0. Fixed by `PIPEFAIL_PREFIX` after this run; needs validation.
- Django `11564` and `11905` are flagged `patch_successfully_applied=false`, but their logs show the patches applied cleanly. The failures happened at import/test time.

Method: all 26 failures were compared against the issue text, agent patch, gold patch/tests and evaluator output. No reruns. The agent saw only `problem_statement` (no hints or gold tests).

## Failure categories

| Code | Category | n | Cases |
|---|---|---:|---|
| S | Unclear/conflicting acceptance contract | 9 | django 11905, 13660, 14155 · matplotlib 18869 · seaborn 3407 · flask 5063 · sklearn 10949 · sympy 13043, 16281 |
| G | Gold scope broader than / different from issue | 4 | django 16820 · matplotlib 22711 · flask 4045 · sympy 11870 |
| R | Symptom fixed, root cause remains | 5 | requests 3362 · sklearn 25500 · sympy 13895, 16503, 20322 |
| E | Missed edge case / related input | 4 | django 12308 · sklearn 25638 · sympy 16106, 22840 |
| I | Implementation bug / regression | 4 | django 11564 · sphinx 10451, 7686 · sympy 11400 |

## Every failure

\* = hit 40-step limit. † = no per-test grading (run crashed).

| Instance | Cat | F2P | P2P | Cause |
|---|:-:|--:|--:|---|
| [django-11564](cases/django__django-11564.md)*† | I | – | – | Circular import (`get_script_prefix`) breaks startup. |
| [django-11905](cases/django__django-11905.md)† | S | – | – | Issue asks for ValueError; gold expects a deprecation warning. The ValueError crashed the test runner itself. |
| [django-12308](cases/django__django-12308.md) | E | 0/2 | 20/20 | Used the issue's suggested `prepare_value`, which raises TypeError on unserializable dicts (tuple keys); gold falls back. Borderline S. |
| [django-13660](cases/django__django-13660.md) | S | 2/2 | 4/6 | Passed an empty dict as the issue literally suggests; gold uses `globals()` and tests check `__name__`. |
| [django-14155](cases/django__django-14155.md)* | S | 0/3 | 89/89 | Unwrapped partials as the issue suggests; tests require keeping the partial and `%r` quoting. |
| [django-16820](cases/django__django-16820.md)* | G | 1/7 | 203/203 | Only RenameIndex folded; gold also folds Add/RemoveIndex. |
| [matplotlib-18869](cases/matplotlib__matplotlib-18869.md) | S | 0/4 | 3/3 | Tests need an unspecified private version parser. |
| [matplotlib-22711](cases/matplotlib__matplotlib-22711.md) | G | 0/2 | 91/91 | Fixed the reported IndexError; gold tests check handle positions. |
| [seaborn-3407](cases/mwaskom__seaborn-3407.md) | S | 0/1 | 120/120 | Root cause fixed; test compares `diag_vars` to a list, agent kept an array. |
| [flask-4045](cases/pallets__flask-4045.md) | G | 1/2 | 50/50 | Requested check done; gold also converts endpoint asserts to ValueError. |
| [flask-5063](cases/pallets__flask-5063.md) | S | 0/2 | 54/54 | Used `Domain` heading from the issue; tests want `Host`/`Subdomain`. |
| [requests-3362](cases/psf__requests-3362.md) | R | 0/1 | 75/75 | Added JSON default encoding; `encoding=None` decoder unchanged. |
| [sklearn-10949](cases/scikit-learn__scikit-learn-10949.md) | S | 0/1 | 42/42 | Warning text copied verbatim from the issue; test wants different wording. Mixed-dtype gap is secondary. |
| [sklearn-25500](cases/scikit-learn__scikit-learn-25500.md) | R | 0/1 | 45/45 | Flattened at caller; isotonic `predict` still returns DataFrame. |
| [sklearn-25638](cases/scikit-learn__scikit-learn-25638.md) | E | 9/10 | 194/194 | Nullable Series work; nullable DataFrame → `unknown`. |
| [sphinx-10451](cases/sphinx-doc__sphinx-10451.md)* | I | 2/2 | 77/80 | Keys annotations as `*args`; breaks bare `:param args:` form used by gold tests. |
| [sphinx-7686](cases/sphinx-doc__sphinx-7686.md)* | I | 0/2 | 13/15 | Loop variable shadows `name` param; skip-member hook missing. |
| [sympy-11400](cases/sympy__sympy-11400.md)* | I | 1/2 | 29/29 | `Ne` not imported → NameError. |
| [sympy-11870](cases/sympy__sympy-11870.md)* | G | 0/1 | 58/58 | Changed `trigsimp`; test checks `sinc.rewrite(sin)`. |
| [sympy-13043](cases/sympy__sympy-13043.md) | S | 0/1 | 1/1 | Chose sorted list (allowed by issue); gold uses set. |
| [sympy-13895](cases/sympy__sympy-13895.md)* | R | 1/2 | 82/82 | Blocked a simplification; negative-base powers still wrong. |
| [sympy-16106](cases/sympy__sympy-16106.md) | E | 0/1 | 55/55 | Multiple indices not rendered as fenced tuple. |
| [sympy-16281](cases/sympy__sympy-16281.md) | S | 0/2 | 120/120 | Removed bottom row as the issue asked; snapshots keep it. |
| [sympy-16503](cases/sympy__sympy-16503.md)* | R | 0/1 | 121/121 | Outer baseline shift; inner centering unaddressed. Own checks failed at steps 37/39. Issue leaves direction open. |
| [sympy-20322](cases/sympy__sympy-20322.md)* | R | 0/1 | 89/89 | Patched ceiling; `Mul.is_integer` still wrong. |
| [sympy-22840](cases/sympy__sympy-22840.md) | E | 1/2 | 95/95 | MatrixElement temporaries remain in generated C. |

## Process findings

1. **False-green verification.** `pytest` was not installed, yet `... | tail` returned 0 (django 11564 step 38, sympy 11400 step 28). Sphinx 10451 showed assertion failures with exit 0; sympy 16281 ran a selector matching zero tests. `python -c "sympy.test(...)"` also returns 0 regardless.
2. **Self-written tests confirm the agent's own interpretation** (sympy 13043, flask 5063, sphinx 10451, sklearn 25500). Passing them shows consistency, not correctness.
3. **Step budget matters, but it doesn't explain most failures.** 10/25 runs at the 40-step limit failed (40%) vs 16/75 otherwise (21%). Several cases made their final edit at step 39 with no time to verify.

| Outcome | n | Mean steps | Median | At 40 | Mean runtime | Total cost |
|---|--:|--:|--:|--:|--:|--:|
| Resolved | 74 | 28.4 | 29 | 15 | 3.49 min | $6.75 |
| Unresolved | 26 | 32.4 | 37.5 | 10 | 5.08 min | $2.99 |

## Recommendations

1. **Trustworthy verification:** propagate exit codes (validate pipefail), fail on zero collected tests, keep full logs.
2. **Final fresh-process check:** run a minimal reproduction after the last edit, and reserve a few steps for fixing or rolling back. This would have caught 11564 and 11400.
3. **Trace one level below the symptom** and test the producer, not the caller (R cases).
4. **Small contract matrix** around each change: invalid values, neighboring types/shapes, related operations (E cases).
5. **Audit S/G cases as benchmark issues** separately; keep the official 74% unchanged.
6. **Test each change separately** on this frozen 100-instance sample.

## Caveats

- Django `report.json` parsing merges some test names (12308, 14155); official counts are kept.
- Raw suites include unrelated noise (deprecation warnings, requests `httpbin` fixture errors). P2P uses the gold-patched test files.

## Artifacts

[Official report](../../.agents/evals/test-100-1022-1755/official-reports/deepseek__deepseek-v4-flash.test-100-1022-1755.json) · [Dataset](../../swe_bench_lite_test_100.jsonl) · [Sampling meta](../../swe_bench_lite_test_100.meta.json) (seed `20261022`) · [Predictions](../../.agents/evals/test-100-1022-1755/predictions.jsonl) · [Metrics](../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) · [Classifications](failure-classifications.json) · [Audit](audit.json) · [Per-case details](cases/)
