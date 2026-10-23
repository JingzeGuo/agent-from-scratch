# sympy__sympy-13043: A deterministic list is rejected because gold chooses a set

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P 0/1; P2P 1/1. **Steps:** 36/40. Counts are harness classifications, subject to the report's parser caveats.

Both approaches are suggested by the issue; grading commits to only one return type.

## Issue requirement

The arbitrary-order result should be sorted somehow or returned as a set.

## Agent change

Sorts both separate=True return paths using default_sort_key and retains a list.

## Golden change

Returns sets in both paths and changes all corresponding expectations to sets.

## Observed failure

test_decompose fails immediately on decompose(x, True) == {x}, since the agent returns [x]. The one scored P2P case passes.

## Scope and attribution

This is a direct choice among alternatives allowed by the problem statement, not a missed polynomial edge case.

## Verification in the trajectory

The agent updates deterministic-list expectations, runs the native focused test and later the intpoly suite. The raw official log also has an unrelated exception outside the scored P2P set.

## What would improve the outcome

Specify the return-type contract in acceptance. For agent design, preserve existing API types unless evidence favors a deliberate breaking change; do not count this alternative as automatically correct under the official score.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sympy__sympy-13043`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 80 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 80 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sympy__sympy-13043.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

```text
decompose() function in intpoly returns a list of arbitrary order
The decompose() function, with separate=True, returns `list(poly_dict.values())`, which is ordered arbitrarily.  

What is this used for? It should be sorted somehow, or returning a set (in which case, why not just use the returned dictionary and have the caller take the values). This is causing test failures for me after some changes to the core. 

CC @ArifAhmed1995 @certik
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/integrals/intpoly.py b/sympy/integrals/intpoly.py
index fc72bd2..f364cf6 100755
--- a/sympy/integrals/intpoly.py
+++ b/sympy/integrals/intpoly.py
@@ -14,6 +14,7 @@
 from functools import cmp_to_key
 
 from sympy.core import S, diff, Expr, Symbol
+from sympy.core.compatibility import default_sort_key
 
 from sympy.geometry import Segment2D, Polygon, Point
 from sympy.abc import x, y
@@ -556,7 +557,7 @@ def decompose(expr, separate=False):
     >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5)
     {1: x + y, 2: x**2 + x*y, 5: x**3*y**2 + y**5}
     >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5, True)
-    [x, y, x**2, y**5, x*y, x**3*y**2]
+    [x, x**2, y, y**5, x*y, x**3*y**2]
     """
     expr = S(expr)
     poly_dict = {}
@@ -569,7 +570,8 @@ def decompose(expr, separate=False):
             degrees = [(sum(degree_list(monom, *symbols)), monom)
                        for monom in expr.args]
             if separate:
-                return [monom[1] for monom in degrees]
+                return sorted([monom[1] for monom in degrees],
+                              key=default_sort_key)
             else:
                 for monom in degrees:
                     degree, term = monom
@@ -593,7 +595,7 @@ def decompose(expr, separate=False):
         poly_dict[0] = expr
 
     if separate:
-        return list(poly_dict.values())
+        return sorted(poly_dict.values(), key=default_sort_key)
     return poly_dict
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/integrals/tests/test_intpoly.py b/sympy/integrals/tests/test_intpoly.py
index 1e0cc29..9e2e3cb 100755
--- a/sympy/integrals/tests/test_intpoly.py
+++ b/sympy/integrals/tests/test_intpoly.py
@@ -30,11 +30,11 @@ def test_decompose():
     assert decompose(x ** 2, True) == [x ** 2]
     assert decompose(x * y, True) == [x * y]
     assert decompose(x + y, True) == [x, y]
-    assert decompose(x ** 2 + y, True) == [y, x ** 2]
-    assert decompose(8 * x ** 2 + 4 * y + 7, True) == [7, 4*y, 8*x**2]
+    assert decompose(x ** 2 + y, True) == [x ** 2, y]
+    assert decompose(8 * x ** 2 + 4 * y + 7, True) == [7, 8*x**2, 4*y]
     assert decompose(x ** 2 + 3 * y * x, True) == [x ** 2, 3 * x * y]
     assert decompose(9 * x ** 2 + y + 4 * x + x ** 3 + y ** 2 * x + 3, True) == \
-           [3, y, x**3, 4*x, 9*x**2, x*y**2]
+           [3, 4*x, 9*x**2, x**3, y, x*y**2]
 
 
 def test_best_origin():
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/integrals/intpoly.py b/sympy/integrals/intpoly.py
--- a/sympy/integrals/intpoly.py
+++ b/sympy/integrals/intpoly.py
@@ -556,7 +556,7 @@ def decompose(expr, separate=False):
     >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5)
     {1: x + y, 2: x**2 + x*y, 5: x**3*y**2 + y**5}
     >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5, True)
-    [x, y, x**2, y**5, x*y, x**3*y**2]
+    {x, x**2, y, y**5, x*y, x**3*y**2}
     """
     expr = S(expr)
     poly_dict = {}
@@ -569,7 +569,7 @@ def decompose(expr, separate=False):
             degrees = [(sum(degree_list(monom, *symbols)), monom)
                        for monom in expr.args]
             if separate:
-                return [monom[1] for monom in degrees]
+                return {monom[1] for monom in degrees}
             else:
                 for monom in degrees:
                     degree, term = monom
@@ -593,7 +593,7 @@ def decompose(expr, separate=False):
         poly_dict[0] = expr
 
     if separate:
-        return list(poly_dict.values())
+        return set(poly_dict.values())
     return poly_dict
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/integrals/tests/test_intpoly.py b/sympy/integrals/tests/test_intpoly.py
--- a/sympy/integrals/tests/test_intpoly.py
+++ b/sympy/integrals/tests/test_intpoly.py
@@ -26,15 +26,15 @@ def test_decompose():
     assert decompose(9*x**2 + y + 4*x + x**3 + y**2*x + 3) ==\
         {0: 3, 1: 4*x + y, 2: 9*x**2, 3: x**3 + x*y**2}
 
-    assert decompose(x, True) == [x]
-    assert decompose(x ** 2, True) == [x ** 2]
-    assert decompose(x * y, True) == [x * y]
-    assert decompose(x + y, True) == [x, y]
-    assert decompose(x ** 2 + y, True) == [y, x ** 2]
-    assert decompose(8 * x ** 2 + 4 * y + 7, True) == [7, 4*y, 8*x**2]
-    assert decompose(x ** 2 + 3 * y * x, True) == [x ** 2, 3 * x * y]
+    assert decompose(x, True) == {x}
+    assert decompose(x ** 2, True) == {x**2}
+    assert decompose(x * y, True) == {x * y}
+    assert decompose(x + y, True) == {x, y}
+    assert decompose(x ** 2 + y, True) == {y, x ** 2}
+    assert decompose(8 * x ** 2 + 4 * y + 7, True) == {7, 4*y, 8*x**2}
+    assert decompose(x ** 2 + 3 * y * x, True) == {x ** 2, 3 * x * y}
     assert decompose(9 * x ** 2 + y + 4 * x + x ** 3 + y ** 2 * x + 3, True) == \
-           [3, y, x**3, 4*x, 9*x**2, x*y**2]
+           {3, y, 4*x, 9*x**2, x*y**2, x**3}
 
 
 def test_best_origin():
```

</details>

<details>
<summary>Verification command at step 21 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 114
Step: 21

COMMAND
python -m pytest sympy/integrals/tests/test_intpoly.py::test_decompose -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.344
stdout:
[empty]
stderr:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
```

</details>

<details>
<summary>Verification command at step 22 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 119
Step: 22

COMMAND
ls bin/test* 2>/dev/null; ls sympy/testing 2>/dev/null | head; grep -R "def test(" -n sympy/utilities/runtests.py | head

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.326
stdout:
bin/test
bin/test_import
bin/test_import.py
bin/test_isolated
bin/test_setup.py
bin/test_travis.sh
320:def test(*paths, **kwargs):
1062:    def test(self, sort=False, timeout=False, slow=False, enhance_asserts=False):
1294:    def test(self):
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 23 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 124
Step: 23

COMMAND
bin/test sympy/integrals/tests/test_intpoly.py -k test_decompose --tb=short

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 1.063
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
random seed:        79570446
hash randomization: on (PYTHONHASHSEED=796364588)

sympy/integrals/tests/test_intpoly.py[1] F                                [FAIL]

____________________________... [truncated]
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 159
Step: 30

COMMAND
bin/test sympy/integrals/tests/test_intpoly.py -k test_decompose --tb=short

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.931
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
random seed:        72358056
hash randomization: on (PYTHONHASHSEED=4123610939)

sympy/integrals/tests/test_intpoly.py[1] .                                  [OK]

================== tests fi... [truncated]
```

</details>

<details>
<summary>Verification command at step 33 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 174
Step: 33

COMMAND
bin/test sympy/integrals/tests/test_intpoly.py --tb=short

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.888
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
random seed:        70429635
hash randomization: on (PYTHONHASHSEED=2932271742)

sympy/integrals/tests/test_intpoly.py[4] ...f                               [OK]

======== tests finished: 3 ... [truncated]
```

</details>


[Return to main report](../README.md)
