# sympy__sympy-16281: A subjective layout request permits choices that exact snapshots reject

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P 0/2; P2P 120/120. **Steps:** 37/40. Counts are harness classifications, subject to the report's parser caveats.

The agent removes the complained-about bottom row; gold retains it and changes the bar geometry differently.

## Issue requirement

The Product sign is too wide, has an unwanted empty bottom row, and might benefit from an extended top bar; the issue expresses uncertainty about appearance.

## Agent change

Uses width=func_height+1, extends the top bar and changes the number of vertical rows from func_height+1 to func_height.

## Golden change

Retains the old width formula and row count, moves the vertical strokes inward by two columns and extends the bar.

## Observed failure

test_pretty_product and test_issue_6359 fail exact ASCII/Unicode snapshots; all 120 P2P tests pass.

## Scope and attribution

The chosen golden geometry is not uniquely recoverable from the prose, and retaining the bottom row conflicts with one explicit complaint. This is an underspecified layout contract with potential composition risks, not evidence that every new layout is equivalent.

## Verification in the trajectory

The agent updates product snapshots to its own output. One attempted keyword selection runs zero tests; a later full native run reports 123 passed and two exceptions.

## What would improve the outcome

Define layout invariants or approved snapshots before treating aesthetic output as unique. Test nested products, powers and ASCII/Unicode independently without automatically replacing expected strings.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sympy__sympy-16281`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 88 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 88 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sympy__sympy-16281.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
Product pretty print could be improved
This is what the pretty printing for `Product` looks like:

```
>>> pprint(Product(1, (n, 1, oo)))
  ∞
┬───┬
│   │ 1
│   │
n = 1
>>> pprint(Product(1/n, (n, 1, oo)))
   ∞
┬──────┬
│      │ 1
│      │ ─
│      │ n
│      │
 n = 1
>>> pprint(Product(1/n**2, (n, 1, oo)))
    ∞
┬────────┬
│        │ 1
│        │ ──
│        │  2
│        │ n
│        │
  n = 1
>>> pprint(Product(1, (n, 1, oo)), use_unicode=False)
  oo
_____
|   | 1
|   |
n = 1
>>> pprint(Product(1/n, (n, 1, oo)), use_unicode=False)
   oo
________
|      | 1
|      | -
|      | n
|      |
 n = 1
>>> pprint(Product(1/n**2, (n, 1, oo)), use_unicode=False)
    oo
__________
|        | 1
|        | --
|        |  2
|        | n
|        |
  n = 1
```

(if those don't look good in your browser copy paste them into the terminal)

This could be improved:

- Why is there always an empty line at the bottom of the ∏? Keeping everything below the horizontal line is good, but the bottom looks asymmetric, and it makes the ∏ bigger than it needs to be.

- The ∏ is too fat IMO. 

- It might look better if we extended the top bar. I'm unsure about this. 

Compare this

```
    ∞
─┬─────┬─
 │     │  1
 │     │  ──
 │     │   2
 │     │  n
  n = 1
```

That's still almost twice as wide as the equivalent Sum, but if you make it much skinnier it starts to look bad.

```
  ∞
 ____
 ╲
  ╲   1
   ╲  ──
   ╱   2
  ╱   n
 ╱
 ‾‾‾‾
n = 1
```
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
index 6d3cc05..2f5faa0 100755
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -490,11 +490,14 @@ def _print_Product(self, expr):
         sign_height = 0
 
         for lim in expr.limits:
-            width = (func_height + 2) * 5 // 3 - 2
+            width = func_height + 1
             sign_lines = []
-            sign_lines.append(corner_chr + (horizontal_chr*width) + corner_chr)
-            for i in range(func_height + 1):
-                sign_lines.append(vertical_chr + (' '*width) + vertical_chr)
+            sign_lines.append(horizontal_chr + corner_chr +
+                              (horizontal_chr*width) + corner_chr +
+                              horizontal_chr)
+            for i in range(func_height):
+                sign_lines.append(' ' + vertical_chr + (' '*width) +
+                                  vertical_chr + ' ')
 
             pretty_sign = stringPict('')
             pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
index cfaa65a..1ae18c8 100755
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -2053,26 +2053,24 @@ def test_pretty_product():
 
     unicode_str = \
 u("""\
-    l           \n\
-┬────────┬      \n\
-│        │  ⎛ 2⎞\n\
-│        │  ⎜n ⎟\n\
-│        │ f⎜──⎟\n\
-│        │  ⎝9 ⎠\n\
-│        │      \n\
-       2        \n\
-  n = k         """)
-    ascii_str = \
-"""\
-    l           \n\
-__________      \n\
-|        |  / 2\\\n\
-|        |  |n |\n\
-|        | f|--|\n\
-|        |  \\9 /\n\
-|        |      \n\
-       2        \n\
-  n = k         """
+    l          \n\
+─┬─────┬─      \n\
+ │     │   ⎛ 2⎞\n\
+ │     │   ⎜n ⎟\n\
+ │     │  f⎜──⎟\n\
+ │     │   ⎝9 ⎠\n\
+       2       \n\
+  n = k        """)
+    ascii_str = \
+"""\
+    l          \n\
+_________      \n\
+ |     |   / 2\\\n\
+ |     |   |n |\n\
+ |     |  f|--|\n\
+ |     |   \\9 /\n\
+       2       \n\
+  n = k        """
 
     assert pretty(expr) == ascii_str
     assert upretty(expr) == unicode_str
@@ -2081,26 +2079,24 @@ def test_pretty_product():
 
     unicode_str = \
 u("""\
-    m          l           \n\
-┬────────┬ ┬────────┬      \n\
-│        │ │        │  ⎛ 2⎞\n\
-│        │ │        │  ⎜n ⎟\n\
-│        │ │        │ f⎜──⎟\n\
-│        │ │        │  ⎝9 ⎠\n\
-│        │ │        │      \n\
-  l = 1           2        \n\
-             n = k         """)
+    m         l          \n\
+─┬─────┬─ ─┬─────┬─      \n\
+ │     │   │     │   ⎛ 2⎞\n\
+ │     │   │     │   ⎜n ⎟\n\
+ │     │   │     │  f⎜──⎟\n\
+ │     │   │     │   ⎝9 ⎠\n\
+  l = 1          2       \n\
+            n = k        """)
     ascii_str = \
 """\
-    m          l           \n\
-__________ __________      \n\
-|        | |        |  / 2\\\n\
-|        | |        |  |n |\n\
-|        | |        | f|--|\n\
-|        | |        |  \\9 /\n\
-|        | |        |      \n\
-  l = 1           2        \n\
-             n = k         """
+    m         l          \n\
+_________ _________      \n\
+ |     |   |     |   / 2\\\n\
+ |     |   |     |   |n |\n\
+ |     |   |     |  f|--|\n\
+ |     |   |     |   \\9 /\n\
+  l = 1          2       \n\
+            n = k        """
 
     assert pretty(expr) == ascii_str
     assert upretty(expr) == unicode_str
@@ -5511,23 +5507,21 @@ def test_issue_6359():
 
     assert pretty(Product(x**2, (x, 1, 2))**2) == \
 """\
-           2
-/  2      \\ \n\
-|______   | \n\
-||    |  2| \n\
-||    | x | \n\
-||    |   | \n\
-\\x = 1    / \
+            2
+/   2      \\ \n\
+|_______   | \n\
+| |   |   2| \n\
+| |   |  x | \n\
+\\ x = 1    / \
 """
     assert upretty(Product(x**2, (x, 1, 2))**2) == \
 u("""\
-           2
-⎛  2      ⎞ \n\
-⎜┬────┬   ⎟ \n\
-⎜│    │  2⎟ \n\
-⎜│    │ x ⎟ \n\
-⎜│    │   ⎟ \n\
-⎝x = 1    ⎠ \
+            2
+⎛   2      ⎞ \n\
+⎜─┬───┬─   ⎟ \n\
+⎜ │   │   2⎟ \n\
+⎜ │   │  x ⎟ \n\
+⎝ x = 1    ⎠ \
 """)
 
     f = Function('f')
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -491,10 +491,9 @@ def _print_Product(self, expr):
 
         for lim in expr.limits:
             width = (func_height + 2) * 5 // 3 - 2
-            sign_lines = []
-            sign_lines.append(corner_chr + (horizontal_chr*width) + corner_chr)
-            for i in range(func_height + 1):
-                sign_lines.append(vertical_chr + (' '*width) + vertical_chr)
+            sign_lines = [horizontal_chr + corner_chr + (horizontal_chr * (width-2)) + corner_chr + horizontal_chr]
+            for _ in range(func_height + 1):
+                sign_lines.append(' ' + vertical_chr + (' ' * (width-2)) + vertical_chr + ' ')
 
             pretty_sign = stringPict('')
             pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -2054,51 +2054,48 @@ def test_pretty_product():
     unicode_str = \
 u("""\
     l           \n\
-┬────────┬      \n\
-│        │  ⎛ 2⎞\n\
-│        │  ⎜n ⎟\n\
-│        │ f⎜──⎟\n\
-│        │  ⎝9 ⎠\n\
-│        │      \n\
+─┬──────┬─      \n\
+ │      │   ⎛ 2⎞\n\
+ │      │   ⎜n ⎟\n\
+ │      │  f⎜──⎟\n\
+ │      │   ⎝9 ⎠\n\
+ │      │       \n\
        2        \n\
   n = k         """)
     ascii_str = \
 """\
     l           \n\
 __________      \n\
-|        |  / 2\\\n\
-|        |  |n |\n\
-|        | f|--|\n\
-|        |  \\9 /\n\
-|        |      \n\
+ |      |   / 2\\\n\
+ |      |   |n |\n\
+ |      |  f|--|\n\
+ |      |   \\9 /\n\
+ |      |       \n\
        2        \n\
   n = k         """
 
-    assert pretty(expr) == ascii_str
-    assert upretty(expr) == unicode_str
-
     expr = Product(f((n/3)**2), (n, k**2, l), (l, 1, m))
 
     unicode_str = \
 u("""\
     m          l           \n\
-┬────────┬ ┬────────┬      \n\
-│        │ │        │  ⎛ 2⎞\n\
-│        │ │        │  ⎜n ⎟\n\
-│        │ │        │ f⎜──⎟\n\
-│        │ │        │  ⎝9 ⎠\n\
-│        │ │        │      \n\
+─┬──────┬─ ─┬──────┬─      \n\
+ │      │   │      │   ⎛ 2⎞\n\
+ │      │   │      │   ⎜n ⎟\n\
+ │      │   │      │  f⎜──⎟\n\
+ │      │   │      │   ⎝9 ⎠\n\
+ │      │   │      │       \n\
   l = 1           2        \n\
              n = k         """)
     ascii_str = \
 """\
     m          l           \n\
 __________ __________      \n\
-|        | |        |  / 2\\\n\
-|        | |        |  |n |\n\
-|        | |        | f|--|\n\
-|        | |        |  \\9 /\n\
-|        | |        |      \n\
+ |      |   |      |   / 2\\\n\
+ |      |   |      |   |n |\n\
+ |      |   |      |  f|--|\n\
+ |      |   |      |   \\9 /\n\
+ |      |   |      |       \n\
   l = 1           2        \n\
              n = k         """
 
@@ -5514,19 +5511,19 @@ def test_issue_6359():
            2
 /  2      \\ \n\
 |______   | \n\
-||    |  2| \n\
-||    | x | \n\
-||    |   | \n\
+| |  |   2| \n\
+| |  |  x | \n\
+| |  |    | \n\
 \\x = 1    / \
 """
     assert upretty(Product(x**2, (x, 1, 2))**2) == \
 u("""\
            2
 ⎛  2      ⎞ \n\
-⎜┬────┬   ⎟ \n\
-⎜│    │  2⎟ \n\
-⎜│    │ x ⎟ \n\
-⎜│    │   ⎟ \n\
+⎜─┬──┬─   ⎟ \n\
+⎜ │  │   2⎟ \n\
+⎜ │  │  x ⎟ \n\
+⎜ │  │    ⎟ \n\
 ⎝x = 1    ⎠ \
 """)
```

</details>

<details>
<summary>Verification command at step 28 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 185
Step: 28

COMMAND
python -m pytest sympy/printing/pretty/tests/test_pretty.py -k "product or Product" -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.352
stdout:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 29 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 190
Step: 29

COMMAND
python -c "import sympy; sympy.test('sympy/printing/pretty/tests/test_pretty.py', kw='product or Product')" 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.068
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
numpy:              None
random seed:        74405207
hash randomization: on (PYTHONHASHSEED=2147930502)


================== tests finished: 0 passed, in 0.00 seconds ===================
st... [truncated]
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 195
Step: 30

COMMAND
python -c "import sympy; sympy.test('sympy/printing/pretty/tests/test_pretty.py', test='test_pretty_product')" 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.893
stdout:
  elif transformation.name is 'cylindrical':
sympy/printing/pretty/tests/test_pretty.py[125] ................................
......................................................../testbed/sympy/polys/agca/modules.py:360: SyntaxWarning: "is" with a literal. Did you mean "=="?
  elif elem is 0:
..........E.E...........
.............                                                             [FAIL]

__________________________________... [truncated]
```

</details>

<details>
<summary>Verification command at step 35 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 226
Step: 35

COMMAND
python -c "import sympy; sympy.test('sympy/printing/pretty/tests/test_pretty.py')" 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.775
stdout:
  File "/testbed/sympy/core/sympify.py", line 384, in sympify
    expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
  File "/testbed/sympy/parsing/sympy_parser.py", line 964, in parse_expr
    code = compile(evaluateFalse(code), '<string>', 'eval')
ValueError: Name node can't be used with 'False' constant

========== tests finished: 123 passed, 2 exceptions, in 0.71 seconds ===========
DO *NOT... [truncated]
```

</details>


[Return to main report](../README.md)
