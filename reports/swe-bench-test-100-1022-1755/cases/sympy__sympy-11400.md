# sympy__sympy-11400: A broader rewrite framework leaves Ne undefined

**Primary category:** I — Implementation error or introduced regression. **Confidence:** high.

**Recorded score:** F2P 1/2; P2P 29/29. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The submitted sinc code path raises NameError due to a missing import.

## Issue requirement

Generate C code for sinc using a guarded sin(x)/x expression with value 1 at zero.

## Agent change

Adds a generic code-printer rewrite hook, a sinc-to-Piecewise rewrite in trigonometric.py, and relational printing changes. The new rewrite calls Ne without importing it.

## Golden change

Adds a local _print_sinc method and relational printing directly in the C printer.

## Observed failure

test_ccode_Relational passes, but test_ccode_sinc raises NameError: name Ne is not defined in the new rewrite method.

## Scope and attribution

The agent found the right mathematical structure. This failure is an unfinished implementation, amplified by a wider change than required; no scope mismatch is needed to explain it.

## Verification in the trajectory

Late native SymPy runs still show a traceback, although the command exit code is zero. The final response explicitly identifies the missing import and admits the patch is incomplete at step 40.

## What would improve the outcome

Prefer the local printer extension for a local feature. Run ccode(sinc(x)) in a fresh process immediately after the change, and repair or revert a known failure before spending the final step.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sympy__sympy-11400`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 78 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 78 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sympy__sympy-11400.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
ccode(sinc(x)) doesn't work
```
In [30]: ccode(sinc(x))
Out[30]: '// Not supported in C:\n// sinc\nsinc(x)'
```

I don't think `math.h` has `sinc`, but it could print

```
In [38]: ccode(Piecewise((sin(theta)/theta, Ne(theta, 0)), (1, True)))
Out[38]: '((Ne(theta, 0)) ? (\n   sin(theta)/theta\n)\n: (\n   1\n))'
```
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
@asmeurer I would like to fix this issue. Should I work upon  the codegen.py file ? If there's something else tell me how to start ?

The relevant file is sympy/printing/ccode.py

@asmeurer I am new here. I would like to work on this issue. Please tell me how to start?

Since there are two people asking, maybe one person can try #11286 which is very similar, maybe even easier.
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/functions/elementary/trigonometric.py b/sympy/functions/elementary/trigonometric.py
index 682a57f..dbd6855 100755
--- a/sympy/functions/elementary/trigonometric.py
+++ b/sympy/functions/elementary/trigonometric.py
@@ -1699,6 +1699,10 @@ def _eval_rewrite_as_jn(self, arg):
     def _eval_rewrite_as_sin(self, arg):
         return sin(arg) / arg
 
+    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
+        from sympy.functions import Piecewise
+        return Piecewise((sin(arg)/arg, Ne(arg, 0)), (S.One, True))
+
 
 ###############################################################################
 ########################### TRIGONOMETRIC INVERSES ############################

diff --git a/sympy/printing/ccode.py b/sympy/printing/ccode.py
index 30a07a7..5d1b9f7 100755
--- a/sympy/printing/ccode.py
+++ b/sympy/printing/ccode.py
@@ -88,6 +88,8 @@ class CCodePrinter(CodePrinter):
     printmethod = "_ccode"
     language = "C"
 
+    _rewriteable = {'sinc': ('Piecewise',)}
+
     _default_settings = {
         'order': None,
         'full_prec': 'auto',
@@ -251,6 +253,17 @@ def _print_For(self, expr):
     def _print_sign(self, func):
         return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))
 
+    def _print_Relational(self, expr):
+        charmap = {
+            "==": "==",
+            "!=": "!=",
+        }
+        if expr.rel_op in charmap:
+            return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
+                               charmap[expr.rel_op],
+                               self.parenthesize(expr.rhs, precedence(expr)))
+        return super(CCodePrinter, self)._print_Relational(expr)
+
     def indent_code(self, code):
         """Accepts a string of code or a list of code lines"""
 

diff --git a/sympy/printing/codeprinter.py b/sympy/printing/codeprinter.py
index b46da62..32ac701 100755
--- a/sympy/printing/codeprinter.py
+++ b/sympy/printing/codeprinter.py
@@ -36,6 +36,11 @@ class CodePrinter(StrPrinter):
                          'error_on_reserved': False,
                          'reserved_word_suffix': '_'}
 
+    # Functions that are not directly supported by the target language can be
+    # rewritten in terms of supported ones.  The values are tuples of arguments
+    # passed to ``Basic.rewrite``.
+    _rewriteable = {}
+
     def __init__(self, settings=None):
 
         super(CodePrinter, self).__init__(settings=settings)
@@ -318,6 +323,8 @@ def _print_Function(self, expr):
                         break
             if func is not None:
                 return "%s(%s)" % (func, self.stringify(expr.args, ", "))
+        elif expr.func.__name__ in self._rewriteable:
+            return self._print(expr.rewrite(*self._rewriteable[expr.func.__name__]))
         elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
             # inlined function
             return self._print(expr._imp_(*expr.args))
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/printing/tests/test_ccode.py b/sympy/printing/tests/test_ccode.py
index d8e449d..4caddc9 100755
--- a/sympy/printing/tests/test_ccode.py
+++ b/sympy/printing/tests/test_ccode.py
@@ -1,5 +1,5 @@
 from sympy.core import (pi, oo, symbols, Rational, Integer,
-                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy, Eq)
+                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy, Eq, Ne)
 from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                              gamma, sign)
 from sympy.sets import Range
@@ -162,6 +162,22 @@ def test_ccode_Piecewise():
     raises(ValueError, lambda: ccode(expr))
 
 
+def test_ccode_sinc():
+    from sympy import sinc
+    assert ccode(sinc(x)) == (
+            "((x != 0) ? (\n"
+            "   sin(x)/x\n"
+            ")\n"
+            ": (\n"
+            "   1\n"
+            "))")
+
+
+def test_ccode_Relational():
+    assert ccode(Eq(x, y)) == "x == y"
+    assert ccode(Ne(x, y)) == "x != y"
+
+
 def test_ccode_Piecewise_deep():
     p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
     assert p == (
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/printing/ccode.py b/sympy/printing/ccode.py
--- a/sympy/printing/ccode.py
+++ b/sympy/printing/ccode.py
@@ -231,6 +231,20 @@ def _print_Symbol(self, expr):
         else:
             return name
 
+    def _print_Relational(self, expr):
+        lhs_code = self._print(expr.lhs)
+        rhs_code = self._print(expr.rhs)
+        op = expr.rel_op
+        return ("{0} {1} {2}").format(lhs_code, op, rhs_code)
+
+    def _print_sinc(self, expr):
+        from sympy.functions.elementary.trigonometric import sin
+        from sympy.core.relational import Ne
+        from sympy.functions import Piecewise
+        _piecewise = Piecewise(
+            (sin(expr.args[0]) / expr.args[0], Ne(expr.args[0], 0)), (1, True))
+        return self._print(_piecewise)
+
     def _print_AugmentedAssignment(self, expr):
         lhs_code = self._print(expr.lhs)
         op = expr.rel_op
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/printing/tests/test_ccode.py b/sympy/printing/tests/test_ccode.py
--- a/sympy/printing/tests/test_ccode.py
+++ b/sympy/printing/tests/test_ccode.py
@@ -120,6 +120,16 @@ def test_ccode_boolean():
     assert ccode((x | y) & z) == "z && (x || y)"
 
 
+def test_ccode_Relational():
+    from sympy import Eq, Ne, Le, Lt, Gt, Ge
+    assert ccode(Eq(x, y)) == "x == y"
+    assert ccode(Ne(x, y)) == "x != y"
+    assert ccode(Le(x, y)) == "x <= y"
+    assert ccode(Lt(x, y)) == "x < y"
+    assert ccode(Gt(x, y)) == "x > y"
+    assert ccode(Ge(x, y)) == "x >= y"
+
+
 def test_ccode_Piecewise():
     expr = Piecewise((x, x < 1), (x**2, True))
     assert ccode(expr) == (
@@ -162,6 +172,18 @@ def test_ccode_Piecewise():
     raises(ValueError, lambda: ccode(expr))
 
 
+def test_ccode_sinc():
+    from sympy import sinc
+    expr = sinc(x)
+    assert ccode(expr) == (
+            "((x != 0) ? (\n"
+            "   sin(x)/x\n"
+            ")\n"
+            ": (\n"
+            "   1\n"
+            "))")
+
+
 def test_ccode_Piecewise_deep():
     p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
     assert p == (
```

</details>

<details>
<summary>Verification command at step 28 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 185
Step: 28

COMMAND
python -m pytest sympy/printing/tests/test_ccode.py -k "sinc or Relational" -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.360
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
python -c "import sympy; sympy.test('sympy/printing/tests/test_ccode.py', verbose=False)" 2>&1 | tail -40

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.027
stdout:
/testbed/sympy/core/basic.py:3: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
  from collections import Mapping
/testbed/sympy/plotting/plot.py:28: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
  fr... [truncated]
```

</details>

<details>
<summary>Verification command at step 37 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 236
Step: 37

COMMAND
python -c "import sympy; sympy.test('sympy/printing/tests/test_ccode.py', verbose=False)" 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.046
stdout:
    assert ccode(sinc(x)) == (
  File "/testbed/sympy/printing/ccode.py", line 409, in ccode
    return CCodePrinter(settings).doprint(expr, assign_to)
  File "/testbed/sympy/printing/codeprinter.py", line 85, in doprint
    lines = self._print(expr).splitlines()
  File "/testbed/sympy/printing/printer.py", line 257, in _print
    return getattr(self, printmethod)(expr, *args, **kwargs)
  File "/testbed/sympy/printing/codeprinter.py",... [truncated]
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 249
Step: 39

COMMAND
python -c "import sympy; sympy.test('sympy/printing/tests/test_ccode.py', verbose=False)" 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.968
stdout:

sympy/printing/tests/test_ccode.py[32] .............E..................   [FAIL]

________________________________________________________________________________
______________ sympy/printing/tests/test_ccode.py:test_ccode_sinc ______________
  File "/testbed/sympy/printing/tests/test_ccode.py", line 167, in test_ccode_sinc
    assert ccode(sinc(x)) == (
  File "/testbed/sympy/printing/ccode.py", line 409, in ccode
    return CCodeP... [truncated]
```

</details>


[Return to main report](../README.md)
