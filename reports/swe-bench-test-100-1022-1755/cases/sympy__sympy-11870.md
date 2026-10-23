# sympy__sympy-11870: trigsimp examples are graded through sinc rewrite semantics

**Primary category:** G — Golden change extends or shifts the stated scope. **Confidence:** medium.

**Recorded score:** F2P 0/1; P2P 58/58. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The agent changes the named simplifier, while the scored failure concerns an unchanged zero-safe rewrite API.

## Issue requirement

Simplify imaginary exponentials to sin(k), and ideally the divided expression to sinc(k).

## Agent change

Adds an exponential-to-trigonometric fallback in futrig, selected by operation count; it leaves sinc.rewrite(sin) unchanged.

## Golden change

Adds sin.rewrite(sinc) and makes sinc.rewrite(sin) Piecewise with a value at zero. It does not modify trigsimp.

## Observed failure

The scored test_sinc fails because sinc(x).rewrite(sin) is sin(x)/x rather than Piecewise((sin(x)/x, Ne(x, 0)), (1, True)). All 58 P2P tests pass.

## Scope and attribution

This is not adequately explained by wrong file alone. There is a material issue/golden API mismatch, plus the agent leaves the explicit sinc part of the request incomplete. The exact zero behavior belongs to the accepted rewrite design rather than the shown trigsimp reproduction.

## Verification in the trajectory

The agent adds simplifier tests and the recorded native run reports 23 passed and one expected failure. These tests cover its chosen simplifier behavior, not the golden rewrite contract.

## What would improve the outcome

Separate the two requested transformations and verify each explicitly, including zero-domain behavior. Audit whether benchmark tests actually assess the named API in the issue.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sympy__sympy-11870`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 79 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 79 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sympy__sympy-11870.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
simplifying exponential -> trig identities
```
f = 1 / 2 * (-I*exp(I*k) + I*exp(-I*k))
trigsimp(f)
```

Ideally, this would yield `sin(k)`. Is there a way to do this?

As a corollary, it would be awesome if 

```
f = 1 / 2 / k* (-I*exp(I*k) + I*exp(-I*k))
trigsimp(f)
```

could yield `sinc(k)`. Thank you for your consideration!
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
rewrite can be used:

```
>>> f = S(1) / 2 * (-I*exp(I*k) + I*exp(-I*k))
>>> f.rewrite(sin).simplify()
sin(k)
```

Thank you for that suggestion!

> On Nov 17, 2016, at 01:06, Kalevi Suominen notifications@github.com wrote:
> 
> rewrite can be used:
> 
> > > > f = S(1) / 2 \* (-I_exp(I_k) + I_exp(-I_k))
> > > > f.rewrite(sin).simplify()
> > > > sin(k)
> 
> —
> You are receiving this because you authored the thread.
> Reply to this email directly, view it on GitHub, or mute the thread.

Too bad this doesn't work as expected:

```
ω = sym.symbols('ω', real=True)
k = sym.symbols('k', real=True)
f = 1 / 2 / π * sym.exp(sym.I * ω * k)
F = sym.integrate(f, (ω, -π, π))
F.rewrite(sym.sinc).simplify()
```

It does not produce the desired sinc function in the equation.

It seems that rewrite for sinc has not been implemented.
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/simplify/trigsimp.py b/sympy/simplify/trigsimp.py
index 34a3bab..6566ca8 100755
--- a/sympy/simplify/trigsimp.py
+++ b/sympy/simplify/trigsimp.py
@@ -1104,6 +1104,13 @@ def futrig(e, **kwargs):
     if e != old and e.is_Mul and e.args[0].is_Rational:
         # redistribute leading coeff on 2-arg Add
         e = Mul(*e.as_coeff_Mul())
+
+    if e.has(exp) and e.has(I):
+        # Rewriting imaginary exponentials as trigonometric functions can
+        # lead to a simpler expression, e.g. sin(x) from exponentials.
+        alt = futrig(e.rewrite(cos), **kwargs)
+        if count_ops(alt) < count_ops(e):
+            e = alt
     return e
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/simplify/tests/test_trigsimp.py b/sympy/simplify/tests/test_trigsimp.py
index 67694b4..96393c6 100755
--- a/sympy/simplify/tests/test_trigsimp.py
+++ b/sympy/simplify/tests/test_trigsimp.py
@@ -365,6 +365,12 @@ def valid(a, b):
 
     assert exptrigsimp(exp(x) + exp(-x)) == 2*cosh(x)
     assert exptrigsimp(exp(x) - exp(-x)) == 2*sinh(x)
+
+    # exponential expressions with imaginary args can be written in
+    # terms of trigonometric functions
+    e = S.Half*(-I*exp(I*k) + I*exp(-I*k))
+    assert trigsimp(e) == sin(k)
+    assert trigsimp(e/k) == sin(k)/k
     assert exptrigsimp((2*exp(x)-2*exp(-x))/(exp(x)+exp(-x))) == 2*tanh(x)
     assert exptrigsimp((2*exp(2*x)-2)/(exp(2*x)+1)) == 2*tanh(x)
     e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/functions/elementary/trigonometric.py b/sympy/functions/elementary/trigonometric.py
--- a/sympy/functions/elementary/trigonometric.py
+++ b/sympy/functions/elementary/trigonometric.py
@@ -16,6 +16,8 @@
 from sympy.sets.sets import FiniteSet
 from sympy.utilities.iterables import numbered_symbols
 from sympy.core.compatibility import range
+from sympy.core.relational import Ne
+from sympy.functions.elementary.piecewise import Piecewise
 
 ###############################################################################
 ########################## TRIGONOMETRIC FUNCTIONS ############################
@@ -400,6 +402,9 @@ def _eval_rewrite_as_csc(self, arg):
     def _eval_rewrite_as_sec(self, arg):
         return 1 / sec(arg - S.Pi / 2, evaluate=False)
 
+    def _eval_rewrite_as_sinc(self, arg):
+        return arg*sinc(arg)
+
     def _eval_conjugate(self):
         return self.func(self.args[0].conjugate())
 
@@ -1789,7 +1794,7 @@ def _eval_rewrite_as_jn(self, arg):
         return jn(0, arg)
 
     def _eval_rewrite_as_sin(self, arg):
-        return sin(arg) / arg
+        return Piecewise((sin(arg)/arg, Ne(arg, 0)), (1, True))
 
 
 ###############################################################################
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/functions/elementary/tests/test_trigonometric.py b/sympy/functions/elementary/tests/test_trigonometric.py
--- a/sympy/functions/elementary/tests/test_trigonometric.py
+++ b/sympy/functions/elementary/tests/test_trigonometric.py
@@ -6,6 +6,8 @@
         AccumBounds)
 from sympy.core.compatibility import range
 from sympy.utilities.pytest import XFAIL, slow, raises
+from sympy.core.relational import Ne, Eq
+from sympy.functions.elementary.piecewise import Piecewise
 
 x, y, z = symbols('x y z')
 r = Symbol('r', real=True)
@@ -704,7 +706,7 @@ def test_sinc():
     assert sinc(x).series() == 1 - x**2/6 + x**4/120 + O(x**6)
 
     assert sinc(x).rewrite(jn) == jn(0, x)
-    assert sinc(x).rewrite(sin) == sin(x) / x
+    assert sinc(x).rewrite(sin) == Piecewise((sin(x)/x, Ne(x, 0)), (1, True))
 
 
 def test_asin():
@@ -1507,6 +1509,14 @@ def test_trig_period():
     assert tan(3*x).period(y) == S.Zero
     raises(NotImplementedError, lambda: sin(x**2).period(x))
 
+
 def test_issue_7171():
     assert sin(x).rewrite(sqrt) == sin(x)
     assert sin(x).rewrite(pow) == sin(x)
+
+
+def test_issue_11864():
+    w, k = symbols('w, k', real=True)
+    F = Piecewise((1, Eq(2*pi*k, 0)), (sin(pi*k)/(pi*k), True))
+    soln = Piecewise((1, Eq(2*pi*k, 0)), (sinc(pi*k), True))
+    assert F.rewrite(sinc) == soln
```

</details>

<details>
<summary>Verification command at step 36 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 210
Step: 36

COMMAND
python -m pytest sympy/simplify/tests/test_trigsimp.py -q -k 'exptrigsimp or trigsimp1 or trigsimp2 or fu' 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.351
stdout:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 37 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 215
Step: 37

COMMAND
python bin/test sympy/simplify/tests/test_trigsimp.py -k exptrigsimp 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.279
stdout:
/testbed/sympy/core/basic.py:3: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
  from collections import Mapping, defaultdict
/testbed/sympy/core/containers.py:271: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will s... [truncated]
```

</details>

<details>
<summary>Verification command at step 38 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 220
Step: 38

COMMAND
python bin/test sympy/simplify/tests/test_trigsimp.py 2>&1 | tail -12

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 4.660
stdout:
architecture:       64-bit
cache:              yes
ground types:       python 
numpy:              None
random seed:        4669498
hash randomization: on (PYTHONHASHSEED=3229324097)

sympy/simplify/tests/test_trigsimp.py[24] ......./testbed/sympy/polys/agca/modules.py:351: SyntaxWarning: "is" with a literal. Did you mean "=="?
  elif elem is 0:
...............f.          [OK]

======== tests finished: 23 passed, 1 expected to fail, i... [truncated]
```

</details>


[Return to main report](../README.md)
