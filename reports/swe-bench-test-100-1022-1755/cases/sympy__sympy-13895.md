# sympy__sympy-13895: Preventing a simplification masks incorrect negative-base rational powers

**Primary category:** R — Symptom fix; underlying cause remains. **Confidence:** high.

**Recorded score:** F2P 1/2; P2P 82/82. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The reproduction agrees after the patch, but the underlying numeric power operation remains wrong.

## Issue requirement

An expression changes numeric value after simplify at x=9/5.

## Agent change

Restricts Pow.as_numer_denom when the numerator sign is unknown, preventing the transformation that exposed the inconsistency.

## Golden change

Repairs Integer._eval_power: factor the positive magnitude and handle the negative-one phase separately for rational exponents.

## Observed failure

The issue-specific consistency test passes, but test_powers_Integer still fails for (-8)**Rational(2, 5). All 82 P2P tests pass.

## Scope and attribution

This is a clear root-cause miss: making two paths agree does not show either is mathematically correct. The primitive negative-base operation is independently defective.

## Verification in the trajectory

Most of the budget is spent on exploratory expressions. The final source edit occurs at step 39 and prints equal results for the reproduction; no independent primitive-value check is established.

## What would improve the outcome

Use an independent numeric/branch oracle and test primitive rational powers before disabling higher-level rewrites. Include positive/negative bases and multiple rational exponents.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 82 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 82 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 82 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-13895/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-13895/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-13895/run_instance.log) | — |
| trajectory | [sympy__sympy-13895.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/sympy__sympy-13895.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

```text
(-x/4 - S(1)/12)**x - 1 simplifies to an inequivalent expression
    >>> from sympy import *
    >>> x = Symbol('x')
    >>> e = (-x/4 - S(1)/12)**x - 1
    >>> e
    (-x/4 - 1/12)**x - 1
    >>> f = simplify(e)
    >>> f
    12**(-x)*(-12**x + (-3*x - 1)**x)
    >>> a = S(9)/5
    >>> simplify(e.subs(x,a))
    -1 - 32*15**(1/5)*2**(2/5)/225
    >>> simplify(f.subs(x,a))
    -1 - 32*(-1)**(4/5)*60**(1/5)/225
    >>> N(e.subs(x,a))
    -1.32255049319339
    >>> N(f.subs(x,a))
    -0.739051169462523 - 0.189590423018741*I
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
The expressions really are equivalent, `simplify` is not to blame.  SymPy is inconsistent when raising negative numbers to the power of 9/5 (and probably other rational powers). 
```
>>> (-S(1))**(S(9)/5)
-(-1)**(4/5)                  #  complex number as a result 
>>> (-S(4))**(S(9)/5)
-8*2**(3/5)                  # the result is real
```
In a way, both are reasonable. The computation starts by writing 9/5 as 1 + 4/5. Then we get the base factored out, and are left with `(-1)**(4/5)` or `(-4)**(4/5)`. Somehow, the first is left alone while in the second, noticing that 4 is a square, SymPy does further manipulations, ending up by raising (-4) to the power of 4 and thus canceling the minus sign. So we get the second result.  

Can it be accepted that the expression is multi-valued and which of the possible values is chosen is arbitrary? But one perhaps would like more consistency on this.
OK, "inequivalent" was the wrong word. But is it reasonable to expect sympy to try to keep the same complex root choice through simplification?
Yes, I think there should be consistency there.  The issue is at the level of SymPy taking in an object like (-1)**(S(4)/5) and parsing it into an expression tree. The trees are formed in significantly different ways for different exponents: 
```
>>> for k in range(1, 5):
...     srepr((-4)**(S(k)/5))
'Pow(Integer(-4), Rational(1, 5))'    #  complex
'Pow(Integer(-4), Rational(2, 5))'    # complex 
'Mul(Integer(2), Pow(Integer(-2), Rational(1, 5)))'   # complex, factoring out 2 is okay
'Mul(Integer(2), Pow(Integer(2), Rational(3, 5)))'    # real, where did the minus sign go? 
```
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/core/power.py b/sympy/core/power.py
index b2832dd..2ccf550 100755
--- a/sympy/core/power.py
+++ b/sympy/core/power.py
@@ -1179,6 +1179,12 @@ def as_numer_denom(self):
         if not (d.is_real or int_exp):
             n = base
             d = S.One
+        # The numerator cannot be separated from the denominator if its
+        # sign is unknown unless the exponent is an integer, e.g.
+        # ((a + b)/c)**x != (a + b)**x/c**x in general.
+        if not (n.is_nonnegative or n.is_negative or int_exp):
+            n = base
+            d = S.One
         dnonpos = d.is_nonpositive
         if dnonpos:
             n, d = -n, -d
```

</details>

<details>
<summary>Submitted test patch</summary>

```text

```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -2248,11 +2248,9 @@ def _eval_power(self, expt):
         if p is not False:
             dict = {p[0]: p[1]}
         else:
-            dict = Integer(self).factors(limit=2**15)
+            dict = Integer(b_pos).factors(limit=2**15)
 
         # now process the dict of factors
-        if self.is_negative:
-            dict[-1] = 1
         out_int = 1  # integer part
         out_rad = 1  # extracted radicals
         sqr_int = 1
@@ -2282,10 +2280,12 @@ def _eval_power(self, expt):
                     break
         for k, v in sqr_dict.items():
             sqr_int *= k**(v//sqr_gcd)
-        if sqr_int == self and out_int == 1 and out_rad == 1:
+        if sqr_int == b_pos and out_int == 1 and out_rad == 1:
             result = None
         else:
             result = out_int*out_rad*Pow(sqr_int, Rational(sqr_gcd, expt.q))
+            if self.is_negative:
+                result *= Pow(S.NegativeOne, expt)
         return result
 
     def _eval_is_prime(self):
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/core/tests/test_numbers.py b/sympy/core/tests/test_numbers.py
--- a/sympy/core/tests/test_numbers.py
+++ b/sympy/core/tests/test_numbers.py
@@ -1021,6 +1021,12 @@ def test_powers_Integer():
     assert (-3) ** Rational(-2, 3) == \
         -(-1)**Rational(1, 3)*3**Rational(1, 3)/3
 
+    # negative base and rational power with some simplification
+    assert (-8) ** Rational(2, 5) == \
+        2*(-1)**Rational(2, 5)*2**Rational(1, 5)
+    assert (-4) ** Rational(9, 5) == \
+        -8*(-1)**Rational(4, 5)*2**Rational(3, 5)
+
     assert S(1234).factors() == {617: 1, 2: 1}
     assert Rational(2*3, 3*5*7).factors() == {2: 1, 5: -1, 7: -1}
 
@@ -1194,6 +1200,14 @@ def test_issue_3449():
     assert sqrt(x - 1).subs(x, 5) == 2
 
 
+def test_issue_13890():
+    x = Symbol("x")
+    e = (-x/4 - S(1)/12)**x - 1
+    f = simplify(e)
+    a = S(9)/5
+    assert abs(e.subs(x,a).evalf() - f.subs(x,a).evalf()) < 1e-15
+
+
 def test_Integer_factors():
     def F(i):
         return Integer(i).factors()
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 195
Step: 39

COMMAND
cd /testbed && python - <<'EOF'
p = 'sympy/core/power.py'
s = open(p).read()
old = '''        if not (d.is_real or int_exp):
            n = base
            d = S.One
        dnonpos = d.is_nonpositive'''
new = '''        if not (d.is_real or int_exp):
            n = base
            d = S.One
        # The numerator cannot be separated from the denominator if its
        # sign is unknown unless the exponent is an integer, e.g.
        # ((a + b)/c)**x != (a + b)**x/c**x in general.
        if not (n.is_nonnegative or n.is_negative or int_exp):
            n = base
            d = S.One
        dnonpos = d.is_nonpositive'''
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, new))
EOF
python -m pytest sympy/core/tests/test_eval_power.py -q 2>&1 | tail -5
python -c "
from sympy import *
x = Symbol('x')
e = (-x/4 - S(1)/12)**x - 1
f = simplify(e)
print(f)
a = S(9)/5
print(simplify(e.subs(x, a)))
print(simplify(f.subs(x, a)))
assert N(e.subs(x, a)) == N(f.subs(x, a))
" 2>&1 | tail -5

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.767
stdout:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
/testbed/sympy/plotting/plot.py:28: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
  from collections import Callable
(-x/4 - 1/12)**x - 1
-1 - 32*15**(1/5)*2**(2/5)/225
-1 - 32*15**(1/5)*2**(2/5)/225
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
