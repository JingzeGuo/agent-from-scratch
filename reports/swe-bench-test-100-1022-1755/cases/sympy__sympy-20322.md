# sympy__sympy-20322: A ceiling workaround leaves incorrect Mul.is_integer assumptions intact

**Primary category:** R — Symptom fix; underlying cause remains. **Confidence:** high.

**Recorded score:** F2P 0/1; P2P 89/89. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The agent observes the faulty assumption but patches a consumer of it.

## Issue requirement

evaluate=False parsing changes the result of simplify for a ceiling expression.

## Agent change

Restricts when RoundFunction.eval treats a term as integral, bypassing the misleading is_integer result for some numeric expressions.

## Golden change

Reworks Mul._eval_is_integer to reason about factors and denominators without the erroneous fraction shortcut.

## Observed failure

test_Mul_is_integer still fails on Mul(2., S.Half, evaluate=False).is_integer, which must be None. All 89 P2P tests pass.

## Scope and attribution

The trajectory itself shows a nested unevaluated -3/4 term with is_integer=True. Fixing ceiling leaves that bad shared assumption available to other consumers.

## Verification in the trajectory

The agent explores the false assumption through steps 26–37, then edits RoundFunction at step 39. The same command tries pytest, which is unavailable, after printing the repaired reproduction.

## What would improve the outcome

Repair and test the shared assumption predicate first. Cover nested unevaluated products, rationals, floats and symbolic parity, then recheck floor/ceiling.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sympy__sympy-20322`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 97 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 97 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sympy__sympy-20322.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
Inconsistent behavior for sympify/simplify with ceiling
In sympy v1.5.1:
```python
In [16]: sympy.sympify('4*ceiling(x/4 - 3/4)', evaluate=False).simplify()
Out[16]: 4*ceiling(x/4 - 3/4)

In [17]: sympy.sympify('4*ceiling(x/4 - 3/4)', evaluate=True).simplify()
Out[17]: 4*ceiling(x/4 - 3/4)
```

In sympy v.1.6.2:
```python
In [16]: sympy.sympify('4*ceiling(x/4 - 3/4)', evaluate=False).simplify()
Out[16]: 4*ceiling(x/4) - 3

In [17]: sympy.sympify('4*ceiling(x/4 - 3/4)', evaluate=True).simplify()
Out [17]: 4*ceiling(x/4 - 3/4)
```

Is there a way to ensure that the behavior is consistent, even though evaluate is equal to `False` when parsing?
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
`4*ceiling(x/4) - 3` is simply wrong:
```python
>>> x = Symbol('x')
>>> (4*ceiling(x/4 - 3/4)).subs({x:0})
0
>>> (4*ceiling(x/4) - 3).subs({x:0})
-3
```
Boiling the problem further down we find that already a simpler expression is evaluated/transformed incorrectly:
```python
>>> sympy.sympify('ceiling(x-1/2)', evaluate=False)
ceiling(x) + (-1)*1*1/2
```
The `-1/2` is (under `evaluate=False`) constructed as `Mul(-1, Mul(1, Pow(2, -1)))`, for which the attribute `is_integer` is set incorrectly:
```python
>>> Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False).is_integer
True
```
Since `ceiling` takes out all integer summands from its argument, it also takes out `(-1)*1*1/2`. Maybe somebody else can look into the problem, why `is_integer` is set wrongly for this expression.
The reason `is_integer` is incorrect for the expression is because it returns here:
https://github.com/sympy/sympy/blob/1b4529a95ef641c2fc15889091b281644069d20e/sympy/core/mul.py#L1274-L1277
That is due to
```julia
In [1]: e = Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False)                                                              

In [2]: fraction(e)                                                                                                                            
Out[2]: (-1/2, 1)
```
It seems that the `1/2` is carried into the numerator by fraction giving a denominator of one.

You can see the fraction function here:
https://github.com/sympy/sympy/blob/1b4529a95ef641c2fc15889091b281644069d20e/sympy/simplify/radsimp.py#L1071-L1098

The check `term.is_Rational` will not match an unevaluated `Mul(1, Rational(1, 2), evaluate=False)` so that gets carried into the numerator.

Perhaps the root of the problem is the fact that we have unflattened args and `Mul.make_args` hasn't extracted them:
```julia
In [3]: Mul.make_args(e)                                                                                                                       
Out[3]: 
⎛      1⎞
⎜-1, 1⋅─⎟
⎝      2⎠
```
The `make_args` function does not recurse into the args:
https://github.com/sympy/sympy/blob/1b4529a95ef641c2fc15889091b281644069d20e/sympy/core/operations.py#L425-L428

I'm not sure if `make_args` should recurse. An easier fix would be to recurse into any nested Muls in `fraction`.
What about not setting `is_integer` if `evaluate=False` is set on an expression or one of its sub-expressions? Actually I think one cannot expect `is_integer` to be set correctly without evaluating.
That sounds like a good solution. As a safeguard, another one is to not remove the integer summands from `ceiling` if `evaluate=False`; it could be considered as an evaluation of ceiling w.r.t. its arguments.
There is no way to tell if `evaluate=False` was used in general when creating the `Mul`. It's also not possible in general to know if evaluating the Muls would lead to a different result without evaluating them. We should *not* evaluate the Muls as part of an assumptions query. If they are unevaluated then the user did that deliberately and it is not up to `_eval_is_integer` to evaluate them.

This was discussed when changing this to `fraction(..., exact=True)`: https://github.com/sympy/sympy/pull/19182#issuecomment-619398889

I think that using `fraction` at all is probably too much but we should certainly not replace that with something that evaluates the object.
Hm, does one really need to know whether `evaluate=False` was used? It looks like all we need is the expression tree to decide if `is_integer` is set to `True`. What about setting `is_integer=True` in a conservative way, i.e. only for these expression nodes:

- atoms: type `Integer`, constants `Zero` and `One` and symbols with appropriate assumptions
- `Add` and `Mul` if all args have `is_integer==True`
- `Pow` if base and exponent have `is_integer==True` and exponent is non-negative

I probably missed some cases, but you get the general idea. Would that work? The current implementation would only change in a few places, I guess - to a simpler form. Then also for `ceiling` we do not need to check for `evaluate=False`; its implementation could remain unchanged.
What you describe is more or less the way that it already works. You can find more detail in the (unmerged) #20090  

The code implementing this is in the `Mul._eval_is_integer` function I linked to above.
@oscarbenjamin @coproc Sorry for bugging, but are there any plans about this? We cannot use sympy with Python 3.8 anymore in our package because of this...
I explained a possible solution above:

> An easier fix would be to recurse into any nested Muls in `fraction`.

I think this just needs someone to make a PR for that. If the PR comes *very* quickly it can be included in the 1.7 release (I'm going to put out the RC just as soon as #20307 is fixed).
This diff will do it:
```diff
diff --git a/sympy/simplify/radsimp.py b/sympy/simplify/radsimp.py
index 4609da209c..879ffffdc9 100644
--- a/sympy/simplify/radsimp.py
+++ b/sympy/simplify/radsimp.py
@@ -1074,7 +1074,14 @@ def fraction(expr, exact=False):
 
     numer, denom = [], []
 
-    for term in Mul.make_args(expr):
+    def mul_args(e):
+        for term in Mul.make_args(e):
+            if term.is_Mul:
+                yield from mul_args(term)
+            else:
+                yield term
+
+    for term in mul_args(expr):
         if term.is_commutative and (term.is_Pow or isinstance(term, exp)):
             b, ex = term.as_base_exp()
             if ex.is_negative:
```
With that we get:
```python
>>> e = Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False)
>>> fraction(e) 
(-1, 2)
>>> Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False).is_integer
False
>>> sympy.sympify('ceiling(x-1/2)', evaluate=False)
ceiling(x + (-1)*1*1/2)
>>> sympy.sympify('4*ceiling(x/4 - 3/4)', evaluate=False).simplify()
4*ceiling(x/4 - 3/4)
>>> sympy.sympify('4*ceiling(x/4 - 3/4)', evaluate=True).simplify()
4*ceiling(x/4 - 3/4)
```
If someone wants to put that diff together with tests for the above into a PR then it can go in.
see pull request #20312

I have added a minimal assertion as test (with the minimal expression from above); that should suffice, I think.
Thank you both so much!
As a general rule of thumb, pretty much any function that takes a SymPy expression as input and manipulates it in some way (simplify, solve, integrate, etc.) is liable to give wrong answers if the expression was created with evaluate=False. There is code all over the place that assumes, either explicitly or implicitly, that expressions satisfy various conditions that are true after automatic evaluation happens. For example, if I am reading the PR correctly, there is some code that very reasonably assumes that Mul.make_args(expr) gives the terms of a multiplication. This is not true for evaluate=False because that disables flattening of arguments. 

If you are working with expressions created with evaluate=False, you should always evaluate them first before trying to pass them to functions like simplify(). The result of simplify would be evaluated anyway, so there's no reason to not do this.

This isn't to say I'm necessarily opposed to fixing this issue specifically, but in general I think fixes like this are untenable. There are a handful of things that should definitely work correctly with unevaluated expressions, like the printers, and some very basic expression manipulation functions. I'm less convinced it's a good idea to try to enforce this for something like the assumptions or high level simplification functions. 

This shows we really need to rethink how we represent unevaluated expressions in SymPy. The fact that you can create an expression that looks just fine, but is actually subtly "invalid" for some code is indicative that something is broken in the design. It would be better if unevaluated expressions were more explicitly separate from evaluated ones. Or if expressions just didn't evaluate as much. I'm not sure what the best solution is, just that the current situation isn't ideal.
I think that the real issue is the fact that `fraction` is not a suitable function to use within the core assumptions system. I'm sure I objected to it being introduced somewhere.

The core assumptions should be able to avoid giving True or False erroneously because of unevaluated expressions.
In fact here's a worse form of the bug:
```julia
In [1]: x = Symbol('x', rational=True)                                                                                                                                            

In [2]: fraction(x)                                                                                                                                                               
Out[2]: (x, 1)

In [3]: y = Symbol('y', rational=True)                                                                                                                                            

In [4]: (x*y).is_integer                                                                                                                                                          
Out[4]: True
```
Here's a better fix:
```diff
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
index 46f310b122..01db7d951b 100644
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -1271,18 +1271,34 @@ def _eval_is_integer(self):
             return False
 
         # use exact=True to avoid recomputing num or den
-        n, d = fraction(self, exact=True)
-        if is_rational:
-            if d is S.One:
-                return True
-        if d.is_even:
-            if d.is_prime:  # literal or symbolic 2
-                return n.is_even
-            if n.is_odd:
-                return False  # true even if d = 0
-        if n == d:
-            return fuzzy_and([not bool(self.atoms(Float)),
-            fuzzy_not(d.is_zero)])
+        numerators = []
+        denominators = []
+        for a in self.args:
+            if a.is_integer:
+                numerators.append(a)
+            elif a.is_Rational:
+                n, d = a.as_numer_denom()
+                numerators.append(n)
+                denominators.append(d)
+            elif a.is_Pow:
+                b, e = a.as_base_exp()
+                if e is S.NegativeOne and b.is_integer:
+                    denominators.append(b)
+                else:
+                    return
+            else:
+                return
+
+        if not denominators:
+            return True
+
+        odd = lambda ints: all(i.is_odd for i in ints)
+        even = lambda ints: any(i.is_even for i in ints)
+
+        if odd(numerators) and even(denominators):
+            return False
+        elif even(numerators) and denominators == [2]:
+            return True
 
     def _eval_is_polar(self):
         has_polar = any(arg.is_polar for arg in self.args)
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
index e05cdf6ac1..c52408b906 100644
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -391,10 +391,10 @@ def test_Mul_is_integer():
     assert (e/o).is_integer is None
     assert (o/e).is_integer is False
     assert (o/i2).is_integer is False
-    assert Mul(o, 1/o, evaluate=False).is_integer is True
+    #assert Mul(o, 1/o, evaluate=False).is_integer is True
     assert Mul(k, 1/k, evaluate=False).is_integer is None
-    assert Mul(nze, 1/nze, evaluate=False).is_integer is True
-    assert Mul(2., S.Half, evaluate=False).is_integer is False
+    #assert Mul(nze, 1/nze, evaluate=False).is_integer is True
+    #assert Mul(2., S.Half, evaluate=False).is_integer is False
 
     s = 2**2**2**Pow(2, 1000, evaluate=False)
     m = Mul(s, s, evaluate=False)
```
I only tested with core tests. It's possible that something elsewhere would break with this change...
> Here's a better fix:
> ```python
> [...]
> def _eval_is_integer(self):
> [...]
> ```

This looks like the right place to not only decide if an expression evaluates to an integer, but also check if the decision is feasible (i.e. possible without full evaluation).

> ```diff
> +    #assert Mul(o, 1/o, evaluate=False).is_integer is True
> ```

Yes, I think such tests/intentions should be dropped: if an expression was constructed with `evaluate=False`, the integer decision can be given up, if it cannot be decided without evaluation.

Without the even/odd assumptions the same implementation as for `Add` would suffice here?
```python
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
```

The handling of `is_Pow` seems too restrictive to me. What about this:
> ```diff
> +        for a in self.args:
> [...]
> +            elif a.is_Pow:
> +                b, e = a.as_base_exp()
> +                if b.is_integer and e.is_integer:
> +                    if e > 0:
> +                        numerators.append(b) # optimization for numerators += e * [b]
> +                    elif e < 0:
> +                        denominators.append(b) # optimization for denominators += (-e) * [b]
> +                else:
> +                    return
> [...]
> ```

I think we probably could just get rid of the even/odd checking. I can't imagine that it helps in many situations. I just added it there because I was looking for a quick minimal change.

> What about this:

You have to be careful with `e > 0` (see the explanation in #20090) because it raises in the indeterminate case:
```julia
In [1]: x, y = symbols('x, y', integer=True)                                                                                                                                      

In [2]: expr = x**y                                                                                                                                                               

In [3]: expr                                                                                                                                                                      
Out[3]: 
 y
x 

In [4]: b, e = expr.as_base_exp()                                                                                                                                                 

In [5]: if e > 0: 
   ...:     print('positive') 
   ...:                                                                                                                                                                           
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-5-c0d1c873f05a> in <module>
----> 1 if e > 0:
      2     print('positive')
      3 

~/current/sympy/sympy/sympy/core/relational.py in __bool__(self)
    393 
    394     def __bool__(self):
--> 395         raise TypeError("cannot determine truth value of Relational")
    396 
    397     def _eval_as_set(self):

TypeError: cannot determine truth value of Relational
```
> I think we probably could just get rid of the even/odd checking.

Then we would loose this feature:
```python
>>> k = Symbol('k', even=True)
>>> (k/2).is_integer
True
```

> You have to be careful with e > 0 (see the explanation in #20090) [...}

Ok. But `e.is_positive` should fix that?
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/functions/elementary/integers.py b/sympy/functions/elementary/integers.py
index fe1e71e..7a3822e 100755
--- a/sympy/functions/elementary/integers.py
+++ b/sympy/functions/elementary/integers.py
@@ -43,7 +43,7 @@ def eval(cls, arg):
         terms = Add.make_args(arg)
 
         for t in terms:
-            if t.is_integer or (t.is_imaginary and im(t).is_integer):
+            if t.is_integer and (t.is_Integer or not t.is_number):
                 ipart += t
             elif t.has(Symbol):
                 spart += t
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
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -7,7 +7,7 @@
 from .singleton import S
 from .operations import AssocOp, AssocOpDispatcher
 from .cache import cacheit
-from .logic import fuzzy_not, _fuzzy_group, fuzzy_and
+from .logic import fuzzy_not, _fuzzy_group
 from .compatibility import reduce
 from .expr import Expr
 from .parameters import global_parameters
@@ -1262,27 +1262,47 @@ def _eval_is_zero(self):
                     zero = None
         return zero
 
+    # without involving odd/even checks this code would suffice:
+    #_eval_is_integer = lambda self: _fuzzy_group(
+    #    (a.is_integer for a in self.args), quick_exit=True)
     def _eval_is_integer(self):
-        from sympy import fraction
-        from sympy.core.numbers import Float
-
         is_rational = self._eval_is_rational()
         if is_rational is False:
             return False
 
-        # use exact=True to avoid recomputing num or den
-        n, d = fraction(self, exact=True)
-        if is_rational:
-            if d is S.One:
-                return True
-        if d.is_even:
-            if d.is_prime:  # literal or symbolic 2
-                return n.is_even
-            if n.is_odd:
-                return False  # true even if d = 0
-        if n == d:
-            return fuzzy_and([not bool(self.atoms(Float)),
-            fuzzy_not(d.is_zero)])
+        numerators = []
+        denominators = []
+        for a in self.args:
+            if a.is_integer:
+                numerators.append(a)
+            elif a.is_Rational:
+                n, d = a.as_numer_denom()
+                numerators.append(n)
+                denominators.append(d)
+            elif a.is_Pow:
+                b, e = a.as_base_exp()
+                if not b.is_integer or not e.is_integer: return
+                if e.is_negative:
+                    denominators.append(b)
+                else:
+                    # for integer b and positive integer e: a = b**e would be integer
+                    assert not e.is_positive
+                    # for self being rational and e equal to zero: a = b**e would be 1
+                    assert not e.is_zero
+                    return # sign of e unknown -> self.is_integer cannot be decided
+            else:
+                return
+
+        if not denominators:
+            return True
+
+        odd = lambda ints: all(i.is_odd for i in ints)
+        even = lambda ints: any(i.is_even for i in ints)
+
+        if odd(numerators) and even(denominators):
+            return False
+        elif even(numerators) and denominators == [2]:
+            return True
 
     def _eval_is_polar(self):
         has_polar = any(arg.is_polar for arg in self.args)
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -374,12 +374,10 @@ def test_Mul_doesnt_expand_exp():
     assert (x**(-log(5)/log(3))*x)/(x*x**( - log(5)/log(3))) == sympify(1)
 
 def test_Mul_is_integer():
-
     k = Symbol('k', integer=True)
     n = Symbol('n', integer=True)
     nr = Symbol('nr', rational=False)
     nz = Symbol('nz', integer=True, zero=False)
-    nze = Symbol('nze', even=True, zero=False)
     e = Symbol('e', even=True)
     o = Symbol('o', odd=True)
     i2 = Symbol('2', prime=True, even=True)
@@ -388,18 +386,31 @@ def test_Mul_is_integer():
     assert (nz/3).is_integer is None
     assert (nr/3).is_integer is False
     assert (x*k*n).is_integer is None
+    assert (e/2).is_integer is True
+    assert (e**2/2).is_integer is True
+    assert (2/k).is_integer is None
+    assert (2/k**2).is_integer is None
+    assert ((-1)**k*n).is_integer is True
+    assert (3*k*e/2).is_integer is True
+    assert (2*k*e/3).is_integer is None
     assert (e/o).is_integer is None
     assert (o/e).is_integer is False
     assert (o/i2).is_integer is False
-    assert Mul(o, 1/o, evaluate=False).is_integer is True
     assert Mul(k, 1/k, evaluate=False).is_integer is None
-    assert Mul(nze, 1/nze, evaluate=False).is_integer is True
-    assert Mul(2., S.Half, evaluate=False).is_integer is False
+    assert Mul(2., S.Half, evaluate=False).is_integer is None
+    assert (2*sqrt(k)).is_integer is None
+    assert (2*k**n).is_integer is None
 
     s = 2**2**2**Pow(2, 1000, evaluate=False)
     m = Mul(s, s, evaluate=False)
     assert m.is_integer
 
+    # broken in 1.6 and before, see #20161
+    xq = Symbol('xq', rational=True)
+    yq = Symbol('yq', rational=True)
+    assert (xq*yq).is_integer is None
+    e_20161 = Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False)
+    assert e_20161.is_integer is not True # expand(e_20161) -> -1/2, but no need to see that in the assumption without evaluation
 
 def test_Add_Mul_is_integer():
     x = Symbol('x')
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 213
Step: 39

COMMAND
python - <<'EOF'
import re
p = 'sympy/functions/elementary/integers.py'
s = open(p).read()
old = """        for t in terms:
            if t.is_integer or (t.is_imaginary and im(t).is_integer):
                ipart += t
            elif t.has(Symbol):
                spart += t
            else:
                npart += t
"""
new = """        for t in terms:
            if t.is_integer and (t.is_Integer or not t.is_number):
                ipart += t
            elif t.has(Symbol):
                spart += t
            else:
                npart += t
"""
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, new))
EOF
python - <<'EOF'
from sympy import sympify, simplify, Symbol, ceiling
from sympy.parsing.sympy_parser import parse_expr
x = Symbol('x')
for ev in (False, True):
    e = sympify('4*ceiling(x/4 - 3/4)', evaluate=ev)
    print(ev, repr(e), '->', repr(e.simplify()))
print('floor evalF', repr(sympify('floor(x/4 - 3/4)', evaluate=False)))
print('ceiling direct', repr(ceiling(x/4 - 3/4)))
assert sympify('4*ceiling(x/4 - 3/4)', evaluate=False).simplify() == \
       sympify('4*ceiling(x/4 - 3/4)', evaluate=True).simplify()
EOF
python -m pytest sympy/functions/elementary/tests/test_integers.py -q 2>&1 | tail -5

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.881
stdout:
False 4*ceiling(x/4 + (-1)*3*1/4) -> 4*ceiling(x/4 - 3/4)
True 4*ceiling(x/4 - 3/4) -> 4*ceiling(x/4 - 3/4)
floor evalF floor(x/4 + (-1)*3*1/4)
ceiling direct ceiling(x/4 - 0.75)
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
