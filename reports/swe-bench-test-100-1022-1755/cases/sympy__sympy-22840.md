# sympy__sympy-22840: MatrixSymbol is atomic for CSE, but MatrixElement is still extracted

**Primary category:** E — Missing edge case or input/operation family. **Confidence:** medium.

**Recorded score:** F2P 1/2; P2P 95/95. **Steps:** 35/40. Counts are harness classifications, subject to the report's parser caveats.

The patch removes whole-matrix copies yet leaves scalar element temporaries that the accepted codegen contract excludes.

## Issue requirement

CSE of indexed MatrixSymbols generates redundant copies of whole matrices before code generation.

## Agent change

Adds MatrixSymbol to the atomic early-return cases in tree_cse._find_repeated.

## Golden change

Treats both MatrixSymbol and MatrixElement as atomic, while allowing larger repeated products to be extracted.

## Observed failure

The MatrixSymbol F2P test passes. test_multidim_c_argument_cse fails because the generated code still extracts repeated element temporaries; all 95 P2P tests pass.

## Scope and attribution

This is an adjacent-node/end-to-end coverage gap, with a genuine scope caveat: the issue specifically complains about whole-matrix copies, and exact scalar-temporary suppression is an additional accepted design choice.

## Verification in the trajectory

The agent inspects MatrixElement and demonstrates remaining element substitutions, but validates only the CSE test module. The recorded native CSE run passes 42 tests with five expected failures; codegen is not validated.

## What would improve the outcome

Check MatrixSymbol, MatrixElement and composite matrix expressions together, then inspect generated C. Explicitly decide which repeated objects merit temporaries.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 100 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 100 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 100 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-22840/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-22840/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-22840/run_instance.log) | — |
| trajectory | [sympy__sympy-22840.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/sympy__sympy-22840.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

````text
cse() has strange behaviour for MatrixSymbol indexing
Example: 
```python
import sympy as sp
from pprint import pprint


def sub_in_matrixsymbols(exp, matrices):
    for matrix in matrices:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                name = "%s_%d_%d" % (matrix.name, i, j)
                sym = sp.symbols(name)
                exp = exp.subs(sym, matrix[i, j])
    return exp


def t44(name):
    return sp.Matrix(4, 4, lambda i, j: sp.symbols('%s_%d_%d' % (name, i, j)))


# Construct matrices of symbols that work with our
# expressions. (MatrixSymbols does not.)
a = t44("a")
b = t44("b")

# Set up expression. This is a just a simple example.
e = a * b

# Put in matrixsymbols. (Gives array-input in codegen.)
e2 = sub_in_matrixsymbols(e, [sp.MatrixSymbol("a", 4, 4), sp.MatrixSymbol("b", 4, 4)])
cse_subs, cse_reduced = sp.cse(e2)
pprint((cse_subs, cse_reduced))

# Codegen, etc..
print "\nccode:"
for sym, expr in cse_subs:
    constants, not_c, c_expr = sympy.printing.ccode(
        expr,
        human=False,
        assign_to=sympy.printing.ccode(sym),
    )
    assert not constants, constants
    assert not not_c, not_c
    print "%s\n" % c_expr

```

This gives the following output:

```
([(x0, a),
  (x1, x0[0, 0]),
  (x2, b),
  (x3, x2[0, 0]),
  (x4, x0[0, 1]),
  (x5, x2[1, 0]),
  (x6, x0[0, 2]),
  (x7, x2[2, 0]),
  (x8, x0[0, 3]),
  (x9, x2[3, 0]),
  (x10, x2[0, 1]),
  (x11, x2[1, 1]),
  (x12, x2[2, 1]),
  (x13, x2[3, 1]),
  (x14, x2[0, 2]),
  (x15, x2[1, 2]),
  (x16, x2[2, 2]),
  (x17, x2[3, 2]),
  (x18, x2[0, 3]),
  (x19, x2[1, 3]),
  (x20, x2[2, 3]),
  (x21, x2[3, 3]),
  (x22, x0[1, 0]),
  (x23, x0[1, 1]),
  (x24, x0[1, 2]),
  (x25, x0[1, 3]),
  (x26, x0[2, 0]),
  (x27, x0[2, 1]),
  (x28, x0[2, 2]),
  (x29, x0[2, 3]),
  (x30, x0[3, 0]),
  (x31, x0[3, 1]),
  (x32, x0[3, 2]),
  (x33, x0[3, 3])],
 [Matrix([
[    x1*x3 + x4*x5 + x6*x7 + x8*x9,     x1*x10 + x11*x4 + x12*x6 + x13*x8,     x1*x14 + x15*x4 + x16*x6 + x17*x8,     x1*x18 + x19*x4 + x20*x6 + x21*x8],
[x22*x3 + x23*x5 + x24*x7 + x25*x9, x10*x22 + x11*x23 + x12*x24 + x13*x25, x14*x22 + x15*x23 + x16*x24 + x17*x25, x18*x22 + x19*x23 + x20*x24 + x21*x25],
[x26*x3 + x27*x5 + x28*x7 + x29*x9, x10*x26 + x11*x27 + x12*x28 + x13*x29, x14*x26 + x15*x27 + x16*x28 + x17*x29, x18*x26 + x19*x27 + x20*x28 + x21*x29],
[x3*x30 + x31*x5 + x32*x7 + x33*x9, x10*x30 + x11*x31 + x12*x32 + x13*x33, x14*x30 + x15*x31 + x16*x32 + x17*x33, x18*x30 + x19*x31 + x20*x32 + x21*x33]])])

ccode:
x0[0] = a[0];
x0[1] = a[1];
x0[2] = a[2];
x0[3] = a[3];
x0[4] = a[4];
x0[5] = a[5];
x0[6] = a[6];
x0[7] = a[7];
x0[8] = a[8];
x0[9] = a[9];
x0[10] = a[10];
x0[11] = a[11];
x0[12] = a[12];
x0[13] = a[13];
x0[14] = a[14];
x0[15] = a[15];
x1 = x0[0];
x2[0] = b[0];
x2[1] = b[1];
x2[2] = b[2];
x2[3] = b[3];
x2[4] = b[4];
x2[5] = b[5];
x2[6] = b[6];
x2[7] = b[7];
x2[8] = b[8];
x2[9] = b[9];
x2[10] = b[10];
x2[11] = b[11];
x2[12] = b[12];
x2[13] = b[13];
x2[14] = b[14];
x2[15] = b[15];
x3 = x2[0];
x4 = x0[1];
x5 = x2[4];
x6 = x0[2];
x7 = x2[8];
x8 = x0[3];
x9 = x2[12];
x10 = x2[1];
x11 = x2[5];
x12 = x2[9];
x13 = x2[13];
x14 = x2[2];
x15 = x2[6];
x16 = x2[10];
x17 = x2[14];
x18 = x2[3];
x19 = x2[7];
x20 = x2[11];
x21 = x2[15];
x22 = x0[4];
x23 = x0[5];
x24 = x0[6];
x25 = x0[7];
x26 = x0[8];
x27 = x0[9];
x28 = x0[10];
x29 = x0[11];
x30 = x0[12];
x31 = x0[13];
x32 = x0[14];
x33 = x0[15];
```

`x0` and `x2` are just copies of the matrices `a` and `b`, respectively.
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
Can you create a very simple example using MatrixSymbol and the expected output that you'd like to see?
I think one would expect the output to be similar to the following (except for the expression returned by CSE being a matrix where the individual elements are terms as defined by matrix multiplication, that is, unchanged by `cse()`).

```py
import sympy as sp
from pprint import pprint
import sympy.printing.ccode


def print_ccode(assign_to, expr):
    constants, not_c, c_expr = sympy.printing.ccode(
        expr,
        human=False,
        assign_to=assign_to,
    )
    assert not constants, constants
    assert not not_c, not_c
    print "%s" % c_expr


a = sp.MatrixSymbol("a", 4, 4)
b = sp.MatrixSymbol("b", 4, 4)

# Set up expression. This is a just a simple example.
e = a * b
print "\nexpr:"
print e

cse_subs, cse_reduced = sp.cse(e)
print "\ncse(expr):"
pprint((cse_subs, cse_reduced))

# Codegen.
print "\nccode:"
for sym, expr in cse_subs:
    print_ccode(sympy.printing.ccode(sym), expr)
assert len(cse_reduced) == 1
print_ccode(sympy.printing.ccode(sp.symbols("result")), cse_reduced[0])
```

Gives the output:

```
expr:
a*b

cse(expr):
([], [a*b])

ccode:
result[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8] + a[3]*b[12];
result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9] + a[3]*b[13];
result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14];
result[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15];
result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8] + a[7]*b[12];
result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9] + a[7]*b[13];
result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14];
result[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15];
result[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8] + a[11]*b[12];
result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9] + a[11]*b[13];
result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10] + a[11]*b[14];
result[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]*b[15];
result[12] = a[12]*b[0] + a[13]*b[4] + a[14]*b[8] + a[15]*b[12];
result[13] = a[12]*b[1] + a[13]*b[5] + a[14]*b[9] + a[15]*b[13];
result[14] = a[12]*b[2] + a[13]*b[6] + a[14]*b[10] + a[15]*b[14];
result[15] = a[12]*b[3] + a[13]*b[7] + a[14]*b[11] + a[15]*b[15];
```
Thanks. Note that it doesn't look like cse is well tested (i.e. designed) for MatrixSymbols based on the unit tests: https://github.com/sympy/sympy/blob/master/sympy/simplify/tests/test_cse.py#L315. Those tests don't really prove that it works as desired. So this definitely needs to be fixed.
The first part works as expected:

```
In [1]: import sympy as sm

In [2]: M = sm.MatrixSymbol('M', 3, 3)

In [3]: B = sm.MatrixSymbol('B', 3, 3)

In [4]: M * B
Out[4]: M*B

In [5]: sm.cse(M * B)
Out[5]: ([], [M*B])
```
For the ccode of an expression of MatrixSymbols, I would not expect it to print the results as you have them. MatrixSymbols should map to a matrix algebra library like BLAS and LINPACK. But Matrix, on the other hand, should do what you expect. Note how this works:

```
In [8]: M = sm.Matrix(3, 3, lambda i, j: sm.Symbol('M_{}{}'.format(i, j)))

In [9]: M
Out[9]: 
Matrix([
[M_00, M_01, M_02],
[M_10, M_11, M_12],
[M_20, M_21, M_22]])

In [10]: B = sm.Matrix(3, 3, lambda i, j: sm.Symbol('B_{}{}'.format(i, j)))

In [11]: B
Out[11]: 
Matrix([
[B_00, B_01, B_02],
[B_10, B_11, B_12],
[B_20, B_21, B_22]])

In [12]: M * B
Out[12]: 
Matrix([
[B_00*M_00 + B_10*M_01 + B_20*M_02, B_01*M_00 + B_11*M_01 + B_21*M_02, B_02*M_00 + B_12*M_01 + B_22*M_02],
[B_00*M_10 + B_10*M_11 + B_20*M_12, B_01*M_10 + B_11*M_11 + B_21*M_12, B_02*M_10 + B_12*M_11 + B_22*M_12],
[B_00*M_20 + B_10*M_21 + B_20*M_22, B_01*M_20 + B_11*M_21 + B_21*M_22, B_02*M_20 + B_12*M_21 + B_22*M_22]])

In [13]: sm.cse(M * B)
Out[13]: 
([], [Matrix([
  [B_00*M_00 + B_10*M_01 + B_20*M_02, B_01*M_00 + B_11*M_01 + B_21*M_02, B_02*M_00 + B_12*M_01 + B_22*M_02],
  [B_00*M_10 + B_10*M_11 + B_20*M_12, B_01*M_10 + B_11*M_11 + B_21*M_12, B_02*M_10 + B_12*M_11 + B_22*M_12],
  [B_00*M_20 + B_10*M_21 + B_20*M_22, B_01*M_20 + B_11*M_21 + B_21*M_22, B_02*M_20 + B_12*M_21 + B_22*M_22]])])

In [17]: print(sm.ccode(M * B, assign_to=sm.MatrixSymbol('E', 3, 3)))
E[0] = B_00*M_00 + B_10*M_01 + B_20*M_02;
E[1] = B_01*M_00 + B_11*M_01 + B_21*M_02;
E[2] = B_02*M_00 + B_12*M_01 + B_22*M_02;
E[3] = B_00*M_10 + B_10*M_11 + B_20*M_12;
E[4] = B_01*M_10 + B_11*M_11 + B_21*M_12;
E[5] = B_02*M_10 + B_12*M_11 + B_22*M_12;
E[6] = B_00*M_20 + B_10*M_21 + B_20*M_22;
E[7] = B_01*M_20 + B_11*M_21 + B_21*M_22;
E[8] = B_02*M_20 + B_12*M_21 + B_22*M_22;
```
But in order to get a single input argument from codegen it cannot be different symbols, and if you replace each symbol with a `MatrixSymbol[i, j]` then `cse()` starts doing the above non-optiimizations for some reason.
As far as I know, `codegen` does not work with Matrix or MatrixSymbol's in any meaningful way. There are related issues:

#11456
#4367
#10522

In general, there needs to be work done in the code generators to properly support matrices.

As a work around, I suggest using `ccode` and a custom template to get the result you want.
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
index d649dd0..4721294 100755
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -586,7 +586,12 @@ def _find_repeated(expr):
         if isinstance(expr, RootOf):
             return
 
-        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order):
+        if isinstance(expr, Basic) and (
+                expr.is_Atom or expr.is_Order or
+                isinstance(expr, MatrixSymbol)):
+            # MatrixSymbol is not an Atom but it should be treated as one:
+            # its arguments (the name and the dimensions) should not be
+            # searched for subexpressions.
             if expr.is_Symbol:
                 excluded_symbols.add(expr)
             return
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/simplify/tests/test_cse.py b/sympy/simplify/tests/test_cse.py
index eb9cc23..fcde00a 100755
--- a/sympy/simplify/tests/test_cse.py
+++ b/sympy/simplify/tests/test_cse.py
@@ -347,6 +347,14 @@ def test_cse_MatrixSymbol():
     B = MatrixSymbol("B", n, n)
     assert cse(B) == ([], [B])
 
+    # MatrixSymbols are atomic and should not be extracted as
+    # common subexpressions; only their elements should be.
+    A = MatrixSymbol("A", 2, 2)
+    assert cse(A[0, 0]*A[1, 1] + A[0, 0]*A[1, 1]) == (
+        [], [2*A[0, 0]*A[1, 1]])
+    assert cse(A[0, 0]*A[0, 1] + A[0, 0]*A[1, 1]) == (
+        [(x0, A[0, 0])], [x0*A[0, 1] + x0*A[1, 1]])
+
 def test_cse_MatrixExpr():
     A = MatrixSymbol('A', 3, 3)
     y = MatrixSymbol('y', 3, 1)
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -567,6 +567,7 @@ def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
         Substitutions containing any Symbol from ``ignore`` will be ignored.
     """
     from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd
+    from sympy.matrices.expressions.matexpr import MatrixElement
     from sympy.polys.rootoftools import RootOf
 
     if opt_subs is None:
@@ -586,7 +587,10 @@ def _find_repeated(expr):
         if isinstance(expr, RootOf):
             return
 
-        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order):
+        if isinstance(expr, Basic) and (
+                expr.is_Atom or
+                expr.is_Order or
+                isinstance(expr, (MatrixSymbol, MatrixElement))):
             if expr.is_Symbol:
                 excluded_symbols.add(expr)
             return
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/simplify/tests/test_cse.py b/sympy/simplify/tests/test_cse.py
--- a/sympy/simplify/tests/test_cse.py
+++ b/sympy/simplify/tests/test_cse.py
@@ -347,6 +347,10 @@ def test_cse_MatrixSymbol():
     B = MatrixSymbol("B", n, n)
     assert cse(B) == ([], [B])
 
+    assert cse(A[0] * A[0]) == ([], [A[0]*A[0]])
+
+    assert cse(A[0,0]*A[0,1] + A[0,0]*A[0,1]*A[0,2]) == ([(x0, A[0, 0]*A[0, 1])], [x0*A[0, 2] + x0])
+
 def test_cse_MatrixExpr():
     A = MatrixSymbol('A', 3, 3)
     y = MatrixSymbol('y', 3, 1)
diff --git a/sympy/utilities/tests/test_codegen.py b/sympy/utilities/tests/test_codegen.py
--- a/sympy/utilities/tests/test_codegen.py
+++ b/sympy/utilities/tests/test_codegen.py
@@ -531,26 +531,9 @@ def test_multidim_c_argument_cse():
         '#include "test.h"\n'
         "#include <math.h>\n"
         "void c(double *A, double *b, double *out) {\n"
-        "   double x0[9];\n"
-        "   x0[0] = A[0];\n"
-        "   x0[1] = A[1];\n"
-        "   x0[2] = A[2];\n"
-        "   x0[3] = A[3];\n"
-        "   x0[4] = A[4];\n"
-        "   x0[5] = A[5];\n"
-        "   x0[6] = A[6];\n"
-        "   x0[7] = A[7];\n"
-        "   x0[8] = A[8];\n"
-        "   double x1[3];\n"
-        "   x1[0] = b[0];\n"
-        "   x1[1] = b[1];\n"
-        "   x1[2] = b[2];\n"
-        "   const double x2 = x1[0];\n"
-        "   const double x3 = x1[1];\n"
-        "   const double x4 = x1[2];\n"
-        "   out[0] = x2*x0[0] + x3*x0[1] + x4*x0[2];\n"
-        "   out[1] = x2*x0[3] + x3*x0[4] + x4*x0[5];\n"
-        "   out[2] = x2*x0[6] + x3*x0[7] + x4*x0[8];\n"
+        "   out[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];\n"
+        "   out[1] = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];\n"
+        "   out[2] = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];\n"
         "}\n"
     )
     assert code == expected
```

</details>

<details>
<summary>Verification command at step 20 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 124
Step: 20

COMMAND
python -m pytest sympy/simplify/tests/test_cse.py -x -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.389
stdout:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 21 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 129
Step: 21

COMMAND
python -c "import sympy; sympy.test('sympy/simplify/tests/test_cse.py', verbose=False)" 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.506
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
numpy:              None
random seed:        87558041
hash randomization: on (PYTHONHASHSEED=239520187)

sympy/simplify/tests/test_cse.py[47] .........f.fff....................f........
....... [truncated]
```

</details>

<details>
<summary>Verification command at step 28 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 170
Step: 28

COMMAND
python -c "import sympy; sympy.test('sympy/simplify/tests/test_cse.py', verbose=False)" 2>&1 | tail -12

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.421
stdout:
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
numpy:              None
random seed:        60739365
hash randomization: on (PYTHONHASHSEED=1850369649)

sympy/simplify/tests/test_cse.py[47] .........f.fff....................f........
....                                                                        [OK]

==... [truncated]
```

</details>

<details>
<summary>Verification command at step 31 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 189
Step: 31

COMMAND
python -c "import sympy; sympy.test('sympy/simplify/tests/test_cse.py', verbose=False)" 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.406
stdout:
numpy:              None
random seed:        31495708
hash randomization: on (PYTHONHASHSEED=3346195683)

sympy/simplify/tests/test_cse.py[47] .........f.fff....................f........
....                                                                        [OK]

======== tests finished: 42 passed, 5 expected to fail, in 0.47 seconds ========
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
