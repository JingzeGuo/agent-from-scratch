# sympy__sympy-16503: Moving the outer baseline does not fully correct summand centering

**Primary category:** R — Symptom fix; underlying cause remains. **Confidence:** medium.

**Recorded score:** F2P 0/1; P2P 121/121. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

A one-line adjustment aligns the simple example but misses internal placement and mode-dependent geometry.

## Issue requirement

Align x and +3 in Sum(x, ...)+3; the issue allows moving either component.

## Agent change

Subtracts adjustment from the final prettyF.baseline only.

## Golden change

Changes the summand baseline calculation and returns the ASCII height adjustment, then sets the overall baseline with mode-dependent correction.

## Observed failure

test_pretty_sum fails a Unicode snapshot for a larger summand; all 121 P2P tests pass. The agent had already seen the native test_pretty_sum fail during its own run.

## Scope and attribution

There is some layout latitude in the issue, but the outer-baseline patch leaves the accepted internal-centering problem untouched. Treat this as incomplete baseline reasoning, with snapshot-policy ambiguity secondary.

## Verification in the trajectory

Steps 37 and 39 fail direct verification; the last attempt fails on an import before rewriting/rerunning snapshots. The final patch is not backed by a passing full sum test.

## What would improve the outcome

Model the summand baseline and the composed expression baseline separately. Verify ASCII/Unicode, odd/even heights, nested sums and tall expressions before updating snapshots.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sympy__sympy-16503`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 89 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 89 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sympy__sympy-16503.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
Bad centering for Sum pretty print
```
>>> pprint(Sum(x, (x, 1, oo)) + 3)
  ∞
 ___
 ╲
  ╲   x
  ╱     + 3
 ╱
 ‾‾‾
x = 1
```

The `x` and the `+ 3` should be aligned. I'm not sure if the `x` should be lower of if the `+ 3` should be higher.
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
```
>>> pprint(Sum(x**2, (x, 1, oo)) + 3)
 ∞         
 ___        
 ╲          
  ╲    2    
  ╱   x  + 3
 ╱          
 ‾‾‾        
x = 1
```
This works well. So, I suppose that `x`, in the above case should be lower.
Could you tell me, how I can correct it?
The issue might be with the way adjustments are calculated, and this definitely works for simpler expressions:
```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
index 7a3de3352..07198bea4 100644
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -575,7 +575,7 @@ def adjust(s, wid=None, how='<^>'):
                 for i in reversed(range(0, d)):
                     lines.append('%s%s%s' % (' '*i, vsum[4], ' '*(w - i - 1)))
                 lines.append(vsum[8]*(w))
-                return d, h + 2*more, lines, more
+                return d, h + 2*more, lines, more // 2
 
         f = expr.function
```
as in
```python
>>> pprint(Sum(x ** n, (n, 1, oo)) + x)
      ∞     
     ___    
     ╲      
      ╲    n
x +   ╱   x 
     ╱      
     ‾‾‾    
    n = 1   

>>> pprint(Sum(n, (n, 1, oo)) + x)
      ∞    
     ___   
     ╲     
      ╲    
x +   ╱   n
     ╱     
     ‾‾‾   
    n = 1   
```

but this leads to test failures for more complex expressions. However, many of the tests look like they expect the misaligned sum.
The ascii printer also has this issue:
```
In [1]: pprint(x + Sum(x + Integral(x**2 + x + 1, (x, 0, n)), (n, 1, oo)), use_unicode=False)
       oo                            
    ______                           
    \     `                          
     \      /      n                \
      \     |      /                |
       \    |     |                 |
x +     \   |     |  / 2        \   |
        /   |x +  |  \x  + x + 1/ dx|
       /    |     |                 |
      /     |    /                  |
     /      \    0                  /
    /_____,                          
     n = 1                           

```
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
index 7a3de33..16fa895 100755
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -629,7 +629,7 @@ def adjust(s, wid=None, how='<^>'):
             # put the present prettyF to the right
             prettyF = prettyForm(*prettySign.right(prettyF))
 
-        prettyF.baseline = max_upper + sign_height//2
+        prettyF.baseline = max_upper + sign_height//2 - adjustment
         prettyF.binding = prettyForm.MUL
         return prettyF
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
index 4c48fda..b875ef6 100755
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -4862,6 +4862,30 @@ def test_pretty_sum():
     assert pretty(expr) == ascii_str
     assert upretty(expr) == ucode_str
 
+    expr = Sum(x, (x, 1, oo)) + 3
+    ascii_str = \
+"""\
+  oo       \n\
+ __        \n\
+ \\ `       \n\
+  )   x + 3\n\
+ /_,       \n\
+x = 1      \
+"""
+    ucode_str = \
+u("""\
+  ∞        \n\
+ ___       \n\
+ ╲         \n\
+  ╲   x + 3\n\
+  ╱        \n\
+ ╱         \n\
+ ‾‾‾       \n\
+x = 1      \
+""")
+    assert pretty(expr) == ascii_str
+    assert upretty(expr) == ucode_str
+
 
 def test_units():
     expr = joule
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -564,7 +564,7 @@ def adjust(s, wid=None, how='<^>'):
                 for i in reversed(range(1, d)):
                     lines.append('%s/%s' % (' '*i, ' '*(w - i)))
                 lines.append("/" + "_"*(w - 1) + ',')
-                return d, h + more, lines, 0
+                return d, h + more, lines, more
             else:
                 w = w + more
                 d = d + more
@@ -619,7 +619,7 @@ def adjust(s, wid=None, how='<^>'):
             if first:
                 # change F baseline so it centers on the sign
                 prettyF.baseline -= d - (prettyF.height()//2 -
-                                         prettyF.baseline) - adjustment
+                                         prettyF.baseline)
                 first = False
 
             # put padding to the right
@@ -629,7 +629,11 @@ def adjust(s, wid=None, how='<^>'):
             # put the present prettyF to the right
             prettyF = prettyForm(*prettySign.right(prettyF))
 
-        prettyF.baseline = max_upper + sign_height//2
+        # adjust baseline of ascii mode sigma with an odd height so that it is
+        # exactly through the center
+        ascii_adjustment = ascii_mode if not adjustment else 0
+        prettyF.baseline = max_upper + sign_height//2 + ascii_adjustment
+
         prettyF.binding = prettyForm.MUL
         return prettyF
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -4423,14 +4423,14 @@ def test_pretty_sum():
   n             \n\
 ______          \n\
 ╲               \n\
- ╲      ∞       \n\
-  ╲     ⌠       \n\
-   ╲    ⎮   n   \n\
-    ╲   ⎮  x  dx\n\
-    ╱   ⌡       \n\
-   ╱    -∞      \n\
-  ╱    k        \n\
- ╱              \n\
+ ╲              \n\
+  ╲     ∞       \n\
+   ╲    ⌠       \n\
+    ╲   ⎮   n   \n\
+    ╱   ⎮  x  dx\n\
+   ╱    ⌡       \n\
+  ╱     -∞      \n\
+ ╱     k        \n\
 ╱               \n\
 ‾‾‾‾‾‾          \n\
 k = 0           \
@@ -4474,14 +4474,14 @@ def test_pretty_sum():
 -∞                \n\
  ______           \n\
  ╲                \n\
-  ╲       ∞       \n\
-   ╲      ⌠       \n\
-    ╲     ⎮   n   \n\
-     ╲    ⎮  x  dx\n\
-     ╱    ⌡       \n\
-    ╱     -∞      \n\
-   ╱     k        \n\
-  ╱               \n\
+  ╲               \n\
+   ╲      ∞       \n\
+    ╲     ⌠       \n\
+     ╲    ⎮   n   \n\
+     ╱    ⎮  x  dx\n\
+    ╱     ⌡       \n\
+   ╱      -∞      \n\
+  ╱      k        \n\
  ╱                \n\
  ‾‾‾‾‾‾           \n\
  k = 0            \
@@ -4527,14 +4527,14 @@ def test_pretty_sum():
           -∞                         \n\
            ______                    \n\
            ╲                         \n\
-            ╲                ∞       \n\
-             ╲               ⌠       \n\
-              ╲              ⎮   n   \n\
-               ╲             ⎮  x  dx\n\
-               ╱             ⌡       \n\
-              ╱              -∞      \n\
-             ╱              k        \n\
-            ╱                        \n\
+            ╲                        \n\
+             ╲               ∞       \n\
+              ╲              ⌠       \n\
+               ╲             ⎮   n   \n\
+               ╱             ⎮  x  dx\n\
+              ╱              ⌡       \n\
+             ╱               -∞      \n\
+            ╱               k        \n\
            ╱                         \n\
            ‾‾‾‾‾‾                    \n\
      2        2       1   x          \n\
@@ -4572,14 +4572,14 @@ def test_pretty_sum():
                   x   n          \n\
          ______                  \n\
          ╲                       \n\
-          ╲              ∞       \n\
-           ╲             ⌠       \n\
-            ╲            ⎮   n   \n\
-             ╲           ⎮  x  dx\n\
-             ╱           ⌡       \n\
-            ╱            -∞      \n\
-           ╱            k        \n\
-          ╱                      \n\
+          ╲                      \n\
+           ╲             ∞       \n\
+            ╲            ⌠       \n\
+             ╲           ⎮   n   \n\
+             ╱           ⎮  x  dx\n\
+            ╱            ⌡       \n\
+           ╱             -∞      \n\
+          ╱             k        \n\
          ╱                       \n\
          ‾‾‾‾‾‾                  \n\
          k = 0                   \
@@ -4602,8 +4602,8 @@ def test_pretty_sum():
   ∞    \n\
  ___   \n\
  ╲     \n\
-  ╲   x\n\
-  ╱    \n\
+  ╲    \n\
+  ╱   x\n\
  ╱     \n\
  ‾‾‾   \n\
 x = 0  \
@@ -4655,10 +4655,10 @@ def test_pretty_sum():
   ∞    \n\
  ____  \n\
  ╲     \n\
-  ╲   x\n\
-   ╲  ─\n\
-   ╱  2\n\
-  ╱    \n\
+  ╲    \n\
+   ╲  x\n\
+   ╱  ─\n\
+  ╱   2\n\
  ╱     \n\
  ‾‾‾‾  \n\
 x = 0  \
@@ -4716,12 +4716,12 @@ def test_pretty_sum():
   ∞           \n\
 _____         \n\
 ╲             \n\
- ╲           n\n\
-  ╲   ⎛    x⎞ \n\
-   ╲  ⎜    ─⎟ \n\
-   ╱  ⎜ 3  2⎟ \n\
-  ╱   ⎝x ⋅y ⎠ \n\
- ╱            \n\
+ ╲            \n\
+  ╲          n\n\
+   ╲  ⎛    x⎞ \n\
+   ╱  ⎜    ─⎟ \n\
+  ╱   ⎜ 3  2⎟ \n\
+ ╱    ⎝x ⋅y ⎠ \n\
 ╱             \n\
 ‾‾‾‾‾         \n\
 x = 0         \
@@ -4844,14 +4844,14 @@ def test_pretty_sum():
     ∞          n                         \n\
   ______   ______                        \n\
   ╲        ╲                             \n\
-   ╲        ╲     ⎛        1    ⎞        \n\
-    ╲        ╲    ⎜1 + ─────────⎟        \n\
-     ╲        ╲   ⎜          1  ⎟        \n\
-      ╲        ╲  ⎜    1 + ─────⎟     1  \n\
-      ╱        ╱  ⎜            1⎟ + ─────\n\
-     ╱        ╱   ⎜        1 + ─⎟       1\n\
-    ╱        ╱    ⎝            k⎠   1 + ─\n\
-   ╱        ╱                           k\n\
+   ╲        ╲                            \n\
+    ╲        ╲    ⎛        1    ⎞        \n\
+     ╲        ╲   ⎜1 + ─────────⎟        \n\
+      ╲        ╲  ⎜          1  ⎟     1  \n\
+      ╱        ╱  ⎜    1 + ─────⎟ + ─────\n\
+     ╱        ╱   ⎜            1⎟       1\n\
+    ╱        ╱    ⎜        1 + ─⎟   1 + ─\n\
+   ╱        ╱     ⎝            k⎠       k\n\
   ╱        ╱                             \n\
   ‾‾‾‾‾‾   ‾‾‾‾‾‾                        \n\
       1   k = 111                        \n\
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 150
Step: 30

COMMAND
python -m pytest sympy/printing/pretty/tests/test_pretty.py::test_pretty_sum -q 2>&1 | tail -50

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.344
stdout:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 37 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 185
Step: 37

COMMAND
python - <<'EOF'
from sympy.printing.pretty.tests.test_pretty import test_pretty_sum
test_pretty_sum()
print('test_pretty_sum passed')
EOF

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.787
stdout:
[empty]
stderr:
/testbed/sympy/physics/vector/printing.py:153: SyntaxWarning: "is not" with a literal. Did you mean "!="?
  if len(base_split) is not 1:
/testbed/sympy/vector/coordsysrect.py:170: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if transformation.name is 'spherical':
/testbed/sympy/vector/coordsysrect.py:172: SyntaxWarning: "is" with a literal. Did you mean "=="?
  elif transformation.name is 'cylindrical':
Tra... [truncated]
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 195
Step: 39

COMMAND
python - <<'EOF'
import importlib
from sympy import symbols, Sum, oo
import sympy.printing.pretty.tests.test_pretty as tp

path = 'sympy/printing/pretty/tests/test_pretty.py'
src = open(path).read()
start = src.index('    expr = Sum(1/(1 + 1/(\n        1 + 1/k))')
end = src.index('\n\ndef test_units():', start)
expr_end = src.index('\n    ascii_str = \\', start)
exprsrc = src[start:expr_end]

k, n, m = symbols('k n m')
expr = Sum(1/(1 + 1/(1 + 1/k)) + 1, (k, 111, 1 + 1/n),
    (k, 1/(1 + m), oo)) + 1/(1 + 1/k)
a = tp.pretty(expr)
u = tp.upretty(expr)
block = exprsrc + '\n'
block += '    ascii_str = \\\n"""\\\n' + '\\\n'.join(a.split('\n')) + '\\\n"""\n'
block += '    ucode_str = \\\nu("""\\\n' + '\\\n'.join(u.split('\n')) + '\\\n""")\n'
block += '\n    assert pretty(expr) == ascii_str\n    assert upretty(expr) == ucode_str'
open(path, 'w').write(src[:start] + block + src[end:])
importlib.reload(tp)
tp.test_pretty_sum()
print('test_pretty_sum passed')
EOF

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.700
stdout:
[empty]
stderr:
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
ImportError: cannot import name 'tests' from 'pretty' (unknown location)
```

</details>


[Return to main report](../README.md)
