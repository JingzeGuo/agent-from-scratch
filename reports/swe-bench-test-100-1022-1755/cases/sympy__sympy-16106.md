# sympy__sympy-16106: Multiple indices lose grouping in Presentation MathML

**Primary category:** E — Missing edge case or input/operation family. **Confidence:** medium.

**Recorded score:** F2P 0/1; P2P 55/55. **Steps:** 38/40. Counts are harness classifications, subject to the report's parser caveats.

Single-index printing works, but the multi-index representation differs and loses explicit fencing.

## Issue requirement

Print Indexed objects to MathML without TypeError, including larger expressions containing them.

## Agent change

Adds content and presentation handlers, but renders multiple indices as adjacent children of mrow.

## Golden change

Adds tuple rendering with mfenced and uses it for multiple indices in msub.

## Observed failure

test_print_IndexedBase fails at the b,c,d index case, expecting mfenced rather than mrow. All 55 scored P2P tests pass; unrelated raw MathML failures are outside that scored set.

## Scope and attribution

The exact XML element is not stated in the issue, so acceptance is somewhat implementation-specific. However, adjacent unseparated indices omit grouping/separation, making this more than a purely cosmetic spelling difference.

## Verification in the trajectory

The agent tests its own content/presentation strings, sees broader preexisting-environment-style failures, then gets the focused indexed tests to pass. The expected strings encode the same mrow choice as its implementation.

## What would improve the outcome

Test one, multiple and tuple-valued indices with a semantic formatting oracle, and preserve separators/grouping. Compare XML structure rather than incidental attribute order where possible.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 87 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 87 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 87 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-16106/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-16106/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/sympy__sympy-16106/run_instance.log) | — |
| trajectory | [sympy__sympy-16106.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/sympy__sympy-16106.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

````text
mathml printer for IndexedBase required
Writing an `Indexed` object to MathML fails with a `TypeError` exception: `TypeError: 'Indexed' object is not iterable`:

```
In [340]: sympy.__version__
Out[340]: '1.0.1.dev'

In [341]: from sympy.abc import (a, b)

In [342]: sympy.printing.mathml(sympy.IndexedBase(a)[b])
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-342-b32e493b70d3> in <module>()
----> 1 sympy.printing.mathml(sympy.IndexedBase(a)[b])

/dev/shm/gerrit/venv/stable-3.5/lib/python3.5/site-packages/sympy/printing/mathml.py in mathml(expr, **settings)
    442 def mathml(expr, **settings):
    443     """Returns the MathML representation of expr"""
--> 444     return MathMLPrinter(settings).doprint(expr)
    445 
    446 

/dev/shm/gerrit/venv/stable-3.5/lib/python3.5/site-packages/sympy/printing/mathml.py in doprint(self, expr)
     36         Prints the expression as MathML.
     37         """
---> 38         mathML = Printer._print(self, expr)
     39         unistr = mathML.toxml()
     40         xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')

/dev/shm/gerrit/venv/stable-3.5/lib/python3.5/site-packages/sympy/printing/printer.py in _print(self, expr, *args, **kwargs)
    255                 printmethod = '_print_' + cls.__name__
    256                 if hasattr(self, printmethod):
--> 257                     return getattr(self, printmethod)(expr, *args, **kwargs)
    258             # Unknown object, fall back to the emptyPrinter.
    259             return self.emptyPrinter(expr)

/dev/shm/gerrit/venv/stable-3.5/lib/python3.5/site-packages/sympy/printing/mathml.py in _print_Basic(self, e)
    356     def _print_Basic(self, e):
    357         x = self.dom.createElement(self.mathml_tag(e))
--> 358         for arg in e:
    359             x.appendChild(self._print(arg))
    360         return x

TypeError: 'Indexed' object is not iterable
```

It also fails for more complex expressions where at least one element is Indexed.
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
Now it returns
```
'<indexed><indexedbase><ci>a</ci></indexedbase><ci>b</ci></indexed>'
```
for content printer and 
```
'<mrow><mi>indexed</mi><mfenced><mrow><mi>indexedbase</mi><mfenced><mi>a</mi></mfenced></mrow><mi>b</mi></mfenced></mrow>'
```
for presentation printer.

Probably not correct as it seems like it falls back to the printer for `Basic`.

Hence, a method `_print_IndexedBase` is required. Could be good to look at the LaTeX version to see how subscripts etc are handled.
Hi, can I take up this issue if it still needs fixing?
@pragyanmehrotra It is still needed so please go ahead!
@oscargus Sure I'll start working on it right ahead! However, Idk what exactly needs to be done so if you could point out how the output should look like and do I have to implement a new function or edit a current function it'd be a great help, Thanks.
```
from sympy import IndexedBase
a, b = symbols('a b')
IndexedBase(a)[b]
```
which renders as
![image](https://user-images.githubusercontent.com/8114497/53299790-abec5c80-383f-11e9-82c4-6dd3424f37a7.png)

Meaning that the presentation MathML output should be something like
`<msub><mi>a<mi><mi>b<mi></msub>`

Have a look at #16036 for some good resources.

Basically you need to do something like:
```
m = self.dom.createElement('msub')
m.appendChild(self._print(Whatever holds a))
m.appendChild(self._print(Whatever holds b))
```
in a function called `_print_IndexedBase`.
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sympy/printing/mathml.py b/sympy/printing/mathml.py
index 1aa0ee0..a765884 100755
--- a/sympy/printing/mathml.py
+++ b/sympy/printing/mathml.py
@@ -391,6 +391,20 @@ def translate(s):
     _print_MatrixSymbol = _print_Symbol
     _print_RandomSymbol = _print_Symbol
 
+    def _print_Indexed(self, e):
+        x = self.dom.createElement('apply')
+        x.appendChild(self.dom.createElement('selector'))
+        x.appendChild(self._print(e.base))
+        for index in e.indices:
+            x.appendChild(self._print(index))
+        return x
+
+    def _print_IndexedBase(self, e):
+        return self._print(e.label)
+
+    def _print_Idx(self, e):
+        return self._print(e.label)
+
     def _print_Pow(self, e):
         # Here we use root instead of power if the exponent is the reciprocal of an integer
         if self._settings['root_notation'] and e.exp.is_Rational and e.exp.p == 1:
@@ -848,6 +862,24 @@ def _print_MatrixSymbol(self, sym):
 
     _print_RandomSymbol = _print_Symbol
 
+    def _print_Indexed(self, e):
+        x = self.dom.createElement('msub')
+        x.appendChild(self._print(e.base))
+        if len(e.indices) == 1:
+            x.appendChild(self._print(e.indices[0]))
+        else:
+            mrow = self.dom.createElement('mrow')
+            for index in e.indices:
+                mrow.appendChild(self._print(index))
+            x.appendChild(mrow)
+        return x
+
+    def _print_IndexedBase(self, e):
+        return self._print(e.label)
+
+    def _print_Idx(self, e):
+        return self._print(e.label)
+
     def _print_conjugate(self, expr):
         enc = self.dom.createElement('menclose')
         enc.setAttribute('notation', 'top')
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sympy/printing/tests/test_mathml.py b/sympy/printing/tests/test_mathml.py
index 74e09fb..dd29c00 100755
--- a/sympy/printing/tests/test_mathml.py
+++ b/sympy/printing/tests/test_mathml.py
@@ -1,7 +1,7 @@
 from sympy import diff, Integral, Limit, sin, Symbol, Integer, Rational, cos, \
     tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, E, I, oo, \
     pi, GoldenRatio, EulerGamma, Sum, Eq, Ne, Ge, Lt, Float, Matrix, Basic, S, \
-    MatrixSymbol, Function, Derivative, log, Lambda
+    MatrixSymbol, Function, Derivative, log, Lambda, IndexedBase
 from sympy.core.containers import Tuple
 from sympy.functions.elementary.complexes import re, im, Abs, conjugate
 from sympy.functions.elementary.integers import floor, ceiling
@@ -70,6 +70,25 @@ def test_content_mathml_core():
     assert nodes[2].childNodes[0].nodeValue == 'x'
 
 
+def test_content_mathml_indexed():
+    A = IndexedBase('A')
+    i = Symbol('i', integer=True)
+
+    mml = mp._print(A)
+    assert mml.nodeName == 'ci'
+    assert mml.childNodes[0].nodeValue == 'A'
+
+    mml = mp._print(A[i])
+    assert mml.nodeName == 'apply'
+    assert mml.childNodes[0].nodeName == 'selector'
+    assert mml.childNodes[1].childNodes[0].nodeValue == 'A'
+    assert mml.childNodes[2].childNodes[0].nodeValue == 'i'
+
+    mml = mp._print(A[i] + 1)
+    assert mml.childNodes[0].nodeName == 'plus'
+    assert mml.childNodes[1].childNodes[0].nodeName == 'selector'
+
+
 def test_content_mathml_functions():
     mml_1 = mp._print(sin(x))
     assert mml_1.nodeName == 'apply'
@@ -745,6 +764,21 @@ def test_presentation_mathml_relational():
     assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'
 
 
+def test_presentation_mathml_indexed():
+    A = IndexedBase('A')
+    i = Symbol('i', integer=True)
+
+    mml = mpp._print(A)
+    assert mml.nodeName == 'mi'
+    assert mml.childNodes[0].nodeValue == 'A'
+
+    mml = mpp._print(A[i])
+    assert mml.nodeName == 'msub'
+    assert mml.childNodes[0].nodeName == 'mi'
+    assert mml.childNodes[0].childNodes[0].nodeValue == 'A'
+    assert mml.childNodes[1].childNodes[0].nodeValue == 'i'
+
+
 def test_presentation_symbol():
     mml = mpp._print(Symbol("x"))
     assert mml.nodeName == 'mi'
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sympy/printing/mathml.py b/sympy/printing/mathml.py
--- a/sympy/printing/mathml.py
+++ b/sympy/printing/mathml.py
@@ -1271,6 +1271,26 @@ def _print_Lambda(self, e):
         return x
 
 
+    def _print_tuple(self, e):
+        x = self.dom.createElement('mfenced')
+        for i in e:
+            x.appendChild(self._print(i))
+        return x
+
+
+    def _print_IndexedBase(self, e):
+        return self._print(e.label)
+
+    def _print_Indexed(self, e):
+        x = self.dom.createElement('msub')
+        x.appendChild(self._print(e.base))
+        if len(e.indices) == 1:
+            x.appendChild(self._print(e.indices[0]))
+            return x
+        x.appendChild(self._print(e.indices))
+        return x
+
+
 def mathml(expr, printer='content', **settings):
     """Returns the MathML representation of expr. If printer is presentation then
      prints Presentation MathML else prints content MathML.
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sympy/printing/tests/test_mathml.py b/sympy/printing/tests/test_mathml.py
--- a/sympy/printing/tests/test_mathml.py
+++ b/sympy/printing/tests/test_mathml.py
@@ -1,7 +1,7 @@
 from sympy import diff, Integral, Limit, sin, Symbol, Integer, Rational, cos, \
     tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, E, I, oo, \
     pi, GoldenRatio, EulerGamma, Sum, Eq, Ne, Ge, Lt, Float, Matrix, Basic, S, \
-    MatrixSymbol, Function, Derivative, log, Lambda
+    MatrixSymbol, Function, Derivative, log, Lambda, IndexedBase, symbols
 from sympy.core.containers import Tuple
 from sympy.functions.elementary.complexes import re, im, Abs, conjugate
 from sympy.functions.elementary.integers import floor, ceiling
@@ -1139,3 +1139,17 @@ def test_print_random_symbol():
     R = RandomSymbol(Symbol('R'))
     assert mpp.doprint(R) == '<mi>R</mi>'
     assert mp.doprint(R) == '<ci>R</ci>'
+
+
+def test_print_IndexedBase():
+    a,b,c,d,e = symbols('a b c d e')
+    assert mathml(IndexedBase(a)[b],printer='presentation') == '<msub><mi>a</mi><mi>b</mi></msub>'
+    assert mathml(IndexedBase(a)[b,c,d],printer = 'presentation') == '<msub><mi>a</mi><mfenced><mi>b</mi><mi>c</mi><mi>d</mi></mfenced></msub>'
+    assert mathml(IndexedBase(a)[b]*IndexedBase(c)[d]*IndexedBase(e),printer = 'presentation') == '<mrow><msub><mi>a</mi><mi>b</mi></msub><mo>&InvisibleTimes;</mo><msub><mi>c</mi><mi>d</mi></msub><mo>&InvisibleTimes;</mo><mi>e</mi></mrow>'
+
+
+def test_print_Indexed():
+    a,b,c = symbols('a b c')
+    assert mathml(IndexedBase(a),printer = 'presentation') == '<mi>a</mi>'
+    assert mathml(IndexedBase(a/b),printer = 'presentation') == '<mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow>'
+    assert mathml(IndexedBase((a,b)),printer = 'presentation') == '<mrow><mfenced><mi>a</mi><mi>b</mi></mfenced></mrow>'
```

</details>

<details>
<summary>Verification command at step 27 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 167
Step: 27

COMMAND
python -m pytest sympy/printing/tests/test_mathml.py -q -k indexed

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.352
stdout:
[empty]
stderr:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
```

</details>

<details>
<summary>Verification command at step 28 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 172
Step: 28

COMMAND
python -m sympy.testing.runtests sympy/printing/tests/test_mathml.py -k indexed 2>&1 | tail -40

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.645
stdout:
/opt/miniconda3/envs/testbed/bin/python: Error while finding module specification for 'sympy.testing.runtests' (ModuleNotFoundError: No module named 'sympy.testing')
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 29 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 177
Step: 29

COMMAND
ls bin | head; ls sympy | grep -i test | head; grep -n "runtests\|test runner" setup.py | head

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.331
stdout:
ask_update.py
authors_update.py
coverage_doctest.py
coverage_report.py
diagnose_imports
doctest
generate_module_list.py
generate_test_list.py
get_sympy.py
isympy
conftest.py
254:        from sympy.utilities import runtests
255:        runtests.run_all_tests()
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 182
Step: 30

COMMAND
python bin/test sympy/printing/tests/test_mathml.py 2>&1 | tail -50

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.188
stdout:
______________ sympy/printing/tests/test_mathml.py:test_print_Abs ______________
Traceback (most recent call last):
  File "/testbed/sympy/printing/tests/test_mathml.py", line 982, in test_print_Abs
    assert mpp.doprint(Abs(x)) == '<mrow><mfenced close="|" open="|"><mi>x</mi></mfenced></mrow>'
AssertionError
________________________________________________________________________________
__________ sympy/printing/tests/test_mathml.p... [truncated]
```

</details>

<details>
<summary>Verification command at step 31 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 187
Step: 31

COMMAND
python bin/test sympy/printing/tests/test_mathml.py -k indexed 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.100
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
numpy:              None
random seed:        45279056
hash randomization: on (PYTHONHASHSEED=455831570)

sympy/printing/tests/test_mathml.py[2] FF                                 [FAIL]

___... [truncated]
```

</details>

<details>
<summary>Verification command at step 35 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 207
Step: 35

COMMAND
python bin/test sympy/printing/tests/test_mathml.py -k indexed 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.113
stdout:
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.20-final-0) [CPython]
architecture:       64-bit
cache:              yes
ground types:       python 
numpy:              None
random seed:        87919643
hash randomization: on (PYTHONHASHSEED=4169658477)

sympy/printing/tests/test_mathml.py[2] ..                                   [OK]

==... [truncated]
```

</details>

<details>
<summary>Verification command at step 36 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 212
Step: 36

COMMAND
python bin/test sympy/printing/tests/test_mathml.py 2>&1 | tail -8; python - <<'PY'
from sympy.abc import a, b
from sympy import sin, IndexedBase, Idx
from sympy.printing.mathml import mathml
A = IndexedBase(a)
print(mathml(A[b]))
print(mathml(A[b], printer='presentation'))
print(mathml(A[b] + 1))
print(mathml(sin(A[b])))
print(mathml(A[Idx('i', 3), b]))
PY

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.486
stdout:
____________ sympy/printing/tests/test_mathml.py:test_print_ceiling ____________
Traceback (most recent call last):
  File "/testbed/sympy/printing/tests/test_mathml.py", line 1149, in test_print_ceiling
    assert mathml(expr, printer='presentation') == '<mrow><mfenced close="&#8969;" open="&#8968;"><mi>x</mi></mfenced></mrow>'
AssertionError

============= tests finished: 57 passed, 9 failed, in 0.08 seconds =============
DO *NOT* C... [truncated]
```

</details>


[Return to main report](../README.md)
