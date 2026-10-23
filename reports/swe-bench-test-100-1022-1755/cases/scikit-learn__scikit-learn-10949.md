# scikit-learn__scikit-learn-10949: Collapsing DataFrame dtypes loses per-column conversion information

**Primary category:** S — Unclear or conflicting acceptance contract (reclassified from E on review). **Confidence:** high.

**Recorded score:** F2P 0/1; P2P 42/42. **Steps:** 26/40. Counts are harness classifications, subject to the report's parser caveats.

The log first fails on exact warning wording; static review also identifies a real mixed-dtype gap.

## Issue requirement

warn_on_dtype should warn for DataFrame input when data is converted; the example uses an all-object DataFrame.

## Agent change

Collapses the DataFrame dtypes into np.result_type and reuses the single-original-dtype warning path.

## Golden change

Retains each column dtype and compares the set of original dtypes with the result dtype, issuing a DataFrame-specific message.

## Observed failure

The first golden assertion receives "was converted" instead of "were all converted". Later mixed-dtype assertions in that test are not reached in the recorded run.

## Scope and attribution

The only observed failure is the warning text: the agent emits the issue's "Expected result" verbatim (`was converted to`), while the gold test requires `were all converted to`. Collapsing per-column dtypes with `np.result_type` would also miss mixed int/float conversion (secondary E), but that is a static inference, not an observed failure.

## Verification in the trajectory

The agent runs dtype/DataFrame-focused tests and a filtered validation suite, but the added cases do not establish all mixed-column conversion behaviors.

## What would improve the outcome

Preserve per-column metadata until after conversion. Test homogeneous object data, mixed numeric data, mixed object/numeric data, no conversion and explicit requested dtypes.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `scikit-learn__scikit-learn-10949`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 64 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 64 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `scikit-learn__scikit-learn-10949.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
warn_on_dtype with DataFrame
#### Description

``warn_on_dtype`` has no effect when input is a pandas ``DataFrame``

#### Steps/Code to Reproduce
```python
from sklearn.utils.validation import check_array
import pandas as pd
df = pd.DataFrame([[1, 2, 3], [2, 3, 4]], dtype=object)
checked = check_array(df, warn_on_dtype=True)
```

#### Expected result: 

```python-traceback
DataConversionWarning: Data with input dtype object was converted to float64.
```

#### Actual Results
No warning is thrown

#### Versions
Linux-4.4.0-116-generic-x86_64-with-debian-stretch-sid
Python 3.6.3 |Anaconda, Inc.| (default, Nov  3 2017, 19:19:16) 
[GCC 7.2.0]
NumPy 1.13.1
SciPy 0.19.1
Scikit-Learn 0.20.dev0
Pandas 0.21.0

warn_on_dtype with DataFrame
#### Description

``warn_on_dtype`` has no effect when input is a pandas ``DataFrame``

#### Steps/Code to Reproduce
```python
from sklearn.utils.validation import check_array
import pandas as pd
df = pd.DataFrame([[1, 2, 3], [2, 3, 4]], dtype=object)
checked = check_array(df, warn_on_dtype=True)
```

#### Expected result: 

```python-traceback
DataConversionWarning: Data with input dtype object was converted to float64.
```

#### Actual Results
No warning is thrown

#### Versions
Linux-4.4.0-116-generic-x86_64-with-debian-stretch-sid
Python 3.6.3 |Anaconda, Inc.| (default, Nov  3 2017, 19:19:16) 
[GCC 7.2.0]
NumPy 1.13.1
SciPy 0.19.1
Scikit-Learn 0.20.dev0
Pandas 0.21.0
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text

```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
index fe1f723..a8b406b 100755
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -466,6 +466,15 @@ def check_array(array, accept_sparse=False, accept_large_sparse=True,
         # not a data type (e.g. a column named dtype in a pandas DataFrame)
         dtype_orig = None
 
+    if hasattr(array, "dtypes") and dtype_orig is None:
+        # pandas DataFrame does not have a dtype attribute but a dtypes
+        # attribute
+        dtypes = np.array(array.dtypes)
+        if len(dtypes) > 0 and all(isinstance(dtype, np.dtype)
+                                   for dtype in dtypes):
+            # numpy warns if dtypes are not converted into dtype
+            dtype_orig = np.result_type(*dtypes)
+
     if dtype_numeric:
         if dtype_orig is not None and dtype_orig.kind == "O":
             # if input is object, convert to float.
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sklearn/utils/tests/test_validation.py b/sklearn/utils/tests/test_validation.py
index 00182fa..cad984d 100755
--- a/sklearn/utils/tests/test_validation.py
+++ b/sklearn/utils/tests/test_validation.py
@@ -338,6 +338,19 @@ def test_check_array_pandas_dtype_object_conversion():
     assert_equal(check_array(X_df, ensure_2d=False).dtype.kind, "f")
 
 
+def test_check_array_warn_on_dtype_dataframe():
+    # warn_on_dtype should also work for pandas DataFrames whose dtype is
+    # inferred from the dtypes of their columns
+    pd = pytest.importorskip('pandas')
+    X_df = pd.DataFrame([[1, 2, 3], [2, 3, 4]], dtype=object)
+    X_checked = assert_warns(DataConversionWarning, check_array, X_df,
+                             warn_on_dtype=True)
+    assert_equal(X_checked.dtype, np.float64)
+
+    X_df = X_df.astype(np.float64)
+    assert_no_warnings(check_array, X_df, warn_on_dtype=True)
+
+
 def test_check_array_on_mock_dataframe():
     arr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
     mock_df = MockDataFrame(arr)
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -466,6 +466,12 @@ def check_array(array, accept_sparse=False, accept_large_sparse=True,
         # not a data type (e.g. a column named dtype in a pandas DataFrame)
         dtype_orig = None
 
+    # check if the object contains several dtypes (typically a pandas
+    # DataFrame), and store them. If not, store None.
+    dtypes_orig = None
+    if hasattr(array, "dtypes") and hasattr(array, "__array__"):
+        dtypes_orig = np.array(array.dtypes)
+
     if dtype_numeric:
         if dtype_orig is not None and dtype_orig.kind == "O":
             # if input is object, convert to float.
@@ -581,6 +587,16 @@ def check_array(array, accept_sparse=False, accept_large_sparse=True,
     if copy and np.may_share_memory(array, array_orig):
         array = np.array(array, dtype=dtype, order=order)
 
+    if (warn_on_dtype and dtypes_orig is not None and
+            {array.dtype} != set(dtypes_orig)):
+        # if there was at the beginning some other types than the final one
+        # (for instance in a DataFrame that can contain several dtypes) then
+        # some data must have been converted
+        msg = ("Data with input dtype %s were all converted to %s%s."
+               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
+                  context))
+        warnings.warn(msg, DataConversionWarning, stacklevel=3)
+
     return array
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sklearn/utils/tests/test_validation.py b/sklearn/utils/tests/test_validation.py
--- a/sklearn/utils/tests/test_validation.py
+++ b/sklearn/utils/tests/test_validation.py
@@ -7,6 +7,7 @@
 from itertools import product
 
 import pytest
+from pytest import importorskip
 import numpy as np
 import scipy.sparse as sp
 from scipy import __version__ as scipy_version
@@ -713,6 +714,38 @@ def test_suppress_validation():
     assert_raises(ValueError, assert_all_finite, X)
 
 
+def test_check_dataframe_warns_on_dtype():
+    # Check that warn_on_dtype also works for DataFrames.
+    # https://github.com/scikit-learn/scikit-learn/issues/10948
+    pd = importorskip("pandas")
+
+    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], dtype=object)
+    assert_warns_message(DataConversionWarning,
+                         "Data with input dtype object were all converted to "
+                         "float64.",
+                         check_array, df, dtype=np.float64, warn_on_dtype=True)
+    assert_warns(DataConversionWarning, check_array, df,
+                 dtype='numeric', warn_on_dtype=True)
+    assert_no_warnings(check_array, df, dtype='object', warn_on_dtype=True)
+
+    # Also check that it raises a warning for mixed dtypes in a DataFrame.
+    df_mixed = pd.DataFrame([['1', 2, 3], ['4', 5, 6]])
+    assert_warns(DataConversionWarning, check_array, df_mixed,
+                 dtype=np.float64, warn_on_dtype=True)
+    assert_warns(DataConversionWarning, check_array, df_mixed,
+                 dtype='numeric', warn_on_dtype=True)
+    assert_warns(DataConversionWarning, check_array, df_mixed,
+                 dtype=object, warn_on_dtype=True)
+
+    # Even with numerical dtypes, a conversion can be made because dtypes are
+    # uniformized throughout the array.
+    df_mixed_numeric = pd.DataFrame([[1., 2, 3], [4., 5, 6]])
+    assert_warns(DataConversionWarning, check_array, df_mixed_numeric,
+                 dtype='numeric', warn_on_dtype=True)
+    assert_no_warnings(check_array, df_mixed_numeric.astype(int),
+                       dtype='numeric', warn_on_dtype=True)
+
+
 class DummyMemory(object):
     def cache(self, func):
         return func
```

</details>

<details>
<summary>Verification command at step 12 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 93
Step: 12

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q -k "dtype" 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.224
stdout:
...                                                                      [100%]
3 passed, 49 deselected, 1 warning in 0.70s
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 15 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 108
Step: 15

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q -k "dtype or dataframe" 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.192
stdout:
......                                                                   [100%]
6 passed, 47 deselected in 0.67s
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 18 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 129
Step: 18

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q 2>&1 | tail -10

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.432
stdout:
    def test_check_array_force_all_finiteinvalid(value, force_all_finite,
                                                 match_msg, retype):
        X = retype(np.arange(4).reshape(2, 2).astype(np.float))
        X[0, 0] = value
>       with pytest.raises(ValueError, message=match_msg):
E       TypeError: Unexpected keyword arguments passed to pytest.raises: message
E       Use context-manager form instead?

sklearn/utils/tests/test... [truncated]
```

</details>

<details>
<summary>Verification command at step 20 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 140
Step: 20

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q 2>&1 | grep -E "^(FAILED|sklearn.*F)" ; python -m pytest sklearn/utils/tests/test_validation.py -q 2>&1 | grep -E "^sklearn.*(FAIL|F$)" | head -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 2.692
stdout:
[empty]
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 20 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 142
Step: 20

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q -k "dtype or dataframe or check_array" 2>&1 | tail -15

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.294
stdout:
    )
    @pytest.mark.parametrize(
        "retype",
        [np.asarray, sp.csr_matrix]
    )
    def test_check_array_force_all_finiteinvalid(value, force_all_finite,
                                                 match_msg, retype):
        X = retype(np.arange(4).reshape(2, 2).astype(np.float))
        X[0, 0] = value
>       with pytest.raises(ValueError, message=match_msg):
E       TypeError: Unexpected keyword arguments pass... [truncated]
```

</details>

<details>
<summary>Verification command at step 21 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 147
Step: 21

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q 2>&1 | grep -E "^sklearn/utils/tests/test_validation.py:[0-9]+|^_____" | head -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.461
stdout:
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklearn/utils/tests/test_validation.py:184: TypeError
sklear... [truncated]
```

</details>

<details>
<summary>Verification command at step 22 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 153
Step: 22

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q -k "dtype or dataframe or pandas" 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.165
stdout:
......                                                                   [100%]
6 passed, 47 deselected in 0.64s
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 24 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 166
Step: 24

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q -k "not force_all_finiteinvalid" 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.361
stdout:
...........................................                              [100%]
43 passed, 10 deselected, 9 warnings in 0.79s
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
