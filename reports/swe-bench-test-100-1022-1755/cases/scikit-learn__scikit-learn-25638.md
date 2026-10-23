# scikit-learn__scikit-learn-25638: Nullable Series are handled, but nullable DataFrames are not

**Primary category:** E — Missing edge case or input/operation family. **Confidence:** high.

**Recorded score:** F2P 9/10; P2P 194/194. **Steps:** 33/40. Counts are harness classifications, subject to the report's parser caveats.

A Series-specific dtype conversion misses the two-dimensional input family.

## Issue requirement

Support pandas nullable numeric/boolean dtypes in unique_labels; all shown examples are Series.

## Agent change

Converts inputs that have to_numpy, dtype and dtype.numpy_dtype inside type_of_target.

## Golden change

Uses the shared check_array conversion in both type_of_target and is_multilabel, covering Series and DataFrames.

## Observed failure

9/10 F2P tests pass. The remaining test reaches a Float32 DataFrame and gets unknown instead of continuous-multioutput; all 194 P2P tests pass.

## Scope and attribution

The golden tests extend beyond the Series example, but nullable DataFrames are a natural input-family boundary of the shared classifier. This is a strong coverage-gap case.

## Verification in the trajectory

The added unique_labels test covers the three requested nullable Series dtypes. The source condition depends on singular dtype, which DataFrames do not expose as their per-column type interface.

## What would improve the outcome

Reuse shared validation and build a matrix of Series/DataFrame, single/multiple outputs, and numeric/boolean nullable dtypes.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 72 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 72 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 72 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/scikit-learn__scikit-learn-25638/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/scikit-learn__scikit-learn-25638/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/scikit-learn__scikit-learn-25638/run_instance.log) | — |
| trajectory | [scikit-learn__scikit-learn-25638.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/scikit-learn__scikit-learn-25638.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

````text
Support nullable pandas dtypes in `unique_labels`
### Describe the workflow you want to enable

I would like to be able to pass the nullable pandas dtypes ("Int64", "Float64", "boolean") into sklearn's `unique_labels` function. Because the dtypes become `object` dtype when converted to numpy arrays we get `ValueError: Mix type of y not allowed, got types {'binary', 'unknown'}`:

Repro with sklearn 1.2.1
```py 
    import pandas as pd
    import pytest
    from sklearn.utils.multiclass import unique_labels
    
    for dtype in ["Int64", "Float64", "boolean"]:
        y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
        y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")

        with pytest.raises(ValueError, match="Mix type of y not allowed, got types"):
            unique_labels(y_true, y_predicted)
```

### Describe your proposed solution

We should get the same behavior as when `int64`, `float64`, and `bool` dtypes are used, which is no error:  

```python
    import pandas as pd
    from sklearn.utils.multiclass import unique_labels
    
    for dtype in ["int64", "float64", "bool"]:
        y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
        y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")

        unique_labels(y_true, y_predicted)
```

### Describe alternatives you've considered, if relevant

Our current workaround is to convert the data to numpy arrays with the corresponding dtype that works prior to passing it into `unique_labels`.

### Additional context

_No response_
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sklearn/utils/multiclass.py b/sklearn/utils/multiclass.py
index 5eaef2f..92eeae2 100755
--- a/sklearn/utils/multiclass.py
+++ b/sklearn/utils/multiclass.py
@@ -295,6 +295,21 @@ def type_of_target(y, input_name=""):
     if sparse_pandas:
         raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")
 
+    # pandas nullable dtypes are converted to object dtype by np.asarray,
+    # which prevents type_of_target from inferring the correct target type.
+    # Convert them to their corresponding numpy dtype instead.
+    if (
+        hasattr(y, "to_numpy")
+        and hasattr(y, "dtype")
+        and hasattr(y.dtype, "numpy_dtype")
+    ):
+        try:
+            y = y.to_numpy(dtype=y.dtype.numpy_dtype)
+        except (TypeError, ValueError):
+            # Missing values cannot be represented in the corresponding numpy
+            # dtype; let the generic object dtype conversion handle them.
+            pass
+
     if is_multilabel(y):
         return "multilabel-indicator"
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/sklearn/utils/tests/test_multiclass.py b/sklearn/utils/tests/test_multiclass.py
index cf5858d..b9223d9 100755
--- a/sklearn/utils/tests/test_multiclass.py
+++ b/sklearn/utils/tests/test_multiclass.py
@@ -204,6 +204,17 @@ def test_unique_labels():
     assert_array_equal(unique_labels(np.ones((4, 5)), np.ones((5, 5))), np.arange(5))
 
 
+@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
+def test_unique_labels_pandas_nullable_dtype(dtype):
+    """Check that unique_labels supports pandas nullable dtypes."""
+    pd = pytest.importorskip("pandas")
+
+    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
+    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")
+
+    assert_array_equal(unique_labels(y_true, y_predicted), [0, 1])
+
+
 def test_unique_labels_non_specific():
     # Test unique_labels with a variety of collected examples
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sklearn/utils/multiclass.py b/sklearn/utils/multiclass.py
--- a/sklearn/utils/multiclass.py
+++ b/sklearn/utils/multiclass.py
@@ -155,14 +155,25 @@ def is_multilabel(y):
     if hasattr(y, "__array__") or isinstance(y, Sequence) or is_array_api:
         # DeprecationWarning will be replaced by ValueError, see NEP 34
         # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
+        check_y_kwargs = dict(
+            accept_sparse=True,
+            allow_nd=True,
+            force_all_finite=False,
+            ensure_2d=False,
+            ensure_min_samples=0,
+            ensure_min_features=0,
+        )
         with warnings.catch_warnings():
             warnings.simplefilter("error", np.VisibleDeprecationWarning)
             try:
-                y = xp.asarray(y)
-            except (np.VisibleDeprecationWarning, ValueError):
+                y = check_array(y, dtype=None, **check_y_kwargs)
+            except (np.VisibleDeprecationWarning, ValueError) as e:
+                if str(e).startswith("Complex data not supported"):
+                    raise
+
                 # dtype=object should be provided explicitly for ragged arrays,
                 # see NEP 34
-                y = xp.asarray(y, dtype=object)
+                y = check_array(y, dtype=object, **check_y_kwargs)
 
     if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
         return False
@@ -302,15 +313,27 @@ def type_of_target(y, input_name=""):
     # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
     # We therefore catch both deprecation (NumPy < 1.24) warning and
     # value error (NumPy >= 1.24).
+    check_y_kwargs = dict(
+        accept_sparse=True,
+        allow_nd=True,
+        force_all_finite=False,
+        ensure_2d=False,
+        ensure_min_samples=0,
+        ensure_min_features=0,
+    )
+
     with warnings.catch_warnings():
         warnings.simplefilter("error", np.VisibleDeprecationWarning)
         if not issparse(y):
             try:
-                y = xp.asarray(y)
-            except (np.VisibleDeprecationWarning, ValueError):
+                y = check_array(y, dtype=None, **check_y_kwargs)
+            except (np.VisibleDeprecationWarning, ValueError) as e:
+                if str(e).startswith("Complex data not supported"):
+                    raise
+
                 # dtype=object should be provided explicitly for ragged arrays,
                 # see NEP 34
-                y = xp.asarray(y, dtype=object)
+                y = check_array(y, dtype=object, **check_y_kwargs)
 
     # The old sequence of sequences format
     try:
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/sklearn/metrics/tests/test_classification.py b/sklearn/metrics/tests/test_classification.py
--- a/sklearn/metrics/tests/test_classification.py
+++ b/sklearn/metrics/tests/test_classification.py
@@ -1079,6 +1079,24 @@ def test_confusion_matrix_dtype():
     assert cm[1, 1] == -2
 
 
+@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
+def test_confusion_matrix_pandas_nullable(dtype):
+    """Checks that confusion_matrix works with pandas nullable dtypes.
+
+    Non-regression test for gh-25635.
+    """
+    pd = pytest.importorskip("pandas")
+
+    y_ndarray = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1])
+    y_true = pd.Series(y_ndarray, dtype=dtype)
+    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")
+
+    output = confusion_matrix(y_true, y_predicted)
+    expected_output = confusion_matrix(y_ndarray, y_predicted)
+
+    assert_array_equal(output, expected_output)
+
+
 def test_classification_report_multiclass():
     # Test performance report
     iris = datasets.load_iris()
diff --git a/sklearn/preprocessing/tests/test_label.py b/sklearn/preprocessing/tests/test_label.py
--- a/sklearn/preprocessing/tests/test_label.py
+++ b/sklearn/preprocessing/tests/test_label.py
@@ -117,6 +117,22 @@ def test_label_binarizer_set_label_encoding():
     assert_array_equal(lb.inverse_transform(got), inp)
 
 
+@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
+def test_label_binarizer_pandas_nullable(dtype):
+    """Checks that LabelBinarizer works with pandas nullable dtypes.
+
+    Non-regression test for gh-25637.
+    """
+    pd = pytest.importorskip("pandas")
+    from sklearn.preprocessing import LabelBinarizer
+
+    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
+    lb = LabelBinarizer().fit(y_true)
+    y_out = lb.transform([1, 0])
+
+    assert_array_equal(y_out, [[1], [0]])
+
+
 @ignore_warnings
 def test_label_binarizer_errors():
     # Check that invalid arguments yield ValueError
diff --git a/sklearn/utils/tests/test_multiclass.py b/sklearn/utils/tests/test_multiclass.py
--- a/sklearn/utils/tests/test_multiclass.py
+++ b/sklearn/utils/tests/test_multiclass.py
@@ -346,6 +346,42 @@ def test_type_of_target_pandas_sparse():
         type_of_target(y)
 
 
+def test_type_of_target_pandas_nullable():
+    """Check that type_of_target works with pandas nullable dtypes."""
+    pd = pytest.importorskip("pandas")
+
+    for dtype in ["Int32", "Float32"]:
+        y_true = pd.Series([1, 0, 2, 3, 4], dtype=dtype)
+        assert type_of_target(y_true) == "multiclass"
+
+        y_true = pd.Series([1, 0, 1, 0], dtype=dtype)
+        assert type_of_target(y_true) == "binary"
+
+    y_true = pd.DataFrame([[1.4, 3.1], [3.1, 1.4]], dtype="Float32")
+    assert type_of_target(y_true) == "continuous-multioutput"
+
+    y_true = pd.DataFrame([[0, 1], [1, 1]], dtype="Int32")
+    assert type_of_target(y_true) == "multilabel-indicator"
+
+    y_true = pd.DataFrame([[1, 2], [3, 1]], dtype="Int32")
+    assert type_of_target(y_true) == "multiclass-multioutput"
+
+
+@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
+def test_unique_labels_pandas_nullable(dtype):
+    """Checks that unique_labels work with pandas nullable dtypes.
+
+    Non-regression test for gh-25634.
+    """
+    pd = pytest.importorskip("pandas")
+
+    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
+    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")
+
+    labels = unique_labels(y_true, y_predicted)
+    assert_array_equal(labels, [0, 1])
+
+
 def test_class_distribution():
     y = np.array(
         [
```

</details>

<details>
<summary>Verification command at step 18 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 120
Step: 18

COMMAND
python -m pytest sklearn/utils/tests/test_multiclass.py -k "pandas_nullable" -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.639
stdout:
        Returns
        -------
        out : ndarray of shape (n_unique_labels,)
            An ordered array of unique labels.
    
        Examples
        --------
        >>> from sklearn.utils.multiclass import unique_labels
        >>> unique_labels([3, 5, 5, 5, 7, 7])
        array([3, 5, 7])
        >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
        array([1, 2, 3, 4])
        >>> unique_labels([1, 2, 10], [5, 11])
       ... [truncated]
```

</details>

<details>
<summary>Verification command at step 27 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 174
Step: 27

COMMAND
python -m pytest sklearn/utils/tests/test_multiclass.py -k "pandas_nullable or unique_labels or type_of_target" -q 2>&1 | tail -15

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 2.057
stdout:
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[33m                                                                 [100%][0m
[33m[32m8 passed[0m, [33m[1m5 deselected[0m, [33m[1m1 warning[0m[33m in 0.76s[0m[0m
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 28 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 182
Step: 28

COMMAND
python -m pytest sklearn/utils/tests/test_multiclass.py -q 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.719
stdout:
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[33m                                                            [100%][0m
[33m[32m13 passed[0m, [33m[1m1 warning[0m[33m in 0.73s[0m[0m
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 193
Step: 30

COMMAND
python -m pytest sklearn/utils/tests/test_multiclass.py sklearn/tests/test_multiclass.py -q 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 2.735
stdout:
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[... [truncated]
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 195
Step: 30

COMMAND
python -m pytest sklearn/utils/tests/test_validation.py -q 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.735
stdout:
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[... [truncated]
```

</details>


[Return to main report](../README.md)
