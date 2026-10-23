# mwaskom__seaborn-3407: Plotting succeeds; an internal-container equality assertion fails

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P 0/1; P2P 120/120. **Steps:** 19/40. Counts are harness classifications, subject to the report's parser caveats.

A one-dimensional object array fixes the tuple-label bug, but the golden test assumes a list.

## Issue requirement

pairplot must accept a DataFrame with MultiIndex columns without raising KeyError.

## Agent change

Allocates one-dimensional object arrays and assigns tuple labels into them, preventing NumPy from creating a two-dimensional array.

## Golden change

Stores the original Python lists directly for diag_vars and diag_axes.

## Observed failure

The golden test gets past g = pairplot(df). It fails only at assert g.diag_vars == list(cols), with ValueError about an array truth value. All 120 P2P tests pass.

## Scope and attribution

The observed failure is tied to container/equality semantics, not the original plotting crash. The issue does not specify a list return type for diag_vars. This is strong evidence of implementation-sensitive acceptance, although it does not establish equivalence for every downstream use.

## Verification in the trajectory

The agent identified the tuple-to-2D-array root cause and its PairGrid test run reports 47 passed. Its own test uses array-compatible assertions.

## What would improve the outcome

For benchmark review, assert label contents and successful plotting unless a list is a documented contract. For implementation simplicity, retain the existing list instead of converting it unnecessarily.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 50 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 50 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 50 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/mwaskom__seaborn-3407/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/mwaskom__seaborn-3407/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/mwaskom__seaborn-3407/run_instance.log) | — |
| trajectory | [mwaskom__seaborn-3407.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/mwaskom__seaborn-3407.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

````text
pairplot raises KeyError with MultiIndex DataFrame
When trying to pairplot a MultiIndex DataFrame, `pairplot` raises a `KeyError`:

MRE:

```python
import numpy as np
import pandas as pd
import seaborn as sns


data = {
    ("A", "1"): np.random.rand(100),
    ("A", "2"): np.random.rand(100),
    ("B", "1"): np.random.rand(100),
    ("B", "2"): np.random.rand(100),
}
df = pd.DataFrame(data)
sns.pairplot(df)
```

Output:

```
[c:\Users\KLuu\anaconda3\lib\site-packages\seaborn\axisgrid.py](file:///C:/Users/KLuu/anaconda3/lib/site-packages/seaborn/axisgrid.py) in pairplot(data, hue, hue_order, palette, vars, x_vars, y_vars, kind, diag_kind, markers, height, aspect, corner, dropna, plot_kws, diag_kws, grid_kws, size)
   2142     diag_kws.setdefault("legend", False)
   2143     if diag_kind == "hist":
-> 2144         grid.map_diag(histplot, **diag_kws)
   2145     elif diag_kind == "kde":
   2146         diag_kws.setdefault("fill", True)

[c:\Users\KLuu\anaconda3\lib\site-packages\seaborn\axisgrid.py](file:///C:/Users/KLuu/anaconda3/lib/site-packages/seaborn/axisgrid.py) in map_diag(self, func, **kwargs)
   1488                 plt.sca(ax)
   1489 
-> 1490             vector = self.data[var]
   1491             if self._hue_var is not None:
   1492                 hue = self.data[self._hue_var]

[c:\Users\KLuu\anaconda3\lib\site-packages\pandas\core\frame.py](file:///C:/Users/KLuu/anaconda3/lib/site-packages/pandas/core/frame.py) in __getitem__(self, key)
   3765             if is_iterator(key):
   3766                 key = list(key)
-> 3767             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   3768 
   3769         # take() does not accept boolean indexers

[c:\Users\KLuu\anaconda3\lib\site-packages\pandas\core\indexes\multi.py](file:///C:/Users/KLuu/anaconda3/lib/site-packages/pandas/core/indexes/multi.py) in _get_indexer_strict(self, key, axis_name)
   2534             indexer = self._get_indexer_level_0(keyarr)
   2535 
-> 2536             self._raise_if_missing(key, indexer, axis_name)
   2537             return self[indexer], indexer
   2538 

[c:\Users\KLuu\anaconda3\lib\site-packages\pandas\core\indexes\multi.py](file:///C:/Users/KLuu/anaconda3/lib/site-packages/pandas/core/indexes/multi.py) in _raise_if_missing(self, key, indexer, axis_name)
   2552                 cmask = check == -1
   2553                 if cmask.any():
-> 2554                     raise KeyError(f"{keyarr[cmask]} not in index")
   2555                 # We get here when levels still contain values which are not
   2556                 # actually in Index anymore

KeyError: "['1'] not in index"
```

A workaround is to "flatten" the columns:

```python
df.columns = ["".join(column) for column in df.columns]
```
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/seaborn/axisgrid.py b/seaborn/axisgrid.py
index 7534909..8ff2b02 100755
--- a/seaborn/axisgrid.py
+++ b/seaborn/axisgrid.py
@@ -1472,8 +1472,12 @@ class PairGrid(Grid):
                 for ax in diag_axes[1:]:
                     share_axis(diag_axes[0], ax, "y")
 
-            self.diag_vars = np.array(diag_vars, np.object_)
-            self.diag_axes = np.array(diag_axes, np.object_)
+            # Note: passing tuples to np.array would create a 2D array, which
+            # would then be interpreted as a list of keys when indexing data
+            self.diag_vars = np.empty(len(diag_vars), object)
+            self.diag_vars[:] = diag_vars
+            self.diag_axes = np.empty(len(diag_axes), object)
+            self.diag_axes[:] = diag_axes
 
         if "hue" not in signature(func).parameters:
             return self._map_diag_iter_hue(func, **kwargs)
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/test_axisgrid.py b/tests/test_axisgrid.py
index ce8b0ea..c554777 100755
--- a/tests/test_axisgrid.py
+++ b/tests/test_axisgrid.py
@@ -750,6 +750,28 @@ class TestPairGrid:
         assert g.y_vars == ["x", "y", "z"]
         assert g.square_grid
 
+    def test_multiindex_columns(self):
+
+        cols = pd.MultiIndex.from_tuples(
+            [("a", "x"), ("a", "y"), ("a", "z"), ("b", "a"), ("b", "b")]
+        )
+        df = self.df.copy()
+        df.columns = cols
+        plot_vars = [("a", "x"), ("a", "y"), ("a", "z")]
+
+        g = ag.PairGrid(df)
+        assert g.x_vars == plot_vars
+        assert g.y_vars == plot_vars
+
+        g.map_diag(plt.hist)
+        for var, ax in zip(g.diag_vars, g.diag_axes):
+            assert var in plot_vars
+            assert len(ax.patches) == 10
+
+        g = ag.pairplot(df)
+        for ax in g.diag_axes:
+            assert len(ax.patches) > 1
+
     @pytest.mark.parametrize("vars", [["z", "x"], np.array(["z", "x"])])
     def test_specific_square_axes(self, vars):
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/seaborn/axisgrid.py b/seaborn/axisgrid.py
--- a/seaborn/axisgrid.py
+++ b/seaborn/axisgrid.py
@@ -1472,8 +1472,8 @@ def map_diag(self, func, **kwargs):
                 for ax in diag_axes[1:]:
                     share_axis(diag_axes[0], ax, "y")
 
-            self.diag_vars = np.array(diag_vars, np.object_)
-            self.diag_axes = np.array(diag_axes, np.object_)
+            self.diag_vars = diag_vars
+            self.diag_axes = diag_axes
 
         if "hue" not in signature(func).parameters:
             return self._map_diag_iter_hue(func, **kwargs)
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/test_axisgrid.py b/tests/test_axisgrid.py
--- a/tests/test_axisgrid.py
+++ b/tests/test_axisgrid.py
@@ -1422,6 +1422,13 @@ def test_pairplot_markers(self):
         with pytest.warns(UserWarning):
             g = ag.pairplot(self.df, hue="a", vars=vars, markers=markers[:-2])
 
+    def test_pairplot_column_multiindex(self):
+
+        cols = pd.MultiIndex.from_arrays([["x", "y"], [1, 2]])
+        df = self.df[["x", "y"]].set_axis(cols, axis=1)
+        g = ag.pairplot(df)
+        assert g.diag_vars == list(cols)
+
     def test_corner_despine(self):
 
         g = ag.PairGrid(self.df, corner=True, despine=False)
```

</details>

<details>
<summary>Verification command at step 11 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 76
Step: 11

COMMAND
python3 -m pytest tests/test_axisgrid.py::TestPairGrid::test_multiindex_columns -x -q 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 2.154
stdout:
           )
key = array(['a', 'x'], dtype=object), indexer = array([ 0,  1,  2, -1])
axis_name = 'columns'

    def _raise_if_missing(self, key, indexer, axis_name: str) -> None:
        keyarr = key
        if not isinstance(key, Index):
            keyarr = com.asarray_tuplesafe(key)
    
        if len(keyarr) and not isinstance(keyarr[0], tuple):
            # i.e. same condition for special case in MultiIndex._get_indexer_strict... [truncated]
```

</details>

<details>
<summary>Verification command at step 13 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 86
Step: 13

COMMAND
python3 -m pytest tests/test_axisgrid.py::TestPairGrid -q 2>&1 | tail -20

STORED OUTPUT PREVIEW
Validation error for tool 'run_command':
  - field 'timeout_seconds': Input should be less than or equal to 120
```

</details>

<details>
<summary>Verification command at step 14 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 91
Step: 14

COMMAND
python3 -m pytest tests/test_axisgrid.py::TestPairGrid -q 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 28.507
stdout:
...............................................                          [100%]
=============================== warnings summary ===============================
tests/test_axisgrid.py: 11 warnings
  /testbed/seaborn/axisgrid.py:118: UserWarning: The figure layout has changed to tight
    self._figure.tight_layout(*args, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
47 passed, 11 warnings in 27.43s... [truncated]
```

</details>

<details>
<summary>Verification command at step 17 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 109
Step: 17

COMMAND
python3 -m pytest tests/test_axisgrid.py -q 2>&1 | tail -8

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 48.388
stdout:
tests/test_axisgrid.py::TestFacetGrid::test_categorical_warning
tests/test_axisgrid.py::TestFacetGrid::test_categorical_warning
  /opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pandas/core/algorithms.py:525: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
  See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy ... [truncated]
```

</details>


[Return to main report](../README.md)
