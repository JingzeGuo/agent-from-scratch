# matplotlib__matplotlib-22711: The issue reproduces a polygon crash; the golden patch fixes handle synchronization

**Primary category:** G — Golden change extends or shifts the stated scope. **Confidence:** high.

**Recorded score:** F2P 0/2; P2P 91/91. **Steps:** 25/40. Counts are harness classifications, subject to the report's parser caveats.

The agent addresses the reported degenerate initialization, while grading checks a different RangeSlider behavior.

## Issue requirement

RangeSlider(valinit=[0, 0]) raises IndexError at xy[4]; the issue suggests removing that assignment.

## Agent change

Rebuilds polygon vertices and uses Polygon.set_xy, avoiding assumptions about an existing closing vertex.

## Golden change

Leaves the xy[4] assignments in place and instead updates handle coordinates, vertical dragging and handle selection.

## Observed failure

Both F2P orientation cases fail after set_val((0.2, 0.6)): handle positions remain [0.1, 0.34]. The failing assertion is not the reported IndexError. All 91 scored P2P tests pass.

## Scope and attribution

This is a strong issue/golden focus mismatch. Handle synchronization is a valid widget requirement, but the golden source and tests target a different symptom than the supplied reproduction.

## Verification in the trajectory

The agent wrote and ran equal-initial-value cases for both orientations, then ran slider-focused tests. A separate rectangle-selector deprecation failure appears in the broader log and is not a scored regression.

## What would improve the outcome

Retain this as a benchmark audit example. For broader validation, check value, polygon and handle positions together after initialization, set_val, reset and dragging.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `matplotlib__matplotlib-22711`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 43 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 43 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `matplotlib__matplotlib-22711.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
[Bug]: cannot give init value for RangeSlider widget
### Bug summary

I think `xy[4] = .25, val[0]` should be commented in /matplotlib/widgets. py", line 915, in set_val
as it prevents to initialized value for RangeSlider

### Code for reproduction

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

# generate a fake image
np.random.seed(19680801)
N = 128
img = np.random.randn(N, N)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(bottom=0.25)

im = axs[0].imshow(img)
axs[1].hist(img.flatten(), bins='auto')
axs[1].set_title('Histogram of pixel intensities')

# Create the RangeSlider
slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])

# Create the Vertical lines on the histogram
lower_limit_line = axs[1].axvline(slider.val[0], color='k')
upper_limit_line = axs[1].axvline(slider.val[1], color='k')


def update(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the image's colormap
    im.norm.vmin = val[0]
    im.norm.vmax = val[1]

    # Update the position of the vertical lines
    lower_limit_line.set_xdata([val[0], val[0]])
    upper_limit_line.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
```


### Actual outcome

```python
  File "<ipython-input-52-b704c53e18d4>", line 19, in <module>
    slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])

  File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 778, in __init__
    self.set_val(valinit)

  File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 915, in set_val
    xy[4] = val[0], .25

IndexError: index 4 is out of bounds for axis 0 with size 4
```

### Expected outcome

range slider with user initial values

### Additional information

error can be removed by commenting this line
```python

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
        val = np.sort(np.asanyarray(val))
        if val.shape != (2,):
            raise ValueError(
                f"val must have shape (2,) but has shape {val.shape}"
            )
        val[0] = self._min_in_bounds(val[0])
        val[1] = self._max_in_bounds(val[1])
        xy = self.poly.xy
        if self.orientation == "vertical":
            xy[0] = .25, val[0]
            xy[1] = .25, val[1]
            xy[2] = .75, val[1]
            xy[3] = .75, val[0]
            # xy[4] = .25, val[0]
        else:
            xy[0] = val[0], .25
            xy[1] = val[0], .75
            xy[2] = val[1], .75
            xy[3] = val[1], .25
            # xy[4] = val[0], .25
        self.poly.xy = xy
        self.valtext.set_text(self._format(val))
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process("changed", val)

```

### Operating system

OSX

### Matplotlib Version

3.5.1

### Matplotlib Backend

_No response_

### Python version

3.8

### Jupyter version

_No response_

### Installation

pip
````

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

````text
Huh, the polygon object must have changed inadvertently. Usually, you have
to "close" the polygon by repeating the first vertex, but we make it
possible for polygons to auto-close themselves. I wonder how (when?) this
broke?

On Tue, Mar 22, 2022 at 10:29 PM vpicouet ***@***.***> wrote:

> Bug summary
>
> I think xy[4] = .25, val[0] should be commented in /matplotlib/widgets.
> py", line 915, in set_val
> as it prevents to initialized value for RangeSlider
> Code for reproduction
>
> import numpy as npimport matplotlib.pyplot as pltfrom matplotlib.widgets import RangeSlider
> # generate a fake imagenp.random.seed(19680801)N = 128img = np.random.randn(N, N)
> fig, axs = plt.subplots(1, 2, figsize=(10, 5))fig.subplots_adjust(bottom=0.25)
> im = axs[0].imshow(img)axs[1].hist(img.flatten(), bins='auto')axs[1].set_title('Histogram of pixel intensities')
> # Create the RangeSliderslider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])
> # Create the Vertical lines on the histogramlower_limit_line = axs[1].axvline(slider.val[0], color='k')upper_limit_line = axs[1].axvline(slider.val[1], color='k')
>
> def update(val):
>     # The val passed to a callback by the RangeSlider will
>     # be a tuple of (min, max)
>
>     # Update the image's colormap
>     im.norm.vmin = val[0]
>     im.norm.vmax = val[1]
>
>     # Update the position of the vertical lines
>     lower_limit_line.set_xdata([val[0], val[0]])
>     upper_limit_line.set_xdata([val[1], val[1]])
>
>     # Redraw the figure to ensure it updates
>     fig.canvas.draw_idle()
>
> slider.on_changed(update)plt.show()
>
> Actual outcome
>
>   File "<ipython-input-52-b704c53e18d4>", line 19, in <module>
>     slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])
>
>   File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 778, in __init__
>     self.set_val(valinit)
>
>   File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 915, in set_val
>     xy[4] = val[0], .25
>
> IndexError: index 4 is out of bounds for axis 0 with size 4
>
> Expected outcome
>
> range slider with user initial values
> Additional information
>
> error can be
>
>
>     def set_val(self, val):
>         """
>         Set slider value to *val*.
>
>         Parameters
>         ----------
>         val : tuple or array-like of float
>         """
>         val = np.sort(np.asanyarray(val))
>         if val.shape != (2,):
>             raise ValueError(
>                 f"val must have shape (2,) but has shape {val.shape}"
>             )
>         val[0] = self._min_in_bounds(val[0])
>         val[1] = self._max_in_bounds(val[1])
>         xy = self.poly.xy
>         if self.orientation == "vertical":
>             xy[0] = .25, val[0]
>             xy[1] = .25, val[1]
>             xy[2] = .75, val[1]
>             xy[3] = .75, val[0]
>             # xy[4] = .25, val[0]
>         else:
>             xy[0] = val[0], .25
>             xy[1] = val[0], .75
>             xy[2] = val[1], .75
>             xy[3] = val[1], .25
>             # xy[4] = val[0], .25
>         self.poly.xy = xy
>         self.valtext.set_text(self._format(val))
>         if self.drawon:
>             self.ax.figure.canvas.draw_idle()
>         self.val = val
>         if self.eventson:
>             self._observers.process("changed", val)
>
>
> Operating system
>
> OSX
> Matplotlib Version
>
> 3.5.1
> Matplotlib Backend
>
> *No response*
> Python version
>
> 3.8
> Jupyter version
>
> *No response*
> Installation
>
> pip
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/matplotlib/matplotlib/issues/22686>, or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AACHF6CW2HVLKT5Q56BVZDLVBJ6X7ANCNFSM5RMUEIDQ>
> .
> You are receiving this because you are subscribed to this thread.Message
> ID: ***@***.***>
>

Yes, i might have been too fast, cause it allows to skip the error but then it seems that the polygon is not right...
Let me know if you know how this should be solved...
![Capture d’écran, le 2022-03-22 à 23 20 23](https://user-images.githubusercontent.com/37241971/159617326-44c69bfc-bf0a-4f79-ab23-925c7066f2c2.jpg)


So I think you found an edge case because your valinit has both values equal. This means that the poly object created by `axhspan` is not as large as the rest of the code expects. 

https://github.com/matplotlib/matplotlib/blob/11737d0694109f71d2603ba67d764aa2fb302761/lib/matplotlib/widgets.py#L722

A quick workaround is to have the valinit contain two different numbers (even if only minuscule difference)
Yes you are right!
Thanks a lot for digging into this!!
Compare:

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
poly_same_valinit = ax.axvspan(0, 0, 0, 1)
poly_diff_valinit = ax.axvspan(0, .5, 0, 1)
print(poly_same_valinit.xy.shape)
print(poly_diff_valinit.xy.shape)
```

which gives:

```
(4, 2)
(5, 2)
```

Two solutions spring to mind:

1. Easier option
Throw an error in init if `valinit[0] == valinit[1]`

2. Maybe better option?
Don't use axhspan and manually create the poly to ensure it always has the expected number of vertices
Option 2 might be better yes
@vpicouet any interest in opening a PR?
I don't think I am qualified to do so, never opened a PR yet. 
RangeSlider might also contain another issue. 
When I call `RangeSlider.set_val([0.0,0.1])`
It changes only the blue poly object and the range value on the right side of the slider not the dot:
![Capture d’écran, le 2022-03-25 à 15 53 44](https://user-images.githubusercontent.com/37241971/160191943-aef5fbe2-2f54-42ae-9719-23375767b212.jpg)
 
> I don't think I am qualified to do so, never opened a PR yet.

That's always true until you've opened your first one :). But I also understand that it can be intimidating.


>  RangeSlider might also contain another issue.
> When I call RangeSlider.set_val([0.0,0.1])
> It changes only the blue poly object and the range value on the right side of the slider not the dot:


oh hmm - good catch! may be worth opening a separate issue there as these are two distinct bugs and this one may be a bit more comlicated to fix.
Haha true! I might try when I have more time!
Throwing an error at least as I have never worked with axhspan and polys.
Ok, openning another issue.
Can I try working on this? @ianhi @vpicouet 
From the discussion, I could identify that a quick fix would be to use a try-except block to throw an error 
if valinit[0] == valinit[1]

Please let me know your thoughts.
Sure!
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/lib/matplotlib/widgets.py b/lib/matplotlib/widgets.py
index da5b40a..0c7c827 100755
--- a/lib/matplotlib/widgets.py
+++ b/lib/matplotlib/widgets.py
@@ -897,20 +897,15 @@ class RangeSlider(SliderBase):
         _api.check_shape((2,), val=val)
         val[0] = self._min_in_bounds(val[0])
         val[1] = self._max_in_bounds(val[1])
-        xy = self.poly.xy
         if self.orientation == "vertical":
-            xy[0] = .25, val[0]
-            xy[1] = .25, val[1]
-            xy[2] = .75, val[1]
-            xy[3] = .75, val[0]
-            xy[4] = .25, val[0]
+            xy = [(.25, val[0]), (.25, val[1]),
+                  (.75, val[1]), (.75, val[0])]
         else:
-            xy[0] = val[0], .25
-            xy[1] = val[0], .75
-            xy[2] = val[1], .75
-            xy[3] = val[1], .25
-            xy[4] = val[0], .25
-        self.poly.xy = xy
+            xy = [(val[0], .25), (val[0], .75),
+                  (val[1], .75), (val[1], .25)]
+        # Polygon.set_xy() takes care of adding the closing vertex, which is
+        # absent from the initial polygon when the range is degenerate.
+        self.poly.set_xy(xy)
         self.valtext.set_text(self._format(val))
         if self.drawon:
             self.ax.figure.canvas.draw_idle()
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/lib/matplotlib/tests/test_widgets.py b/lib/matplotlib/tests/test_widgets.py
index a19770e..72e23af 100755
--- a/lib/matplotlib/tests/test_widgets.py
+++ b/lib/matplotlib/tests/test_widgets.py
@@ -1120,6 +1120,19 @@ def test_range_slider(orientation):
     assert_allclose(slider.val, [0.1, 0.34])
 
 
+@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
+@pytest.mark.parametrize("valinit", [(0.0, 0.0), (1.0, 1.0)])
+def test_range_slider_initial_value_equal(orientation, valinit):
+    # A degenerate initial range produces a polygon without an explicit
+    # closing vertex; set_val() must not assume that vertex exists.
+    fig, ax = plt.subplots()
+    slider = widgets.RangeSlider(
+        ax=ax, label="", valmin=0.0, valmax=1.0, orientation=orientation,
+        valinit=valinit
+    )
+    assert_allclose(slider.val, valinit)
+
+
 def check_polygon_selector(event_sequence, expected_result, selections_count,
                            **kwargs):
     """
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/lib/matplotlib/widgets.py b/lib/matplotlib/widgets.py
--- a/lib/matplotlib/widgets.py
+++ b/lib/matplotlib/widgets.py
@@ -813,7 +813,10 @@ def _update_val_from_pos(self, pos):
             val = self._max_in_bounds(pos)
             self.set_max(val)
         if self._active_handle:
-            self._active_handle.set_xdata([val])
+            if self.orientation == "vertical":
+                self._active_handle.set_ydata([val])
+            else:
+                self._active_handle.set_xdata([val])
 
     def _update(self, event):
         """Update the slider position."""
@@ -836,11 +839,16 @@ def _update(self, event):
             return
 
         # determine which handle was grabbed
-        handle = self._handles[
-            np.argmin(
+        if self.orientation == "vertical":
+            handle_index = np.argmin(
+                np.abs([h.get_ydata()[0] - event.ydata for h in self._handles])
+            )
+        else:
+            handle_index = np.argmin(
                 np.abs([h.get_xdata()[0] - event.xdata for h in self._handles])
             )
-        ]
+        handle = self._handles[handle_index]
+
         # these checks ensure smooth behavior if the handles swap which one
         # has a higher value. i.e. if one is dragged over and past the other.
         if handle is not self._active_handle:
@@ -904,14 +912,22 @@ def set_val(self, val):
             xy[2] = .75, val[1]
             xy[3] = .75, val[0]
             xy[4] = .25, val[0]
+
+            self._handles[0].set_ydata([val[0]])
+            self._handles[1].set_ydata([val[1]])
         else:
             xy[0] = val[0], .25
             xy[1] = val[0], .75
             xy[2] = val[1], .75
             xy[3] = val[1], .25
             xy[4] = val[0], .25
+
+            self._handles[0].set_xdata([val[0]])
+            self._handles[1].set_xdata([val[1]])
+
         self.poly.xy = xy
         self.valtext.set_text(self._format(val))
+
         if self.drawon:
             self.ax.figure.canvas.draw_idle()
         self.val = val
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/lib/matplotlib/tests/test_widgets.py b/lib/matplotlib/tests/test_widgets.py
--- a/lib/matplotlib/tests/test_widgets.py
+++ b/lib/matplotlib/tests/test_widgets.py
@@ -1105,19 +1105,30 @@ def test_range_slider(orientation):
     # Check initial value is set correctly
     assert_allclose(slider.val, (0.1, 0.34))
 
+    def handle_positions(slider):
+        if orientation == "vertical":
+            return [h.get_ydata()[0] for h in slider._handles]
+        else:
+            return [h.get_xdata()[0] for h in slider._handles]
+
     slider.set_val((0.2, 0.6))
     assert_allclose(slider.val, (0.2, 0.6))
+    assert_allclose(handle_positions(slider), (0.2, 0.6))
+
     box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
     assert_allclose(box.get_points().flatten()[idx], [0.2, .25, 0.6, .75])
 
     slider.set_val((0.2, 0.1))
     assert_allclose(slider.val, (0.1, 0.2))
+    assert_allclose(handle_positions(slider), (0.1, 0.2))
 
     slider.set_val((-1, 10))
     assert_allclose(slider.val, (0, 1))
+    assert_allclose(handle_positions(slider), (0, 1))
 
     slider.reset()
-    assert_allclose(slider.val, [0.1, 0.34])
+    assert_allclose(slider.val, (0.1, 0.34))
+    assert_allclose(handle_positions(slider), (0.1, 0.34))
 
 
 def check_polygon_selector(event_sequence, expected_result, selections_count,
```

</details>

<details>
<summary>Verification command at step 7 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 47
Step: 7

COMMAND
python -m pytest lib/matplotlib/tests/test_widgets.py::test_range_slider -q

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 3.446
stdout:
..                                                                       [100%]
=============================== warnings summary ===============================
../opt/miniconda3/envs/testbed/lib/python3.11/site-packages/pyparsing.py:108
  /opt/miniconda3/envs/testbed/lib/python3.11/site-packages/pyparsing.py:108: DeprecationWarning: module 'sre_constants' is deprecated
    import sre_constants

-- Docs: https://docs.pytest.org/en/sta... [truncated]
```

</details>

<details>
<summary>Verification command at step 14 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 82
Step: 14

COMMAND
python -m pytest lib/matplotlib/tests/test_widgets.py::test_range_slider_initial_value_equal -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 1.503
stdout:
F.F.                                                                     [100%]
=================================== FAILURES ===================================
__________ test_range_slider_initial_value_equal[valinit0-horizontal] __________

orientation = 'horizontal', valinit = (0.0, 0.0)

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    @pytest.mark.parametrize("valinit", [(0.0, 0.0), (1.0, 1.0)])
    def... [truncated]
```

</details>

<details>
<summary>Verification command at step 17 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 100
Step: 17

COMMAND
python -m pytest lib/matplotlib/tests/test_widgets.py::test_range_slider_initial_value_equal lib/matplotlib/tests/test_widgets.py::test_range_slider lib/matplotlib/tests/test_widgets.py::test_slider_horizontal_vertical lib/matplotlib/tests/test_widgets.py::test_slider_reset -q

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 2.028
stdout:
........                                                                 [100%]
=============================== warnings summary ===============================
../opt/miniconda3/envs/testbed/lib/python3.11/site-packages/pyparsing.py:108
  /opt/miniconda3/envs/testbed/lib/python3.11/site-packages/pyparsing.py:108: DeprecationWarning: module 'sre_constants' is deprecated
    import sre_constants

-- Docs: https://docs.pytest.org/en/sta... [truncated]
```

</details>

<details>
<summary>Verification command at step 18 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 105
Step: 18

COMMAND
python -m pytest lib/matplotlib/tests/test_widgets.py -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 10.950
stdout:
F........................................................s.............. [ 72%]
...........s................                                             [100%]
=================================== FAILURES ===================================
___________________________ test_rectangle_selector ____________________________

    def test_rectangle_selector():
        check_rectangle()
    
>       with pytest.warns(
            Matplotli... [truncated]
```

</details>

<details>
<summary>Verification command at step 23 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 130
Step: 23

COMMAND
python -m pytest lib/matplotlib/tests/test_widgets.py -q -k "slider"

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.450
stdout:
............                                                             [100%]
=============================== warnings summary ===============================
../opt/miniconda3/envs/testbed/lib/python3.11/site-packages/pyparsing.py:108
  /opt/miniconda3/envs/testbed/lib/python3.11/site-packages/pyparsing.py:108: DeprecationWarning: module 'sre_constants' is deprecated
    import sre_constants

-- Docs: https://docs.pytest.org/en/sta... [truncated]
```

</details>


[Return to main report](../README.md)
