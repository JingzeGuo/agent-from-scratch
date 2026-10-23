# matplotlib__matplotlib-18869: An open-ended version feature is graded against one private parser and tuple format

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P 0/4; P2P 3/3. **Steps:** 17/40. Counts are harness classifications, subject to the report's parser caveats.

The agent exposes a comparable release tuple; all four golden failures call an unspecified helper.

## Issue requirement

Expose version information that is easier to compare; a tuple or LooseVersion is explicitly acceptable in the discussion.

## Agent change

Adds lazy __version_info__ using parse_version(...).release, a tuple of numeric release components.

## Golden change

Defines a five-field _VersionInfo, a private _parse_to_version_info helper, prerelease labels and special post-release handling.

## Observed failure

All four F2P cases raise AttributeError because matplotlib._parse_to_version_info does not exist. The three P2P tests pass.

## Scope and attribution

The helper name and five-field schema are unspecified implementation choices. Separately, a release-only tuple loses prerelease distinctions, which is a substantive limitation for general version comparison.

## Verification in the trajectory

The agent verified a public tuple and ran its modified test_matplotlib module. Its printed runtime version tuple was (0, 0) for a 0.0.post... version, showing it did not adopt a fixed five-field shape.

## What would improve the outcome

Review acceptance tests for public-behavior coverage. For the agent, explicitly test final, prerelease, development and post-release versions before declaring a comparison API complete.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 42 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 42 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 42 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/matplotlib__matplotlib-18869/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/matplotlib__matplotlib-18869/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/matplotlib__matplotlib-18869/run_instance.log) | — |
| trajectory | [matplotlib__matplotlib-18869.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/matplotlib__matplotlib-18869.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

```text
Add easily comparable version info to toplevel
<!--
Welcome! Thanks for thinking of a way to improve Matplotlib.


Before creating a new feature request please search the issues for relevant feature requests.
-->

### Problem

Currently matplotlib only exposes `__version__`.  For quick version checks, exposing either a `version_info` tuple (which can be compared with other tuples) or a `LooseVersion` instance (which can be properly compared with other strings) would be a small usability improvement.

(In practice I guess boring string comparisons will work just fine until we hit mpl 3.10 or 4.10 which is unlikely to happen soon, but that feels quite dirty :))
<!--
Provide a clear and concise description of the problem this feature will solve. 

For example:
* I'm always frustrated when [...] because [...]
* I would like it if [...] happened when I [...] because [...]
* Here is a sample image of what I am asking for [...]
-->

### Proposed Solution

I guess I slightly prefer `LooseVersion`, but exposing just a `version_info` tuple is much more common in other packages (and perhaps simpler to understand).  The hardest(?) part is probably just bikeshedding this point :-)
<!-- Provide a clear and concise description of a way to accomplish what you want. For example:

* Add an option so that when [...]  [...] will happen
 -->

### Additional context and prior art

`version_info` is a pretty common thing (citation needed).
<!-- Add any other context or screenshots about the feature request here. You can also include links to examples of other programs that have something similar to your request. For example:

* Another project [...] solved this by [...]
-->
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
It seems that `__version_info__` is the way to go.

### Prior art
- There's no official specification for version tuples. [PEP 396 - Module Version Numbers](https://www.python.org/dev/peps/pep-0396/) only defines the string `__version__`.

- Many projects don't bother with version tuples.

- When they do, `__version_info__` seems to be a common thing:
  - [Stackoverflow discussion](https://stackoverflow.com/a/466694)
  - [PySide2](https://doc.qt.io/qtforpython-5.12/pysideversion.html#printing-project-and-qt-version) uses it.

- Python itself has the string [sys.version](https://docs.python.org/3/library/sys.html#sys.version) and the (named)tuple [sys.version_info](https://docs.python.org/3/library/sys.html#sys.version_info). In analogy to that `__version_info__` next to `__version__` makes sense for packages.
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/lib/matplotlib/__init__.py b/lib/matplotlib/__init__.py
index b657a35..80839aa 100755
--- a/lib/matplotlib/__init__.py
+++ b/lib/matplotlib/__init__.py
@@ -130,24 +130,32 @@ __bibtex__ = r"""@Article{Hunter:2007,
 }"""
 
 
+def _get_version():
+    import setuptools_scm
+    global __version__  # cache it.
+    # Only shell out to a git subprocess if really needed, and not on a
+    # shallow clone, such as those used by CI, as the latter would trigger
+    # a warning from setuptools_scm.
+    root = Path(__file__).resolve().parents[2]
+    if (root / ".git").exists() and not (root / ".git/shallow").exists():
+        __version__ = setuptools_scm.get_version(
+            root=root,
+            version_scheme="post-release",
+            local_scheme="node-and-date",
+            fallback_version=_version.version,
+        )
+    else:  # Get the version from the _version.py setuptools_scm file.
+        __version__ = _version.version
+    return __version__
+
+
 def __getattr__(name):
     if name == "__version__":
-        import setuptools_scm
-        global __version__  # cache it.
-        # Only shell out to a git subprocess if really needed, and not on a
-        # shallow clone, such as those used by CI, as the latter would trigger
-        # a warning from setuptools_scm.
-        root = Path(__file__).resolve().parents[2]
-        if (root / ".git").exists() and not (root / ".git/shallow").exists():
-            __version__ = setuptools_scm.get_version(
-                root=root,
-                version_scheme="post-release",
-                local_scheme="node-and-date",
-                fallback_version=_version.version,
-            )
-        else:  # Get the version from the _version.py setuptools_scm file.
-            __version__ = _version.version
-        return __version__
+        return _get_version()
+    if name == "__version_info__":
+        global __version_info__  # cache it.
+        __version_info__ = parse_version(_get_version()).release
+        return __version_info__
     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/lib/matplotlib/tests/test_matplotlib.py b/lib/matplotlib/tests/test_matplotlib.py
index cad9433..e509fe6 100755
--- a/lib/matplotlib/tests/test_matplotlib.py
+++ b/lib/matplotlib/tests/test_matplotlib.py
@@ -3,10 +3,16 @@ import subprocess
 import sys
 
 import pytest
+from packaging.version import parse as parse_version
 
 import matplotlib
 
 
+def test_version_info():
+    assert matplotlib.__version_info__ == parse_version(
+        matplotlib.__version__).release
+
+
 @pytest.mark.skipif(
     os.name == "nt", reason="chmod() doesn't work as is on Windows")
 @pytest.mark.skipif(os.name != "nt" and os.geteuid() == 0,
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/lib/matplotlib/__init__.py b/lib/matplotlib/__init__.py
--- a/lib/matplotlib/__init__.py
+++ b/lib/matplotlib/__init__.py
@@ -129,25 +129,60 @@
   year      = 2007
 }"""
 
+# modelled after sys.version_info
+_VersionInfo = namedtuple('_VersionInfo',
+                          'major, minor, micro, releaselevel, serial')
 
-def __getattr__(name):
-    if name == "__version__":
+
+def _parse_to_version_info(version_str):
+    """
+    Parse a version string to a namedtuple analogous to sys.version_info.
+
+    See:
+    https://packaging.pypa.io/en/latest/version.html#packaging.version.parse
+    https://docs.python.org/3/library/sys.html#sys.version_info
+    """
+    v = parse_version(version_str)
+    if v.pre is None and v.post is None and v.dev is None:
+        return _VersionInfo(v.major, v.minor, v.micro, 'final', 0)
+    elif v.dev is not None:
+        return _VersionInfo(v.major, v.minor, v.micro, 'alpha', v.dev)
+    elif v.pre is not None:
+        releaselevel = {
+            'a': 'alpha',
+            'b': 'beta',
+            'rc': 'candidate'}.get(v.pre[0], 'alpha')
+        return _VersionInfo(v.major, v.minor, v.micro, releaselevel, v.pre[1])
+    else:
+        # fallback for v.post: guess-next-dev scheme from setuptools_scm
+        return _VersionInfo(v.major, v.minor, v.micro + 1, 'alpha', v.post)
+
+
+def _get_version():
+    """Return the version string used for __version__."""
+    # Only shell out to a git subprocess if really needed, and not on a
+    # shallow clone, such as those used by CI, as the latter would trigger
+    # a warning from setuptools_scm.
+    root = Path(__file__).resolve().parents[2]
+    if (root / ".git").exists() and not (root / ".git/shallow").exists():
         import setuptools_scm
+        return setuptools_scm.get_version(
+            root=root,
+            version_scheme="post-release",
+            local_scheme="node-and-date",
+            fallback_version=_version.version,
+        )
+    else:  # Get the version from the _version.py setuptools_scm file.
+        return _version.version
+
+
+def __getattr__(name):
+    if name in ("__version__", "__version_info__"):
         global __version__  # cache it.
-        # Only shell out to a git subprocess if really needed, and not on a
-        # shallow clone, such as those used by CI, as the latter would trigger
-        # a warning from setuptools_scm.
-        root = Path(__file__).resolve().parents[2]
-        if (root / ".git").exists() and not (root / ".git/shallow").exists():
-            __version__ = setuptools_scm.get_version(
-                root=root,
-                version_scheme="post-release",
-                local_scheme="node-and-date",
-                fallback_version=_version.version,
-            )
-        else:  # Get the version from the _version.py setuptools_scm file.
-            __version__ = _version.version
-        return __version__
+        __version__ = _get_version()
+        global __version__info__  # cache it.
+        __version_info__ = _parse_to_version_info(__version__)
+        return __version__ if name == "__version__" else __version_info__
     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/lib/matplotlib/tests/test_matplotlib.py b/lib/matplotlib/tests/test_matplotlib.py
--- a/lib/matplotlib/tests/test_matplotlib.py
+++ b/lib/matplotlib/tests/test_matplotlib.py
@@ -7,6 +7,16 @@
 import matplotlib
 
 
+@pytest.mark.parametrize('version_str, version_tuple', [
+    ('3.5.0', (3, 5, 0, 'final', 0)),
+    ('3.5.0rc2', (3, 5, 0, 'candidate', 2)),
+    ('3.5.0.dev820+g6768ef8c4c', (3, 5, 0, 'alpha', 820)),
+    ('3.5.0.post820+g6768ef8c4c', (3, 5, 1, 'alpha', 820)),
+])
+def test_parse_to_version_info(version_str, version_tuple):
+    assert matplotlib._parse_to_version_info(version_str) == version_tuple
+
+
 @pytest.mark.skipif(
     os.name == "nt", reason="chmod() doesn't work as is on Windows")
 @pytest.mark.skipif(os.name != "nt" and os.geteuid() == 0,
```

</details>

<details>
<summary>Verification command at step 8 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 60
Step: 8

COMMAND
python3 -m pytest lib/matplotlib/tests/test_matplotlib.py::test_version_info -q 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 3.409
stdout:
    def __getattr__(name):
        if name == "__version__":
            import setuptools_scm
            global __version__  # cache it.
            # Only shell out to a git subprocess if really needed, and not on a
            # shallow clone, such as those used by CI, as the latter would trigger
            # a warning from setuptools_scm.
            root = Path(__file__).resolve().parents[2]
            if (root / ".git").exist... [truncated]
```

</details>

<details>
<summary>Verification command at step 14 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 100
Step: 14

COMMAND
python3 -m pytest lib/matplotlib/tests/test_matplotlib.py -q 2>&1 | tail -15

STORED OUTPUT PREVIEW
Validation error for tool 'run_command':
  - field 'timeout_seconds': Input should be less than or equal to 120
```

</details>

<details>
<summary>Verification command at step 15 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 107
Step: 15

COMMAND
python3 -m pytest lib/matplotlib/tests/test_matplotlib.py -q 2>&1 | tail -15

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 4.684
stdout:
.s...                                                                    [100%]
4 passed, 1 skipped in 3.90s
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
