# django__django-14155: Follows the issue’s unwrap suggestion; gold tests require a different repr

**Primary category:** S — Unclear or conflicting acceptance contract (reclassified from I on review). **Confidence:** high.

**Recorded score:** F2P 0/3; P2P 89/89. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The agent unwraps the callable and merges bound arguments into ResolverMatch instead of limiting the change to display.

## Issue requirement

Show the underlying function and bound arguments in ResolverMatch.__repr__; the issue suggests unwrapping partials in __init__.

## Agent change

Replaces self.func with the underlying function and merges partial arguments into match args/kwargs. __repr__ itself remains unchanged.

## Golden change

Preserves the stored callable and arguments, uses repr(self.func) for a partial, and changes several display fields from %s to %r.

## Observed failure

The existing repr expectation fails on string quoting, and partial/nested/wrapped cases fail because they expect functools.partial(...) and unchanged match kwargs. The extra report entry test_resolver_match_on_request is on a merged raw-log line ending in ok, so it is not independently established as a failing assertion.

## Scope and attribution

The agent implements the issue's explicit suggestion (unwrap partials in `__init__`). Every failing assertion checks a repr format the issue never specifies: `%r` quoting of all fields and keeping the partial object itself. The third F2P name is a parser artifact. Changing stored state for a display problem is a secondary design concern, not the decisive failure.

## Verification in the trajectory

New tests assert the unwrapping design. At steps 30–38 the agent iterates until those tests pass, reaching the 40-step ceiling without validating a representation-only solution.

## What would improve the outcome

Separate representation from execution state. Preserve callable identity unless the behavior contract explicitly requires otherwise, and test partial, nested and wrapped callables.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 28 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 28 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 28 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-14155/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-14155/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-14155/run_instance.log) | — |
| trajectory | [django__django-14155.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/django__django-14155.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

```text
ResolverMatch.__repr__() doesn't handle functools.partial() nicely.
Description
	 
		(last modified by Nick Pope)
	 
When a partial function is passed as the view, the __repr__ shows the func argument as functools.partial which isn't very helpful, especially as it doesn't reveal the underlying function or arguments provided.
Because a partial function also has arguments provided up front, we need to handle those specially so that they are accessible in __repr__.
ISTM that we can simply unwrap functools.partial objects in ResolverMatch.__init__().
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
index 9b00e24..d2f6b34 100755
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -31,6 +31,11 @@ from .utils import get_callable
 
 class ResolverMatch:
     def __init__(self, func, args, kwargs, url_name=None, app_names=None, namespaces=None, route=None, tried=None):
+        if isinstance(func, functools.partial):
+            # Unwrap partial functions and include their arguments in the match.
+            args = func.args + args
+            kwargs = {**func.keywords, **kwargs}
+            func = func.func
         self.func = func
         self.args = args
         self.kwargs = kwargs
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/urlpatterns_reverse/tests.py b/tests/urlpatterns_reverse/tests.py
index 09028be..6d3948a 100755
--- a/tests/urlpatterns_reverse/tests.py
+++ b/tests/urlpatterns_reverse/tests.py
@@ -1145,6 +1145,23 @@ class ResolverMatchTests(SimpleTestCase):
             "namespaces=[], route=^no_kwargs/([0-9]+)/([0-9]+)/$)",
         )
 
+    def test_repr_partial(self):
+        self.assertEqual(
+            repr(resolve('/partial/', urlconf='urlpatterns_reverse.urls')),
+            "ResolverMatch(func=urlpatterns_reverse.views.empty_view, "
+            "args=(), kwargs={'template_name': 'template.html'}, url_name=partial, "
+            "app_names=[], namespaces=[], route=partial/)",
+        )
+
+    def test_repr_nested_partial(self):
+        self.assertEqual(
+            repr(resolve('/partial_nested/', urlconf='urlpatterns_reverse.urls')),
+            "ResolverMatch(func=urlpatterns_reverse.views.empty_view, "
+            "args=(), kwargs={'template_name': 'nested_partial.html'}, "
+            "url_name=partial_nested, app_names=[], namespaces=[], "
+            "route=partial_nested/)",
+        )
+
 
 @override_settings(ROOT_URLCONF='urlpatterns_reverse.erroneous_urls')
 class ErroneousViewTests(SimpleTestCase):
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/django/urls/resolvers.py b/django/urls/resolvers.py
--- a/django/urls/resolvers.py
+++ b/django/urls/resolvers.py
@@ -59,9 +59,16 @@ def __getitem__(self, index):
         return (self.func, self.args, self.kwargs)[index]
 
     def __repr__(self):
-        return "ResolverMatch(func=%s, args=%s, kwargs=%s, url_name=%s, app_names=%s, namespaces=%s, route=%s)" % (
-            self._func_path, self.args, self.kwargs, self.url_name,
-            self.app_names, self.namespaces, self.route,
+        if isinstance(self.func, functools.partial):
+            func = repr(self.func)
+        else:
+            func = self._func_path
+        return (
+            'ResolverMatch(func=%s, args=%r, kwargs=%r, url_name=%r, '
+            'app_names=%r, namespaces=%r, route=%r)' % (
+                func, self.args, self.kwargs, self.url_name,
+                self.app_names, self.namespaces, self.route,
+            )
         )
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/urlpatterns_reverse/tests.py b/tests/urlpatterns_reverse/tests.py
--- a/tests/urlpatterns_reverse/tests.py
+++ b/tests/urlpatterns_reverse/tests.py
@@ -1141,10 +1141,30 @@ def test_repr(self):
         self.assertEqual(
             repr(resolve('/no_kwargs/42/37/')),
             "ResolverMatch(func=urlpatterns_reverse.views.empty_view, "
-            "args=('42', '37'), kwargs={}, url_name=no-kwargs, app_names=[], "
-            "namespaces=[], route=^no_kwargs/([0-9]+)/([0-9]+)/$)",
+            "args=('42', '37'), kwargs={}, url_name='no-kwargs', app_names=[], "
+            "namespaces=[], route='^no_kwargs/([0-9]+)/([0-9]+)/$')",
         )
 
+    @override_settings(ROOT_URLCONF='urlpatterns_reverse.urls')
+    def test_repr_functools_partial(self):
+        tests = [
+            ('partial', 'template.html'),
+            ('partial_nested', 'nested_partial.html'),
+            ('partial_wrapped', 'template.html'),
+        ]
+        for name, template_name in tests:
+            with self.subTest(name=name):
+                func = (
+                    f"functools.partial({views.empty_view!r}, "
+                    f"template_name='{template_name}')"
+                )
+                self.assertEqual(
+                    repr(resolve(f'/{name}/')),
+                    f"ResolverMatch(func={func}, args=(), kwargs={{}}, "
+                    f"url_name='{name}', app_names=[], namespaces=[], "
+                    f"route='{name}/')",
+                )
+
 
 @override_settings(ROOT_URLCONF='urlpatterns_reverse.erroneous_urls')
 class ErroneousViewTests(SimpleTestCase):
```

</details>

<details>
<summary>Verification command at step 27 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 190
Step: 27

COMMAND
python tests/runtests.py urlpatterns_reverse.tests.ResolverMatchTests.test_repr -v 2

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.951
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
Importing application urlpatterns_reverse
Skipping setup of unused database(s): default, other.
System check identified no issues (0 silenced).
stderr:
test_repr (urlpatterns_reverse.tests.ResolverMatchTests) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.006s

OK
```

</details>

<details>
<summary>Verification command at step 30 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 205
Step: 30

COMMAND
python tests/runtests.py urlpatterns_reverse.tests.ResolverMatchTests.test_repr_partial -v 2

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.700
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
Importing application urlpatterns_reverse
Skipping setup of unused database(s): default, other.
System check identified no issues (0 silenced).
stderr:
test_repr_partial (urlpatterns_reverse.tests.ResolverMatchTests) ... ERROR

======================================================================
ERROR: test_repr_partial (urlpatterns_reverse.tests.ResolverM... [truncated]
```

</details>

<details>
<summary>Verification command at step 35 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 233
Step: 35

COMMAND
python tests/runtests.py urlpatterns_reverse.tests.ResolverMatchTests -v 2

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.710
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
Importing application urlpatterns_reverse
Skipping setup of unused database(s): default, other.
System check identified no issues (0 silenced).
stderr:
test_repr (urlpatterns_reverse.tests.ResolverMatchTests) ... ok
test_repr_nested_partial (urlpatterns_reverse.tests.ResolverMatchTests) ... ERROR
test_repr_partial (urlpatterns_reverse.tests.ResolverMatchTest... [truncated]
```

</details>

<details>
<summary>Verification command at step 38 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 248
Step: 38

COMMAND
python tests/runtests.py urlpatterns_reverse.tests.ResolverMatchTests -v 1

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.678
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
System check identified no issues (0 silenced).
stderr:
......
----------------------------------------------------------------------
Ran 6 tests in 0.014s

OK
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 253
Step: 39

COMMAND
python tests/runtests.py urlpatterns_reverse -v 1

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.162
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
System check identified no issues (0 silenced).
stderr:
Creating test database for alias 'default'...
Cloning test database for alias 'default'...
Cloning test database for alias 'default'...
Cloning test database for alias 'default'...
Cloning test database for alias 'default'...
Cloning test database for alias 'default'...
Cloning test database for alias '... [truncated]
```

</details>


[Return to main report](../README.md)
