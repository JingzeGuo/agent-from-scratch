# django__django-12308: Valid JSON works, but unserializable dictionaries still crash

**Primary category:** E — Missing edge case or input/operation family. **Confidence:** medium (the agent followed the issue’s suggested `prepare_value` approach).

**Recorded score:** F2P 0/2; P2P 20/20. **Steps:** 23/40. Counts are harness classifications, subject to the report's parser caveats.

Delegating to the form field misses the invalid raw-value fallback.

## Issue requirement

Render readonly JSONField values as JSON rather than Python dictionaries. The suggested implementation mentions prepare_value and InvalidJSONInput.

## Agent change

Calls field.formfield().prepare_value(value) for JSONField values without catching serialization errors.

## Golden change

Uses field.get_prep_value for nonempty values and falls back to display_for_value on TypeError.

## Observed failure

A dictionary with a tuple key, {("a", "b"): "c"}, reaches json.dumps and raises TypeError: keys must be a string. The report also lists test_label_for_field as failed, but the raw log shows that test ending in ok on a merged output line; the tuple-key error is the supported causal evidence.

## Scope and attribution

Handling an InvalidJSONInput wrapper does not cover an invalid Python dictionary. This is a defensible invalid-input boundary case, with misleading implementation guidance as a contributor.

## Verification in the trajectory

The added test covers ordinary dictionaries, lists and strings. A focused native suite passes, but the recorded tests do not cover the tuple-key fallback.

## What would improve the outcome

Test valid JSON, a wrapper value, an unserializable raw object, empty values and a custom encoder separately; preserve the existing display fallback.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 16 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 16 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 16 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-12308/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-12308/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-12308/run_instance.log) | — |
| trajectory | [django__django-12308.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/django__django-12308.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

```text
JSONField are not properly displayed in admin when they are readonly.
Description
	
JSONField values are displayed as dict when readonly in the admin.
For example, {"foo": "bar"} would be displayed as {'foo': 'bar'}, which is not valid JSON.
I believe the fix would be to add a special case in django.contrib.admin.utils.display_for_field to call the prepare_value of the JSONField (not calling json.dumps directly to take care of the InvalidJSONInput case).
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
​PR
The proposed patch is problematic as the first version coupled contrib.postgres with .admin and the current one is based off the type name which is brittle and doesn't account for inheritance. It might be worth waiting for #12990 to land before proceeding here as the patch will be able to simply rely of django.db.models.JSONField instance checks from that point.
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/django/contrib/admin/utils.py b/django/contrib/admin/utils.py
index e4ddc8f..40f31dc 100755
--- a/django/contrib/admin/utils.py
+++ b/django/contrib/admin/utils.py
@@ -398,6 +398,10 @@ def display_for_field(value, field, empty_value_display):
         return formats.number_format(value)
     elif isinstance(field, models.FileField) and value:
         return format_html('<a href="{}">{}</a>', value.url, value)
+    elif isinstance(field, models.JSONField):
+        # Use the form field's prepare_value() to handle InvalidJSONInput and
+        # custom encoders.
+        return field.formfield().prepare_value(value)
     else:
         return display_for_value(value, empty_value_display)
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/admin_utils/tests.py b/tests/admin_utils/tests.py
index 463ba95..009a83b 100755
--- a/tests/admin_utils/tests.py
+++ b/tests/admin_utils/tests.py
@@ -197,6 +197,19 @@ class UtilsTests(SimpleTestCase):
         display_value = display_for_field(12345, models.IntegerField(), self.empty_value)
         self.assertEqual(display_value, '12,345')
 
+    def test_json_display_for_field(self):
+        tests = [
+            ({'a': {'b': 'c'}}, '{"a": {"b": "c"}}'),
+            (['a', 'b'], '["a", "b"]'),
+            ('a', '"a"'),
+        ]
+        for value, display_value in tests:
+            with self.subTest(value=value):
+                self.assertEqual(
+                    display_for_field(value, models.JSONField(), self.empty_value),
+                    display_value,
+                )
+
     def test_list_display_for_value(self):
         display_value = display_for_value([1, 2, 3], self.empty_value)
         self.assertEqual(display_value, '1, 2, 3')
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/django/contrib/admin/utils.py b/django/contrib/admin/utils.py
--- a/django/contrib/admin/utils.py
+++ b/django/contrib/admin/utils.py
@@ -398,6 +398,11 @@ def display_for_field(value, field, empty_value_display):
         return formats.number_format(value)
     elif isinstance(field, models.FileField) and value:
         return format_html('<a href="{}">{}</a>', value.url, value)
+    elif isinstance(field, models.JSONField) and value:
+        try:
+            return field.get_prep_value(value)
+        except TypeError:
+            return display_for_value(value, empty_value_display)
     else:
         return display_for_value(value, empty_value_display)
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/admin_utils/tests.py b/tests/admin_utils/tests.py
--- a/tests/admin_utils/tests.py
+++ b/tests/admin_utils/tests.py
@@ -176,6 +176,23 @@ def test_null_display_for_field(self):
         display_value = display_for_field(None, models.FloatField(), self.empty_value)
         self.assertEqual(display_value, self.empty_value)
 
+        display_value = display_for_field(None, models.JSONField(), self.empty_value)
+        self.assertEqual(display_value, self.empty_value)
+
+    def test_json_display_for_field(self):
+        tests = [
+            ({'a': {'b': 'c'}}, '{"a": {"b": "c"}}'),
+            (['a', 'b'], '["a", "b"]'),
+            ('a', '"a"'),
+            ({('a', 'b'): 'c'}, "{('a', 'b'): 'c'}"),  # Invalid JSON.
+        ]
+        for value, display_value in tests:
+            with self.subTest(value=value):
+                self.assertEqual(
+                    display_for_field(value, models.JSONField(), self.empty_value),
+                    display_value,
+                )
+
     def test_number_formats_display_for_field(self):
         display_value = display_for_field(12345.6789, models.FloatField(), self.empty_value)
         self.assertEqual(display_value, '12345.6789')
```

</details>

<details>
<summary>Verification command at step 15 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 111
Step: 15

COMMAND
python -m pytest tests/admin_utils/tests.py::UtilsTests::test_json_display_for_field -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.373
stdout:
[empty]
stderr:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
```

</details>

<details>
<summary>Verification command at step 16 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 116
Step: 16

COMMAND
python tests/runtests.py admin_utils.tests.UtilsTests.test_json_display_for_field -v 2

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 1.128
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
Importing application admin_utils
Skipping setup of unused database(s): default, other.
System check identified no issues (0 silenced).
stderr:
test_json_display_for_field (admin_utils.tests.UtilsTests) ... 
======================================================================
FAIL: test_json_display_for_field (admin_utils.tests.UtilsTests) (value={'a': {'b... [truncated]
```

</details>

<details>
<summary>Verification command at step 18 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 126
Step: 18

COMMAND
python tests/runtests.py admin_utils.tests.UtilsTests -v 1

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.772
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
System check identified no issues (0 silenced).
stderr:
...............
----------------------------------------------------------------------
Ran 15 tests in 0.034s

OK
```

</details>


[Return to main report](../README.md)
