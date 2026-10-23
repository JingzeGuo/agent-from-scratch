# django__django-11905: The issue asks for an error; the accepted change is a deprecation warning

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P unavailable; P2P unavailable. **Steps:** 26/40. Counts are harness classifications, subject to the report's parser caveats.

The agent implements the literal issue request, while the golden test requires backward-compatible warning behavior.

## Issue requirement

The supplied problem statement says non-boolean __isnull values should raise an error.

## Agent change

Raises ValueError for a non-bool RHS, using the same error text discussed in the issue history.

## Golden change

Emits RemovedInDjango40Warning and retains the old behavior during deprecation. The dataset hints record the later decision to deprecate first, but those hints were not included in the agent objective.

## Observed failure

The patch applies cleanly. A golden assertWarnsMessage case instead receives ValueError. While formatting the failed subtest, unittest evaluates a QuerySet representation, raises the error again and aborts reporting. report.json consequently has no tests_status and misleadingly says patch_successfully_applied=false.

## Scope and attribution

This is the clearest task/acceptance contradiction in the run. The agent could have considered Django compatibility policy, but the specific deprecation decision was absent from the task it received.

## Verification in the trajectory

The agent wrote tests asserting ValueError and ran native lookup tests after discovering pytest was absent. Those tests validate its interpretation, not the golden warning contract.

## What would improve the outcome

Flag this task for benchmark contract review. In general, inspect version-specific deprecation conventions before replacing previously accepted input with an exception; do not expose gold patches to the solving agent.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 11 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 11 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 11 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-11905/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-11905/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/django__django-11905/run_instance.log) | — |
| trajectory | [django__django-11905.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/django__django-11905.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

```text
Prevent using __isnull lookup with non-boolean value.
Description
	 
		(last modified by Mariusz Felisiak)
	 
__isnull should not allow for non-boolean values. Using truthy/falsey doesn't promote INNER JOIN to an OUTER JOIN but works fine for a simple queries. Using non-boolean values is ​undocumented and untested. IMO we should raise an error for non-boolean values to avoid confusion and for consistency.
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
PR here: ​https://github.com/django/django/pull/11873
After the reconsideration I don't think that we should change this ​documented behavior (that is in Django from the very beginning). __isnull lookup expects boolean values in many places and IMO it would be confusing if we'll allow for truthy/falsy values, e.g. take a look at these examples field__isnull='false' or field__isnull='true' (both would return the same result). You can always call bool() on a right hand side. Sorry for my previous acceptation (I shouldn't triage tickets in the weekend).
Replying to felixxm: After the reconsideration I don't think that we should change this ​documented behavior (that is in Django from the very beginning). __isnull lookup expects boolean values in many places and IMO it would be confusing if we'll allow for truthy/falsy values, e.g. take a look at these examples field__isnull='false' or field__isnull='true' (both would return the same result). You can always call bool() on a right hand side. Sorry for my previous acceptation (I shouldn't triage tickets in the weekend). I understand your point. But is there anything we can do to avoid people falling for the same pitfall I did? The problem, in my opinion, is that it works fine for simple queries but as soon as you add a join that needs promotion it will break, silently. Maybe we should make it raise an exception when a non-boolean is passed? One valid example is to have a class that implements __bool__. You can see here ​https://github.com/django/django/blob/d9881a025c15d87b2a7883ee50771117450ea90d/django/db/models/lookups.py#L465-L470 that non-bool value is converted to IS NULL and IS NOT NULL already using the truthy/falsy values. IMO it would be confusing if we'll allow for truthy/falsy values, e.g. take a look at these examples fieldisnull='false' or fieldisnull='true' (both would return the same result). This is already the case. It just is inconsistent, in lookups.py field__isnull='false' will be a positive condition but on the query.py it will be the negative condition.
Maybe adding a note on the documentation? something like: "Although it might seem like it will work with non-bool fields, this is not supported and can lead to inconsistent behaviours"
Agreed, we should raise an error for non-boolean values, e.g. diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py index 9344979c56..fc4a38c4fe 100644 --- a/django/db/models/lookups.py +++ b/django/db/models/lookups.py @@ -463,6 +463,11 @@ class IsNull(BuiltinLookup): prepare_rhs = False def as_sql(self, compiler, connection): + if not isinstance(self.rhs, bool): + raise ValueError( + 'The QuerySet value for an isnull lookup must be True or ' + 'False.' + ) sql, params = compiler.compile(self.lhs) if self.rhs: return "%s IS NULL" % sql, params I changed the ticket description.
Thanks, I'll work on it! Wouldn't that possibly break backward compatibility? I'm not familiar with how Django moves in that regard.
We can add a release note in "Backwards incompatible changes" or deprecate this and remove in Django 4.0. I have to thing about it, please give me a day, maybe I will change my mind :)
No problem. Thanks for taking the time to look into this!
Another interesting example related to this: As an anecdote, I've also got bitten by this possibility. An attempt to write WHERE (field IS NULL) = boolean_field as .filter(field__isnull=F('boolean_field')) didn't go as I expected. Alexandr Aktsipetrov -- ​https://groups.google.com/forum/#!msg/django-developers/AhY2b3rxkfA/0sz3hNanCgAJ This example will generate the WHERE .... IS NULL. I guess we also would want an exception thrown here.
André, IMO we should deprecate using non-boolean values in Django 3.1 (RemovedInDjango40Warning) and remove in Django 4.0 (even if it is untested and undocumented). I can imagine that a lot of people use e.g. 1 and 0 instead of booleans. Attached diff fixes also issue with passing a F() expression. def as_sql(self, compiler, connection): if not isinstance(self.rhs, bool): raise RemovedInDjango40Warning(...) ....
Replying to felixxm: André, IMO we should deprecate using non-boolean values in Django 3.1 (RemovedInDjango40Warning) and remove in Django 4.0 (even if it is untested and undocumented). I can imagine that a lot of people use e.g. 1 and 0 instead of booleans. Attached diff fixes also issue with passing a F() expression. def as_sql(self, compiler, connection): if not isinstance(self.rhs, bool): raise RemovedInDjango40Warning(...) .... Sound like a good plan. Not super familiar with the branch structure of Django. So, I guess the path to follow is to make a PR to master adding the deprecation warning and eventually when master is 4.x we create the PR raising the ValueError. Is that right? Thanks!
André, yes mostly. You can find more details about that ​from the documentation.
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py
index 9344979..fc4a38c 100755
--- a/django/db/models/lookups.py
+++ b/django/db/models/lookups.py
@@ -463,6 +463,11 @@ class IsNull(BuiltinLookup):
     prepare_rhs = False
 
     def as_sql(self, compiler, connection):
+        if not isinstance(self.rhs, bool):
+            raise ValueError(
+                'The QuerySet value for an isnull lookup must be True or '
+                'False.'
+            )
         sql, params = compiler.compile(self.lhs)
         if self.rhs:
             return "%s IS NULL" % sql, params
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
index 1958b99..172108c 100755
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -880,6 +880,13 @@ class LookupTests(TestCase):
         self.assertTrue(Season.objects.filter(nulled_text_field__nulled__exact=None))
         self.assertTrue(Season.objects.filter(nulled_text_field__nulled=None))
 
+    def test_isnull_non_boolean_value(self):
+        msg = 'The QuerySet value for an isnull lookup must be True or False.'
+        with self.assertRaisesMessage(ValueError, msg):
+            Author.objects.filter(alias__isnull=1).exists()
+        with self.assertRaisesMessage(ValueError, msg):
+            Author.objects.filter(alias__isnull='').exists()
+
     def test_exact_sliced_queryset_limit_one(self):
         self.assertCountEqual(
             Article.objects.filter(author=Author.objects.all()[:1]),
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py
--- a/django/db/models/lookups.py
+++ b/django/db/models/lookups.py
@@ -1,5 +1,6 @@
 import itertools
 import math
+import warnings
 from copy import copy
 
 from django.core.exceptions import EmptyResultSet
@@ -9,6 +10,7 @@
 )
 from django.db.models.query_utils import RegisterLookupMixin
 from django.utils.datastructures import OrderedSet
+from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.functional import cached_property
 
 
@@ -463,6 +465,17 @@ class IsNull(BuiltinLookup):
     prepare_rhs = False
 
     def as_sql(self, compiler, connection):
+        if not isinstance(self.rhs, bool):
+            # When the deprecation ends, replace with:
+            # raise ValueError(
+            #     'The QuerySet value for an isnull lookup must be True or '
+            #     'False.'
+            # )
+            warnings.warn(
+                'Using a non-boolean value for an isnull lookup is '
+                'deprecated, use True or False instead.',
+                RemovedInDjango40Warning,
+            )
         sql, params = compiler.compile(self.lhs)
         if self.rhs:
             return "%s IS NULL" % sql, params
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/lookup/models.py b/tests/lookup/models.py
--- a/tests/lookup/models.py
+++ b/tests/lookup/models.py
@@ -96,3 +96,15 @@ class Product(models.Model):
 class Stock(models.Model):
     product = models.ForeignKey(Product, models.CASCADE)
     qty_available = models.DecimalField(max_digits=6, decimal_places=2)
+
+
+class Freebie(models.Model):
+    gift_product = models.ForeignKey(Product, models.CASCADE)
+    stock_id = models.IntegerField(blank=True, null=True)
+
+    stock = models.ForeignObject(
+        Stock,
+        from_fields=['stock_id', 'gift_product'],
+        to_fields=['id', 'product'],
+        on_delete=models.CASCADE,
+    )
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -9,9 +9,10 @@
 from django.db.models.expressions import Exists, OuterRef
 from django.db.models.functions import Substr
 from django.test import TestCase, skipUnlessDBFeature
+from django.utils.deprecation import RemovedInDjango40Warning
 
 from .models import (
-    Article, Author, Game, IsNullWithNoneAsRHS, Player, Season, Tag,
+    Article, Author, Freebie, Game, IsNullWithNoneAsRHS, Player, Season, Tag,
 )
 
 
@@ -969,3 +970,24 @@ def test_exact_query_rhs_with_selected_columns(self):
         ).values('max_id')
         authors = Author.objects.filter(id=authors_max_ids[:1])
         self.assertEqual(authors.get(), newest_author)
+
+    def test_isnull_non_boolean_value(self):
+        # These tests will catch ValueError in Django 4.0 when using
+        # non-boolean values for an isnull lookup becomes forbidden.
+        # msg = (
+        #     'The QuerySet value for an isnull lookup must be True or False.'
+        # )
+        msg = (
+            'Using a non-boolean value for an isnull lookup is deprecated, '
+            'use True or False instead.'
+        )
+        tests = [
+            Author.objects.filter(alias__isnull=1),
+            Article.objects.filter(author__isnull=1),
+            Season.objects.filter(games__isnull=1),
+            Freebie.objects.filter(stock__isnull=1),
+        ]
+        for qs in tests:
+            with self.subTest(qs=qs):
+                with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+                    qs.exists()
```

</details>

<details>
<summary>Verification command at step 22 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 164
Step: 22

COMMAND
python -m pytest tests/lookup/tests.py -k "isnull_non_boolean or exact_none_transform" -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.347
stdout:
/opt/miniconda3/envs/testbed/bin/python: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 23 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 169
Step: 23

COMMAND
python tests/runtests.py lookup -v 2 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.418
stdout:
test_in_different_database (lookup.tests.LookupTests) ... ok
test_in_keeps_value_ordering (lookup.tests.LookupTests) ... ok
test_isnull_non_boolean_value (lookup.tests.LookupTests) ... ok
test_iterator (lookup.tests.LookupTests) ... ok
test_lookup_collision (lookup.tests.LookupTests) ... ok
test_lookup_date_as_str (lookup.tests.LookupTests) ... ok
test_lookup_int_as_str (lookup.tests.LookupTests) ... ok
test_nested_outerref_lhs (looku... [truncated]
```

</details>

<details>
<summary>Verification command at step 24 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 174
Step: 24

COMMAND
python tests/runtests.py queries foreign_object admin_filters lookup -v 1 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.543
stdout:
Cloning test database for alias 'default'...
Cloning test database for alias 'default'...
sssss........s...........sss......s.........x...........x.................................................s...........s.....................................................s.................................................................................................................................................................................. [truncated]
```

</details>


[Return to main report](../README.md)
