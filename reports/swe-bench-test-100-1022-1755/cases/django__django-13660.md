# django__django-13660: The proposed empty exec namespace conflicts with the accepted globals contract

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P 2/2; P2P 4/6. **Steps:** 13/40. Counts are harness classifications, subject to the report's parser caveats.

The reported function/import crash is fixed, but golden tests require __name__ in the execution globals.

## Issue requirement

The issue explicitly suggests passing a new empty dictionary as the second argument to exec.

## Agent change

Passes {} to exec for both -c commands and stdin.

## Golden change

Passes globals() instead, preserving module globals including __name__.

## Observed failure

Both F2P function-call cases pass. Two P2P cases print False instead of True for "__name__" in globals().

## Scope and attribution

This combines a real namespace compatibility difference with contradictory issue guidance. Calling it only a careless regression would omit the fact that the agent followed the suggested fix exactly.

## Verification in the trajectory

A native shell suite and a new inline-function test were run; no test asserted namespace contents. Earlier piped test failures also surfaced as exit_code 0.

## What would improve the outcome

Treat issue-proposed code as a hypothesis. Verify both nested-name resolution and the public execution namespace before changing exec context.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `django__django-13660`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 25 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 25 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `django__django-13660.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

```text
shell command crashes when passing (with -c) the python code with functions.
Description
	
The examples below use Python 3.7 and Django 2.2.16, but I checked that the code is the same on master and works the same in Python 3.8.
Here's how ​python -c works:
$ python -c <<EOF " 
import django
def f():
		print(django.__version__)
f()"
EOF
2.2.16
Here's how ​python -m django shell -c works (paths shortened for clarify):
$ python -m django shell -c <<EOF "
import django
def f():
		print(django.__version__)
f()"
EOF
Traceback (most recent call last):
 File "{sys.base_prefix}/lib/python3.7/runpy.py", line 193, in _run_module_as_main
	"__main__", mod_spec)
 File "{sys.base_prefix}/lib/python3.7/runpy.py", line 85, in _run_code
	exec(code, run_globals)
 File "{sys.prefix}/lib/python3.7/site-packages/django/__main__.py", line 9, in <module>
	management.execute_from_command_line()
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/__init__.py", line 381, in execute_from_command_line
	utility.execute()
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/__init__.py", line 375, in execute
	self.fetch_command(subcommand).run_from_argv(self.argv)
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/base.py", line 323, in run_from_argv
	self.execute(*args, **cmd_options)
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/base.py", line 364, in execute
	output = self.handle(*args, **options)
 File "{sys.prefix}/lib/python3.7/site-packages/django/core/management/commands/shell.py", line 86, in handle
	exec(options['command'])
 File "<string>", line 5, in <module>
 File "<string>", line 4, in f
NameError: name 'django' is not defined
The problem is in the ​usage of ​exec:
	def handle(self, **options):
		# Execute the command and exit.
		if options['command']:
			exec(options['command'])
			return
		# Execute stdin if it has anything to read and exit.
		# Not supported on Windows due to select.select() limitations.
		if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
			exec(sys.stdin.read())
			return
exec should be passed a dictionary containing a minimal set of globals. This can be done by just passing a new, empty dictionary as the second argument of exec.
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
​PR includes tests and documents the new feature in the release notes (but not in the main docs since it seems more like a bug fix than a new feature to me).
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/django/core/management/commands/shell.py b/django/core/management/commands/shell.py
index 2b306a5..571d120 100755
--- a/django/core/management/commands/shell.py
+++ b/django/core/management/commands/shell.py
@@ -84,13 +84,13 @@ class Command(BaseCommand):
     def handle(self, **options):
         # Execute the command and exit.
         if options['command']:
-            exec(options['command'])
+            exec(options['command'], {})
             return
 
         # Execute stdin if it has anything to read and exit.
         # Not supported on Windows due to select.select() limitations.
         if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
-            exec(sys.stdin.read())
+            exec(sys.stdin.read(), {})
             return
 
         available_shells = [options['interface']] if options['interface'] else self.shells
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/shell/tests.py b/tests/shell/tests.py
index f33a9ae..52a7f17 100755
--- a/tests/shell/tests.py
+++ b/tests/shell/tests.py
@@ -21,6 +21,11 @@ class ShellCommandTestCase(SimpleTestCase):
             )
         self.assertEqual(cm.records[0].getMessage(), __version__)
 
+    def test_command_option_with_function(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command='import django\ndef f():\n    print(django.__version__)\nf()')
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @unittest.skipIf(sys.platform == 'win32', "Windows select() doesn't support file descriptors.")
     @mock.patch('django.core.management.commands.shell.select')
     def test_stdin_read(self, select):
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/django/core/management/commands/shell.py b/django/core/management/commands/shell.py
--- a/django/core/management/commands/shell.py
+++ b/django/core/management/commands/shell.py
@@ -84,13 +84,13 @@ def python(self, options):
     def handle(self, **options):
         # Execute the command and exit.
         if options['command']:
-            exec(options['command'])
+            exec(options['command'], globals())
             return
 
         # Execute stdin if it has anything to read and exit.
         # Not supported on Windows due to select.select() limitations.
         if sys.platform != 'win32' and not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
-            exec(sys.stdin.read())
+            exec(sys.stdin.read(), globals())
             return
 
         available_shells = [options['interface']] if options['interface'] else self.shells
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/shell/tests.py b/tests/shell/tests.py
--- a/tests/shell/tests.py
+++ b/tests/shell/tests.py
@@ -9,6 +9,13 @@
 
 
 class ShellCommandTestCase(SimpleTestCase):
+    script_globals = 'print("__name__" in globals())'
+    script_with_inline_function = (
+        'import django\n'
+        'def f():\n'
+        '    print(django.__version__)\n'
+        'f()'
+    )
 
     def test_command_option(self):
         with self.assertLogs('test', 'INFO') as cm:
@@ -21,6 +28,16 @@ def test_command_option(self):
             )
         self.assertEqual(cm.records[0].getMessage(), __version__)
 
+    def test_command_option_globals(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command=self.script_globals)
+        self.assertEqual(stdout.getvalue().strip(), 'True')
+
+    def test_command_option_inline_function_call(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command=self.script_with_inline_function)
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @unittest.skipIf(sys.platform == 'win32', "Windows select() doesn't support file descriptors.")
     @mock.patch('django.core.management.commands.shell.select')
     def test_stdin_read(self, select):
@@ -30,6 +47,30 @@ def test_stdin_read(self, select):
             call_command('shell')
         self.assertEqual(stdout.getvalue().strip(), '100')
 
+    @unittest.skipIf(
+        sys.platform == 'win32',
+        "Windows select() doesn't support file descriptors.",
+    )
+    @mock.patch('django.core.management.commands.shell.select')  # [1]
+    def test_stdin_read_globals(self, select):
+        with captured_stdin() as stdin, captured_stdout() as stdout:
+            stdin.write(self.script_globals)
+            stdin.seek(0)
+            call_command('shell')
+        self.assertEqual(stdout.getvalue().strip(), 'True')
+
+    @unittest.skipIf(
+        sys.platform == 'win32',
+        "Windows select() doesn't support file descriptors.",
+    )
+    @mock.patch('django.core.management.commands.shell.select')  # [1]
+    def test_stdin_read_inline_function_call(self, select):
+        with captured_stdin() as stdin, captured_stdout() as stdout:
+            stdin.write(self.script_with_inline_function)
+            stdin.seek(0)
+            call_command('shell')
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @mock.patch('django.core.management.commands.shell.select.select')  # [1]
     @mock.patch.dict('sys.modules', {'IPython': None})
     def test_shell_with_ipython_not_installed(self, select):
```

</details>

<details>
<summary>Verification command at step 5 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 31
Step: 5

COMMAND
python3 -m pytest tests/shell/tests.py -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.416
stdout:
/opt/miniconda3/envs/testbed/bin/python3: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 6 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 36
Step: 6

COMMAND
python3 tests/runtests.py shell 2>&1 | tail -40

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.314
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
System check identified no issues (0 silenced).
.....
----------------------------------------------------------------------
Ran 5 tests in 0.004s

OK
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 9 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 51
Step: 9

COMMAND
python3 tests/runtests.py shell 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.667
stdout:
======================================================================
ERROR: test_command_option_with_function (shell.tests.ShellCommandTestCase)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/testbed/tests/shell/tests.py", line 26, in test_command_option_with_function
    call_command('shell', command='import django\ndef f():\n    print(django.__version__)\nf()')
  ... [truncated]
```

</details>

<details>
<summary>Verification command at step 11 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 61
Step: 11

COMMAND
python3 tests/runtests.py shell 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.676
stdout:
Testing against Django installed in '/testbed/django' with up to 12 processes
System check identified no issues (0 silenced).
.....
----------------------------------------------------------------------
Ran 5 tests in 0.005s

OK
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
