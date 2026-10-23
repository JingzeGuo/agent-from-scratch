# sphinx-doc__sphinx-7686: Member filtering shadows the module name and omits the skip-member hook

**Primary category:** I — Implementation error or introduced regression. **Confidence:** high.

**Recorded score:** F2P 0/2; P2P 13/15. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The patch introduces a variable-scope regression while incompletely implementing member selection.

## Issue requirement

Exclude imported members from the module template members variable when autosummary_imported_members is False.

## Agent change

Adds a for name in ns["members"] loop inside generate_autosummary_content, overwriting its name parameter. It filters imports but does not call autodoc-skip-member.

## Golden change

Uses a separate ModuleScanner and honors both imported-member filtering and skip callbacks, without clobbering the generator argument.

## Observed failure

The skip-member test retains Exc/Foo/bar. Other evidence shows a generated title becoming function_ instead of underscore_module_, and recursive imports using __spec__.module. These are consistent with the overwritten name. The report has 0/2 F2P and 13/15 P2P.

## Scope and attribution

Skip-hook integration is additional accepted behavior, but the name-shadowing regression is an independent implementation defect. A separate raw failure involving pytest-injected @py_builtins names should not be conflated with this defect.

## Verification in the trajectory

At step 39, the agent reads a generated target.rst whose title is os. That visible signal of name corruption remains unresolved when the step budget ends.

## What would improve the outcome

Use a separate member_name variable or helper and assert template fullname/name as well as members. Exercise recursive generation and skip callbacks.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `sphinx-doc__sphinx-7686`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 74 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 74 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `sphinx-doc__sphinx-7686.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

````text
autosummary: The members variable for module template contains imported members
**Describe the bug**
autosummary: The members variable for module template contains imported members even if autosummary_imported_members is False.

**To Reproduce**

```
# _templates/autosummary/module.rst
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   .. autosummary::
   {% for item in members %}
      {{ item }}
   {%- endfor %}

```
```
# example.py
import os
```
```
# index.rst
.. autosummary::
   :toctree: generated

   example
```
```
# conf.py
autosummary_generate = True
autosummary_imported_members = False
```

As a result, I got following output:
```
# generated/example.rst
example
=======

.. automodule:: example

   .. autosummary::

      __builtins__
      __cached__
      __doc__
      __file__
      __loader__
      __name__
      __package__
      __spec__
      os
```

**Expected behavior**
The template variable `members` should not contain imported members when `autosummary_imported_members` is False.

**Your project**
No

**Screenshots**
No

**Environment info**
- OS: Mac
- Python version: 3.8.2
- Sphinx version: 3.1.0dev
- Sphinx extensions:  sphinx.ext.autosummary
- Extra tools: No

**Additional context**
No
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/sphinx/ext/autosummary/generate.py b/sphinx/ext/autosummary/generate.py
index a57c73f..9e88918 100755
--- a/sphinx/ext/autosummary/generate.py
+++ b/sphinx/ext/autosummary/generate.py
@@ -18,6 +18,7 @@
 """
 
 import argparse
+import inspect
 import locale
 import os
 import pkgutil
@@ -247,6 +248,26 @@ def generate_autosummary_content(name: str, obj: Any, parent: Any,
 
     if doc.objtype == 'module':
         ns['members'] = dir(obj)
+        if not imported_members:
+            # skip imported members if expected
+            members = []
+            for name in ns['members']:
+                try:
+                    value = safe_getattr(obj, name)
+                except AttributeError:
+                    continue
+
+                if inspect.ismodule(value) and value is not obj:
+                    if not value.__name__.startswith(obj.__name__ + '.'):
+                        continue
+                else:
+                    modname = getattr(value, '__module__', None)
+                    if modname is not None and modname != obj.__name__:
+                        continue
+
+                members.append(name)
+            ns['members'] = members
+
         ns['functions'], ns['all_functions'] = \
             get_members(obj, {'function'}, imported=imported_members)
         ns['classes'], ns['all_classes'] = \
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/roots/test-ext-autosummary-template/_templates/autosummary/module.rst b/tests/roots/test-ext-autosummary-template/_templates/autosummary/module.rst
new file mode 100644
index 0000000..b3be39c
--- /dev/null
+++ b/tests/roots/test-ext-autosummary-template/_templates/autosummary/module.rst
@@ -0,0 +1,6 @@
+{{ fullname }}
+{{ underline }}
+
+{% for item in members %}
+{{ item }}
+{%- endfor %}

diff --git a/tests/roots/test-ext-autosummary-template/index.rst b/tests/roots/test-ext-autosummary-template/index.rst
index c9f28b0..7f7e742 100755
--- a/tests/roots/test-ext-autosummary-template/index.rst
+++ b/tests/roots/test-ext-autosummary-template/index.rst
@@ -3,3 +3,8 @@
    :template: empty.rst
 
    target.Foo
+
+.. autosummary::
+   :toctree: generate
+
+   target

diff --git a/tests/roots/test-ext-autosummary-template/target.py b/tests/roots/test-ext-autosummary-template/target.py
index c607b59..488d48c 100755
--- a/tests/roots/test-ext-autosummary-template/target.py
+++ b/tests/roots/test-ext-autosummary-template/target.py
@@ -1,2 +1,10 @@
+import os
+from typing import Union
+
+
 class Foo:
     """docstring of Foo."""
+
+
+def foo(x: Union[int, str]) -> None:
+    """docstring of foo."""

diff --git a/tests/test_ext_autosummary.py b/tests/test_ext_autosummary.py
index 166029c..caa2928 100755
--- a/tests/test_ext_autosummary.py
+++ b/tests/test_ext_autosummary.py
@@ -398,6 +398,12 @@ def test_autosummary_template(app):
     content = (app.srcdir / 'generate' / 'target.Foo.rst').read_text()
     assert 'EMPTY' in content
 
+    content = (app.srcdir / 'generate' / 'target.rst').read_text()
+    assert 'os' not in content
+    assert 'Union' not in content
+    assert 'Foo' in content
+    assert 'foo' in content
+
 
 @pytest.mark.sphinx('dummy', testroot='ext-autosummary',
                     confoverrides={'autosummary_generate': []})
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/sphinx/ext/autosummary/generate.py b/sphinx/ext/autosummary/generate.py
--- a/sphinx/ext/autosummary/generate.py
+++ b/sphinx/ext/autosummary/generate.py
@@ -18,6 +18,7 @@
 """
 
 import argparse
+import inspect
 import locale
 import os
 import pkgutil
@@ -176,6 +177,56 @@ def render(self, template_name: str, context: Dict) -> str:
 # -- Generating output ---------------------------------------------------------
 
 
+class ModuleScanner:
+    def __init__(self, app: Any, obj: Any) -> None:
+        self.app = app
+        self.object = obj
+
+    def get_object_type(self, name: str, value: Any) -> str:
+        return get_documenter(self.app, value, self.object).objtype
+
+    def is_skipped(self, name: str, value: Any, objtype: str) -> bool:
+        try:
+            return self.app.emit_firstresult('autodoc-skip-member', objtype,
+                                             name, value, False, {})
+        except Exception as exc:
+            logger.warning(__('autosummary: failed to determine %r to be documented, '
+                              'the following exception was raised:\n%s'),
+                           name, exc, type='autosummary')
+            return False
+
+    def scan(self, imported_members: bool) -> List[str]:
+        members = []
+        for name in dir(self.object):
+            try:
+                value = safe_getattr(self.object, name)
+            except AttributeError:
+                value = None
+
+            objtype = self.get_object_type(name, value)
+            if self.is_skipped(name, value, objtype):
+                continue
+
+            try:
+                if inspect.ismodule(value):
+                    imported = True
+                elif safe_getattr(value, '__module__') != self.object.__name__:
+                    imported = True
+                else:
+                    imported = False
+            except AttributeError:
+                imported = False
+
+            if imported_members:
+                # list all members up
+                members.append(name)
+            elif imported is False:
+                # list not-imported members up
+                members.append(name)
+
+        return members
+
+
 def generate_autosummary_content(name: str, obj: Any, parent: Any,
                                  template: AutosummaryRenderer, template_name: str,
                                  imported_members: bool, app: Any,
@@ -246,7 +297,8 @@ def get_modules(obj: Any) -> Tuple[List[str], List[str]]:
     ns.update(context)
 
     if doc.objtype == 'module':
-        ns['members'] = dir(obj)
+        scanner = ModuleScanner(app, obj)
+        ns['members'] = scanner.scan(imported_members)
         ns['functions'], ns['all_functions'] = \
             get_members(obj, {'function'}, imported=imported_members)
         ns['classes'], ns['all_classes'] = \
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/roots/test-ext-autosummary/autosummary_dummy_module.py b/tests/roots/test-ext-autosummary/autosummary_dummy_module.py
--- a/tests/roots/test-ext-autosummary/autosummary_dummy_module.py
+++ b/tests/roots/test-ext-autosummary/autosummary_dummy_module.py
@@ -1,4 +1,4 @@
-from os import *  # NOQA
+from os import path  # NOQA
 from typing import Union
 
 
@@ -17,7 +17,23 @@ def baz(self):
         pass
 
 
-def bar(x: Union[int, str], y: int = 1):
+class _Baz:
+    pass
+
+
+def bar(x: Union[int, str], y: int = 1) -> None:
+    pass
+
+
+def _quux():
+    pass
+
+
+class Exc(Exception):
+    pass
+
+
+class _Exc(Exception):
     pass
 
 
diff --git a/tests/test_ext_autosummary.py b/tests/test_ext_autosummary.py
--- a/tests/test_ext_autosummary.py
+++ b/tests/test_ext_autosummary.py
@@ -19,7 +19,10 @@
 from sphinx.ext.autosummary import (
     autosummary_table, autosummary_toc, mangle_signature, import_by_name, extract_summary
 )
-from sphinx.ext.autosummary.generate import AutosummaryEntry, generate_autosummary_docs, main as autogen_main
+from sphinx.ext.autosummary.generate import (
+    AutosummaryEntry, generate_autosummary_content, generate_autosummary_docs,
+    main as autogen_main
+)
 from sphinx.testing.util import assert_node, etree_parse
 from sphinx.util.docutils import new_document
 from sphinx.util.osutil import cd
@@ -189,6 +192,83 @@ def test_escaping(app, status, warning):
     assert str_content(title) == 'underscore_module_'
 
 
+@pytest.mark.sphinx(testroot='ext-autosummary')
+def test_autosummary_generate_content_for_module(app):
+    import autosummary_dummy_module
+    template = Mock()
+
+    generate_autosummary_content('autosummary_dummy_module', autosummary_dummy_module, None,
+                                 template, None, False, app, False, {})
+    assert template.render.call_args[0][0] == 'module'
+
+    context = template.render.call_args[0][1]
+    assert context['members'] == ['Exc', 'Foo', '_Baz', '_Exc', '__builtins__',
+                                  '__cached__', '__doc__', '__file__', '__name__',
+                                  '__package__', '_quux', 'bar', 'qux']
+    assert context['functions'] == ['bar']
+    assert context['all_functions'] == ['_quux', 'bar']
+    assert context['classes'] == ['Foo']
+    assert context['all_classes'] == ['Foo', '_Baz']
+    assert context['exceptions'] == ['Exc']
+    assert context['all_exceptions'] == ['Exc', '_Exc']
+    assert context['attributes'] == ['qux']
+    assert context['all_attributes'] == ['qux']
+    assert context['fullname'] == 'autosummary_dummy_module'
+    assert context['module'] == 'autosummary_dummy_module'
+    assert context['objname'] == ''
+    assert context['name'] == ''
+    assert context['objtype'] == 'module'
+
+
+@pytest.mark.sphinx(testroot='ext-autosummary')
+def test_autosummary_generate_content_for_module_skipped(app):
+    import autosummary_dummy_module
+    template = Mock()
+
+    def skip_member(app, what, name, obj, skip, options):
+        if name in ('Foo', 'bar', 'Exc'):
+            return True
+
+    app.connect('autodoc-skip-member', skip_member)
+    generate_autosummary_content('autosummary_dummy_module', autosummary_dummy_module, None,
+                                 template, None, False, app, False, {})
+    context = template.render.call_args[0][1]
+    assert context['members'] == ['_Baz', '_Exc', '__builtins__', '__cached__', '__doc__',
+                                  '__file__', '__name__', '__package__', '_quux', 'qux']
+    assert context['functions'] == []
+    assert context['classes'] == []
+    assert context['exceptions'] == []
+
+
+@pytest.mark.sphinx(testroot='ext-autosummary')
+def test_autosummary_generate_content_for_module_imported_members(app):
+    import autosummary_dummy_module
+    template = Mock()
+
+    generate_autosummary_content('autosummary_dummy_module', autosummary_dummy_module, None,
+                                 template, None, True, app, False, {})
+    assert template.render.call_args[0][0] == 'module'
+
+    context = template.render.call_args[0][1]
+    assert context['members'] == ['Exc', 'Foo', 'Union', '_Baz', '_Exc', '__builtins__',
+                                  '__cached__', '__doc__', '__file__', '__loader__',
+                                  '__name__', '__package__', '__spec__', '_quux',
+                                  'bar', 'path', 'qux']
+    assert context['functions'] == ['bar']
+    assert context['all_functions'] == ['_quux', 'bar']
+    assert context['classes'] == ['Foo']
+    assert context['all_classes'] == ['Foo', '_Baz']
+    assert context['exceptions'] == ['Exc']
+    assert context['all_exceptions'] == ['Exc', '_Exc']
+    assert context['attributes'] == ['qux']
+    assert context['all_attributes'] == ['qux']
+    assert context['fullname'] == 'autosummary_dummy_module'
+    assert context['module'] == 'autosummary_dummy_module'
+    assert context['objname'] == ''
+    assert context['name'] == ''
+    assert context['objtype'] == 'module'
+
+
 @pytest.mark.sphinx('dummy', testroot='ext-autosummary')
 def test_autosummary_generate(app, status, warning):
     app.builder.build_all()
```

</details>

<details>
<summary>Verification command at step 31 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 204
Step: 31

COMMAND
python3 -m pytest tests/test_ext_autosummary.py::test_autosummary_template -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 1.638
stdout:
F                                                                        [100%]
=================================== FAILURES ===================================
__________________________ test_autosummary_template ___________________________

app = <SphinxTestApp buildername='html'>

    @pytest.mark.sphinx(testroot='ext-autosummary-template')
    def test_autosummary_template(app):
        app.build()
    
        content = (app.srcd... [truncated]
```

</details>

<details>
<summary>Verification command at step 36 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 235
Step: 36

COMMAND
python3 -m pytest tests/test_ext_autosummary.py::test_autosummary_template tests/test_ext_autosummary.py::test_autosummary_generate tests/test_ext_autosummary.py::test_autosummary_imported_members -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.257
stdout:

tests/test_ext_autosummary.py::test_autosummary_template
  <template>:401: DeprecationWarning: 'soft_unicode' has been renamed to 'soft_str'. The old name will be removed in MarkupSafe 2.1.

tests/test_ext_autosummary.py::test_autosummary_template
tests/test_ext_autosummary.py::test_autosummary_template
tests/test_ext_autosummary.py::test_autosummary_template
tests/test_ext_autosummary.py::test_autosummary_template
tests/test_ext_aut... [truncated]
```

</details>

<details>
<summary>Verification command at step 37 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 240
Step: 37

COMMAND
python3 -m pytest tests/test_ext_autosummary.py::test_autosummary_template -q 2>&1 | grep -A12 "AssertionError\|assert" | head -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.116
stdout:
        assert 'EMPTY' in content
    
        content = (app.srcdir / 'generate' / 'target.rst').read_text()
>       assert 'os' not in content
E       AssertionError: assert 'os' not in 'os\n==\n\n\...ckage__\nfoo'
E         
E         'os' is contained here:
E           os
E           ==
E           
E           
E           Foo...
E         
E         ...Full output truncated (7 lines hidden), use '-vv' to show

tests/test_ext_aut... [truncated]
```

</details>

<details>
<summary>Verification command at step 38 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 245
Step: 38

COMMAND
python3 -m pytest tests/test_ext_autosummary.py::test_autosummary_template -q -vv 2>&1 | sed -n '/Full output/,+18p'

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 1.133
stdout:
E         ...Full output truncated (7 lines hidden), use '-vv' to show

tests/test_ext_autosummary.py:402: AssertionError
--------------------------- Captured stdout teardown ---------------------------
# testroot: root
# builder: html
# srcdir: /tmp/pytest-of-root/pytest-3/ext-autosummary-template
# outdir: /tmp/pytest-of-root/pytest-3/ext-autosummary-template/_build/html
# status: 
[01mRunning Sphinx v3.1.0+/5bcd897[39;49;00m
[aut... [truncated]
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 250
Step: 39

COMMAND
cat /tmp/pytest-of-root/pytest-3/ext-autosummary-template/generate/target.rst 2>/dev/null | head -30; echo '---'; ls /tmp/pytest-of-root/pytest-3/ext-autosummary-template/generate/

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.315
stdout:
os
==


Foo
__builtins__
__cached__
__doc__
__file__
__name__
__package__
foo---
target.Foo.rst
target.rst
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
