# pallets__flask-4045: Blueprint name validation passes; unrelated exception-type harmonization is missing

**Primary category:** G — Golden change extends or shifts the stated scope. **Confidence:** high.

**Recorded score:** F2P 1/2; P2P 50/50. **Steps:** 22/40. Counts are harness classifications, subject to the report's parser caveats.

The only scored failure concerns endpoint/view-function errors, beyond the requested blueprint-name check.

## Issue requirement

Raise an error when a blueprint name contains a dot; the issue says endpoint-name validation already exists.

## Agent change

Adds the same Blueprint.__init__ ValueError check as the golden patch.

## Golden change

Also replaces AssertionError with ValueError for dotted endpoint names and dotted view-function names in add_url_rule.

## Observed failure

The blueprint-name F2P case passes. test_route_decorator_custom_endpoint_with_dots expects ValueError but receives the preexisting AssertionError. All 50 scored P2P tests pass.

## Scope and attribution

The extra exception-type changes are concrete accepted scope beyond the literal request. This should not be described as failure to add blueprint-name validation.

## Verification in the trajectory

The agent runs the new dotted-name test and related defaults tests. Broader raw output contains dependency/deprecation failures; the scored rejection has a direct, separate exception-type explanation.

## What would improve the outcome

Audit this as a scope-expanded task. In general, inspect adjacent validation contracts when adding an error, while distinguishing consistency improvements from explicit issue requirements.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `pallets__flask-4045`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 51 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 51 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `pallets__flask-4045.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

```text
Raise error when blueprint name contains a dot
This is required since every dot is now significant since blueprints can be nested. An error was already added for endpoint names in 1.0, but should have been added for this as well.
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
index 5fb84d8..24a5f4a 100755
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -188,6 +188,10 @@ class Blueprint(Scaffold):
             template_folder=template_folder,
             root_path=root_path,
         )
+
+        if "." in name:
+            raise ValueError("'name' may not contain a dot '.' character.")
+
         self.name = name
         self.url_prefix = url_prefix
         self.subdomain = subdomain
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/test_basic.py b/tests/test_basic.py
index d6ec3fe..7bdeba1 100755
--- a/tests/test_basic.py
+++ b/tests/test_basic.py
@@ -1631,7 +1631,7 @@ def test_url_processors(app, client):
 
 
 def test_inject_blueprint_url_defaults(app):
-    bp = flask.Blueprint("foo.bar.baz", __name__, template_folder="template")
+    bp = flask.Blueprint("foo", __name__, template_folder="template")
 
     @bp.url_defaults
     def bp_defaults(endpoint, values):
@@ -1644,12 +1644,12 @@ def test_inject_blueprint_url_defaults(app):
     app.register_blueprint(bp)
 
     values = dict()
-    app.inject_url_defaults("foo.bar.baz.view", values)
+    app.inject_url_defaults("foo.view", values)
     expected = dict(page="login")
     assert values == expected
 
     with app.test_request_context("/somepage"):
-        url = flask.url_for("foo.bar.baz.view")
+        url = flask.url_for("foo.view")
     expected = "/login"
     assert url == expected
 

diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py
index b986ca0..c0669fa 100755
--- a/tests/test_blueprints.py
+++ b/tests/test_blueprints.py
@@ -253,13 +253,18 @@ def test_templates_list(test_apps):
     assert templates == ["admin/index.html", "frontend/index.html"]
 
 
+def test_blueprint_name_with_dot():
+    with pytest.raises(ValueError, match="may not contain a dot"):
+        flask.Blueprint("myapp.frontend", __name__)
+
+
 def test_dotted_names(app, client):
-    frontend = flask.Blueprint("myapp.frontend", __name__)
-    backend = flask.Blueprint("myapp.backend", __name__)
+    frontend = flask.Blueprint("frontend", __name__)
+    backend = flask.Blueprint("backend", __name__)
 
     @frontend.route("/fe")
     def frontend_index():
-        return flask.url_for("myapp.backend.backend_index")
+        return flask.url_for("backend.backend_index")
 
     @frontend.route("/fe2")
     def frontend_page2():
@@ -267,7 +272,7 @@ def test_dotted_names(app, client):
 
     @backend.route("/be")
     def backend_index():
-        return flask.url_for("myapp.frontend.frontend_index")
+        return flask.url_for("frontend.frontend_index")
 
     app.register_blueprint(frontend)
     app.register_blueprint(backend)
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -188,6 +188,10 @@ def __init__(
             template_folder=template_folder,
             root_path=root_path,
         )
+
+        if "." in name:
+            raise ValueError("'name' may not contain a dot '.' character.")
+
         self.name = name
         self.url_prefix = url_prefix
         self.subdomain = subdomain
@@ -360,12 +364,12 @@ def add_url_rule(
         """Like :meth:`Flask.add_url_rule` but for a blueprint.  The endpoint for
         the :func:`url_for` function is prefixed with the name of the blueprint.
         """
-        if endpoint:
-            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
-        if view_func and hasattr(view_func, "__name__"):
-            assert (
-                "." not in view_func.__name__
-            ), "Blueprint view function name should not contain dots"
+        if endpoint and "." in endpoint:
+            raise ValueError("'endpoint' may not contain a dot '.' character.")
+
+        if view_func and hasattr(view_func, "__name__") and "." in view_func.__name__:
+            raise ValueError("'view_func' name may not contain a dot '.' character.")
+
         self.record(lambda s: s.add_url_rule(rule, endpoint, view_func, **options))
 
     def app_template_filter(self, name: t.Optional[str] = None) -> t.Callable:
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/test_basic.py b/tests/test_basic.py
--- a/tests/test_basic.py
+++ b/tests/test_basic.py
@@ -1631,7 +1631,7 @@ def something_else():
 
 
 def test_inject_blueprint_url_defaults(app):
-    bp = flask.Blueprint("foo.bar.baz", __name__, template_folder="template")
+    bp = flask.Blueprint("foo", __name__, template_folder="template")
 
     @bp.url_defaults
     def bp_defaults(endpoint, values):
@@ -1644,12 +1644,12 @@ def view(page):
     app.register_blueprint(bp)
 
     values = dict()
-    app.inject_url_defaults("foo.bar.baz.view", values)
+    app.inject_url_defaults("foo.view", values)
     expected = dict(page="login")
     assert values == expected
 
     with app.test_request_context("/somepage"):
-        url = flask.url_for("foo.bar.baz.view")
+        url = flask.url_for("foo.view")
     expected = "/login"
     assert url == expected
 
diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py
--- a/tests/test_blueprints.py
+++ b/tests/test_blueprints.py
@@ -1,5 +1,3 @@
-import functools
-
 import pytest
 from jinja2 import TemplateNotFound
 from werkzeug.http import parse_cache_control_header
@@ -253,28 +251,9 @@ def test_templates_list(test_apps):
     assert templates == ["admin/index.html", "frontend/index.html"]
 
 
-def test_dotted_names(app, client):
-    frontend = flask.Blueprint("myapp.frontend", __name__)
-    backend = flask.Blueprint("myapp.backend", __name__)
-
-    @frontend.route("/fe")
-    def frontend_index():
-        return flask.url_for("myapp.backend.backend_index")
-
-    @frontend.route("/fe2")
-    def frontend_page2():
-        return flask.url_for(".frontend_index")
-
-    @backend.route("/be")
-    def backend_index():
-        return flask.url_for("myapp.frontend.frontend_index")
-
-    app.register_blueprint(frontend)
-    app.register_blueprint(backend)
-
-    assert client.get("/fe").data.strip() == b"/be"
-    assert client.get("/fe2").data.strip() == b"/fe"
-    assert client.get("/be").data.strip() == b"/fe"
+def test_dotted_name_not_allowed(app, client):
+    with pytest.raises(ValueError):
+        flask.Blueprint("app.ui", __name__)
 
 
 def test_dotted_names_from_app(app, client):
@@ -343,62 +322,19 @@ def index():
 def test_route_decorator_custom_endpoint_with_dots(app, client):
     bp = flask.Blueprint("bp", __name__)
 
-    @bp.route("/foo")
-    def foo():
-        return flask.request.endpoint
-
-    try:
-
-        @bp.route("/bar", endpoint="bar.bar")
-        def foo_bar():
-            return flask.request.endpoint
-
-    except AssertionError:
-        pass
-    else:
-        raise AssertionError("expected AssertionError not raised")
-
-    try:
-
-        @bp.route("/bar/123", endpoint="bar.123")
-        def foo_bar_foo():
-            return flask.request.endpoint
-
-    except AssertionError:
-        pass
-    else:
-        raise AssertionError("expected AssertionError not raised")
-
-    def foo_foo_foo():
-        pass
-
-    pytest.raises(
-        AssertionError,
-        lambda: bp.add_url_rule("/bar/123", endpoint="bar.123", view_func=foo_foo_foo),
-    )
-
-    pytest.raises(
-        AssertionError, bp.route("/bar/123", endpoint="bar.123"), lambda: None
-    )
-
-    foo_foo_foo.__name__ = "bar.123"
+    with pytest.raises(ValueError):
+        bp.route("/", endpoint="a.b")(lambda: "")
 
-    pytest.raises(
-        AssertionError, lambda: bp.add_url_rule("/bar/123", view_func=foo_foo_foo)
-    )
+    with pytest.raises(ValueError):
+        bp.add_url_rule("/", endpoint="a.b")
 
-    bp.add_url_rule(
-        "/bar/456", endpoint="foofoofoo", view_func=functools.partial(foo_foo_foo)
-    )
+    def view():
+        return ""
 
-    app.register_blueprint(bp, url_prefix="/py")
+    view.__name__ = "a.b"
 
-    assert client.get("/py/foo").data == b"bp.foo"
-    # The rule's didn't actually made it through
-    rv = client.get("/py/bar")
-    assert rv.status_code == 404
-    rv = client.get("/py/bar/123")
-    assert rv.status_code == 404
+    with pytest.raises(ValueError):
+        bp.add_url_rule("/", view_func=view)
 
 
 def test_endpoint_decorator(app, client):
```

</details>

<details>
<summary>Verification command at step 16 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 98
Step: 16

COMMAND
python -m pytest tests/test_blueprints.py tests/test_basic.py::test_inject_blueprint_url_defaults -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 1.982
stdout:
FFFFFFFFFFFFFFFFEFE.FFFFFF....FFFFF....FFFFFF.FFFFF                      [100%]
==================================== ERRORS ====================================
_________________ ERROR at setup of test_templates_and_static __________________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7ffffdacc3d0>

    @pytest.fixture
    def test_apps(monkeypatch):
>       monkeypatch.syspath_prepend(os.path.join(os.path.dirname(__fi... [truncated]
```

</details>

<details>
<summary>Verification command at step 17 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 103
Step: 17

COMMAND
python -m pytest tests/test_blueprints.py::test_blueprint_name_with_dot -q

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.536
stdout:
.                                                                        [100%]
1 passed in 0.01s
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 18 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 108
Step: 18

COMMAND
python -m pytest tests/test_blueprints.py tests/test_basic.py::test_inject_blueprint_url_defaults -q -W ignore::DeprecationWarning

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 1.561
stdout:
FFFFFFFFFFFFFFFFF...FFFFFF....FFFFF....FFFFFF.FFFF.                      [100%]
=================================== FAILURES ===================================
____________________ test_blueprint_specific_error_handling ____________________

app = <Flask 'flask_test'>, client = <FlaskClient <Flask 'flask_test'>>

    def test_blueprint_specific_error_handling(app, client):
        frontend = flask.Blueprint("frontend", __name__)
    ... [truncated]
```

</details>

<details>
<summary>Verification command at step 20 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 122
Step: 20

COMMAND
python -m pytest tests/test_blueprints.py::test_blueprint_name_with_dot tests/test_basic.py::test_inject_blueprint_url_defaults -q -W ignore::DeprecationWarning

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.579
stdout:
..                                                                       [100%]
2 passed in 0.03s
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
