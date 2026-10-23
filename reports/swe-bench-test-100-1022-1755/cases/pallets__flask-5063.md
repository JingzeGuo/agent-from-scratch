# pallets__flask-5063: Domain information is present, but accepted column headings differ from the issue example

**Primary category:** S — Unclear or conflicting acceptance contract. **Confidence:** high.

**Recorded score:** F2P 0/2; P2P 54/54. **Steps:** 39/40. Counts are harness classifications, subject to the report's parser caveats.

Both scored failures are heading checks: Domain versus Subdomain/Host.

## Issue requirement

Display domain/subdomain information in flask routes; the proposed output labels the first column Domain and shows fully qualified domains.

## Agent change

Adds a Domain column, assembles full domains when SERVER_NAME is present, supports host rules and domain sorting.

## Golden change

Uses a Subdomain or Host heading according to host_matching, and displays the corresponding rule attribute.

## Observed failure

Both F2P invocations exit successfully and show the correct a/b domain values. Tests fail solely because Subdomain or Host is absent from the Domain-headed output. All 54 P2P tests pass.

## Scope and attribution

This is a strong acceptance-detail mismatch: the agent follows the issue example. Other design differences exist, but these logs establish heading mismatches as the scored failures, not lost domain data.

## Verification in the trajectory

The agent tests subdomains, host cases and sorting using its chosen output schema; the final recorded CLI suite reports 58 passed.

## What would improve the outcome

Make required CLI headings explicit in task acceptance. Treat table schema as an API choice and verify host versus subdomain semantics without assuming the example is the final specification.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 52 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 52 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 52 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/pallets__flask-5063/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/pallets__flask-5063/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/pallets__flask-5063/run_instance.log) | — |
| trajectory | [pallets__flask-5063.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/pallets__flask-5063.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

````text
Flask routes to return domain/sub-domains information
Currently when checking **flask routes** it provides all routes but **it is no way to see which routes are assigned to which subdomain**.

**Default server name:**
SERVER_NAME: 'test.local'

**Domains (sub-domains):**
test.test.local
admin.test.local
test.local

**Adding blueprints:**
app.register_blueprint(admin_blueprint,url_prefix='',subdomain='admin')
app.register_blueprint(test_subdomain_blueprint,url_prefix='',subdomain='test')


```
$ flask routes
 * Tip: There are .env or .flaskenv files present. Do "pip install python-dotenv" to use them.
Endpoint                                                 Methods    Rule
-------------------------------------------------------  ---------  ------------------------------------------------
admin_blueprint.home                                      GET        /home
test_subdomain_blueprint.home                             GET        /home
static                                                    GET        /static/<path:filename>
...
```


**Feature request**
It will be good to see something like below (that will make more clear which route for which subdomain, because now need to go and check configuration).
**If it is not possible to fix routes**, can you add or tell which method(s) should be used to get below information from flask? 

```
$ flask routes
 * Tip: There are .env or .flaskenv files present. Do "pip install python-dotenv" to use them.
Domain                Endpoint                                             Methods    Rule
-----------------   ----------------------------------------------------  ----------  ------------------------------------------------
admin.test.local     admin_blueprint.home                                  GET        /home
test.test.local      test_subdomain_blueprint.home                         GET        /home
test.local           static                                                GET        /static/<path:filename>
...
```
````

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/src/flask/cli.py b/src/flask/cli.py
index 37a15ff..0d9f5f2 100755
--- a/src/flask/cli.py
+++ b/src/flask/cli.py
@@ -989,7 +989,7 @@ def shell_command() -> None:
 @click.option(
     "--sort",
     "-s",
-    type=click.Choice(("endpoint", "methods", "rule", "match")),
+    type=click.Choice(("endpoint", "methods", "domain", "rule", "match")),
     default="endpoint",
     help=(
         'Method to sort routes by. "match" is the order that Flask will match '
@@ -1007,31 +1007,60 @@ def routes_command(sort: str, all_methods: bool) -> None:
         return
 
     ignored_methods = set(() if all_methods else ("HEAD", "OPTIONS"))
+    server_name = current_app.config["SERVER_NAME"]
+
+    def get_domain(rule):
+        if rule.host is not None:
+            return rule.host
+
+        if server_name is not None:
+            if rule.subdomain:
+                return f"{rule.subdomain}.{server_name}"
+
+            return server_name
+
+        return rule.subdomain or ""
 
     if sort in ("endpoint", "rule"):
         rules = sorted(rules, key=attrgetter(sort))
     elif sort == "methods":
         rules = sorted(rules, key=lambda rule: sorted(rule.methods))  # type: ignore
+    elif sort == "domain":
+        rules = sorted(rules, key=get_domain)
 
     rule_methods = [
         ", ".join(sorted(rule.methods - ignored_methods))  # type: ignore
         for rule in rules
     ]
+    rule_domains = [get_domain(rule) for rule in rules]
+    show_domain = bool(server_name or any(rule_domains))
 
-    headers = ("Endpoint", "Methods", "Rule")
-    widths = (
-        max(len(rule.endpoint) for rule in rules),
-        max(len(methods) for methods in rule_methods),
-        max(len(rule.rule) for rule in rules),
-    )
-    widths = [max(len(h), w) for h, w in zip(headers, widths)]
-    row = "{{0:<{0}}}  {{1:<{1}}}  {{2:<{2}}}".format(*widths)
+    rows = []
+
+    for rule, methods, domain in zip(rules, rule_methods, rule_domains):
+        row = [rule.endpoint, methods, rule.rule]
+
+        if show_domain:
+            row.insert(0, domain)
+
+        rows.append(row)
+
+    headers = ["Endpoint", "Methods", "Rule"]
+
+    if show_domain:
+        headers.insert(0, "Domain")
+
+    widths = [
+        max(len(header), *(len(row[i]) for row in rows))
+        for i, header in enumerate(headers)
+    ]
+    row_format = "  ".join(f"{{:<{width}}}" for width in widths)
 
-    click.echo(row.format(*headers).strip())
-    click.echo(row.format(*("-" * width for width in widths)))
+    click.echo(row_format.format(*headers).strip())
+    click.echo(row_format.format(*("-" * width for width in widths)))
 
-    for rule, methods in zip(rules, rule_methods):
-        click.echo(row.format(rule.endpoint, methods, rule.rule).rstrip())
+    for row in rows:
+        click.echo(row_format.format(*row).rstrip())
 
 
 cli = FlaskGroup(
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/test_cli.py b/tests/test_cli.py
index 0d9625b..3399fe9 100755
--- a/tests/test_cli.py
+++ b/tests/test_cli.py
@@ -498,6 +498,50 @@ class TestRoutes:
         assert result.exit_code == 0
         assert "No routes were registered." in result.output
 
+    def test_host_matching(self, app, runner):
+        app.url_map.host_matching = True
+
+        @app.route("/host", host="example.com")
+        def host():
+            pass
+
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
+        assert result.exit_code == 0
+        assert "Domain" in result.output
+        assert "example.com" in result.output
+
+    def test_subdomain_matching(self, app, runner):
+        app.config["SERVER_NAME"] = "test.local"
+        app.url_map.default_subdomain = ""
+
+        @app.route("/subdomain", subdomain="test")
+        def subdomain():
+            pass
+
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
+        assert result.exit_code == 0
+        assert "Domain" in result.output
+        assert "test.test.local" in result.output
+
+    def test_sort_domain(self, app, runner):
+        app.config["SERVER_NAME"] = "test.local"
+
+        @app.route("/admin", subdomain="admin")
+        def admin():
+            pass
+
+        @app.route("/test", subdomain="test")
+        def test():
+            pass
+
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes", "-s", "domain"])
+        assert result.exit_code == 0
+        endpoints = [line.split()[1] for line in result.output.splitlines()[2:]]
+        assert endpoints == ["admin", "static", "yyy_get_post", "aaa_post", "test"]
+
 
 def dotenv_not_available():
     try:
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/src/flask/cli.py b/src/flask/cli.py
--- a/src/flask/cli.py
+++ b/src/flask/cli.py
@@ -9,7 +9,7 @@
 import traceback
 import typing as t
 from functools import update_wrapper
-from operator import attrgetter
+from operator import itemgetter
 
 import click
 from click.core import ParameterSource
@@ -989,49 +989,62 @@ def shell_command() -> None:
 @click.option(
     "--sort",
     "-s",
-    type=click.Choice(("endpoint", "methods", "rule", "match")),
+    type=click.Choice(("endpoint", "methods", "domain", "rule", "match")),
     default="endpoint",
     help=(
-        'Method to sort routes by. "match" is the order that Flask will match '
-        "routes when dispatching a request."
+        "Method to sort routes by. 'match' is the order that Flask will match routes"
+        " when dispatching a request."
     ),
 )
 @click.option("--all-methods", is_flag=True, help="Show HEAD and OPTIONS methods.")
 @with_appcontext
 def routes_command(sort: str, all_methods: bool) -> None:
     """Show all registered routes with endpoints and methods."""
-
     rules = list(current_app.url_map.iter_rules())
+
     if not rules:
         click.echo("No routes were registered.")
         return
 
-    ignored_methods = set(() if all_methods else ("HEAD", "OPTIONS"))
+    ignored_methods = set() if all_methods else {"HEAD", "OPTIONS"}
+    host_matching = current_app.url_map.host_matching
+    has_domain = any(rule.host if host_matching else rule.subdomain for rule in rules)
+    rows = []
 
-    if sort in ("endpoint", "rule"):
-        rules = sorted(rules, key=attrgetter(sort))
-    elif sort == "methods":
-        rules = sorted(rules, key=lambda rule: sorted(rule.methods))  # type: ignore
+    for rule in rules:
+        row = [
+            rule.endpoint,
+            ", ".join(sorted((rule.methods or set()) - ignored_methods)),
+        ]
 
-    rule_methods = [
-        ", ".join(sorted(rule.methods - ignored_methods))  # type: ignore
-        for rule in rules
-    ]
+        if has_domain:
+            row.append((rule.host if host_matching else rule.subdomain) or "")
 
-    headers = ("Endpoint", "Methods", "Rule")
-    widths = (
-        max(len(rule.endpoint) for rule in rules),
-        max(len(methods) for methods in rule_methods),
-        max(len(rule.rule) for rule in rules),
-    )
-    widths = [max(len(h), w) for h, w in zip(headers, widths)]
-    row = "{{0:<{0}}}  {{1:<{1}}}  {{2:<{2}}}".format(*widths)
+        row.append(rule.rule)
+        rows.append(row)
+
+    headers = ["Endpoint", "Methods"]
+    sorts = ["endpoint", "methods"]
+
+    if has_domain:
+        headers.append("Host" if host_matching else "Subdomain")
+        sorts.append("domain")
+
+    headers.append("Rule")
+    sorts.append("rule")
+
+    try:
+        rows.sort(key=itemgetter(sorts.index(sort)))
+    except ValueError:
+        pass
 
-    click.echo(row.format(*headers).strip())
-    click.echo(row.format(*("-" * width for width in widths)))
+    rows.insert(0, headers)
+    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
+    rows.insert(1, ["-" * w for w in widths])
+    template = "  ".join(f"{{{i}:<{w}}}" for i, w in enumerate(widths))
 
-    for rule, methods in zip(rules, rule_methods):
-        click.echo(row.format(rule.endpoint, methods, rule.rule).rstrip())
+    for row in rows:
+        click.echo(template.format(*row))
 
 
 cli = FlaskGroup(
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/test_cli.py b/tests/test_cli.py
--- a/tests/test_cli.py
+++ b/tests/test_cli.py
@@ -433,16 +433,12 @@ class TestRoutes:
     @pytest.fixture
     def app(self):
         app = Flask(__name__)
-        app.testing = True
-
-        @app.route("/get_post/<int:x>/<int:y>", methods=["GET", "POST"])
-        def yyy_get_post(x, y):
-            pass
-
-        @app.route("/zzz_post", methods=["POST"])
-        def aaa_post():
-            pass
-
+        app.add_url_rule(
+            "/get_post/<int:x>/<int:y>",
+            methods=["GET", "POST"],
+            endpoint="yyy_get_post",
+        )
+        app.add_url_rule("/zzz_post", methods=["POST"], endpoint="aaa_post")
         return app
 
     @pytest.fixture
@@ -450,17 +446,6 @@ def invoke(self, app, runner):
         cli = FlaskGroup(create_app=lambda: app)
         return partial(runner.invoke, cli)
 
-    @pytest.fixture
-    def invoke_no_routes(self, runner):
-        def create_app():
-            app = Flask(__name__, static_folder=None)
-            app.testing = True
-
-            return app
-
-        cli = FlaskGroup(create_app=create_app)
-        return partial(runner.invoke, cli)
-
     def expect_order(self, order, output):
         # skip the header and match the start of each row
         for expect, line in zip(order, output.splitlines()[2:]):
@@ -493,11 +478,31 @@ def test_all_methods(self, invoke):
         output = invoke(["routes", "--all-methods"]).output
         assert "GET, HEAD, OPTIONS, POST" in output
 
-    def test_no_routes(self, invoke_no_routes):
-        result = invoke_no_routes(["routes"])
+    def test_no_routes(self, runner):
+        app = Flask(__name__, static_folder=None)
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
         assert result.exit_code == 0
         assert "No routes were registered." in result.output
 
+    def test_subdomain(self, runner):
+        app = Flask(__name__, static_folder=None)
+        app.add_url_rule("/a", subdomain="a", endpoint="a")
+        app.add_url_rule("/b", subdomain="b", endpoint="b")
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
+        assert result.exit_code == 0
+        assert "Subdomain" in result.output
+
+    def test_host(self, runner):
+        app = Flask(__name__, static_folder=None, host_matching=True)
+        app.add_url_rule("/a", host="a", endpoint="a")
+        app.add_url_rule("/b", host="b", endpoint="b")
+        cli = FlaskGroup(create_app=lambda: app)
+        result = runner.invoke(cli, ["routes"])
+        assert result.exit_code == 0
+        assert "Host" in result.output
+
 
 def dotenv_not_available():
     try:
```

</details>

<details>
<summary>Verification command at step 15 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 105
Step: 15

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.698
stdout:
        assert result.exit_code == 0
>       assert "Host" in result.output
E       AssertionError: assert 'Host' in 'Endpoint      Methods    Rule\n------------  ---------  -------------------------\naaa_post      POST       /zzz_post...T        /host\nstatic        GET        /static/<path:filename>\nyyy_get_post  GET, POST  /get_post/<int:x>/<int:y>\n'
E        +  where 'Endpoint      Methods    Rule\n------------  ---------  -----... [truncated]
```

</details>

<details>
<summary>Verification command at step 18 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 123
Step: 18

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.603
stdout:
>       assert result.exit_code == 0
E       assert 1 == 0
E        +  where 1 = <Result ValueError("Invalid format specifier '<Domain' for object of type 'str'")>.exit_code

tests/test_cli.py:511: AssertionError
______________________ TestRoutes.test_subdomain_matching ______________________

self = <test_cli.TestRoutes object at 0x7ffffd8fa0d0>, app = <Flask 'test_cli'>
runner = <click.testing.CliRunner object at 0x7ffffd8f9750>

  ... [truncated]
```

</details>

<details>
<summary>Verification command at step 21 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 138
Step: 21

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.580
stdout:
        assert result.exit_code == 0
>       assert "Host" in result.output
E       AssertionError: assert 'Host' in 'Domain       Endpoint      Methods    Rule\n-----------  ------------  ---------  -------------------------\n        ...   static        GET        /static/<path:filename>\n             yyy_get_post  GET, POST  /get_post/<int:x>/<int:y>\n'
E        +  where 'Domain       Endpoint      Methods    Rule\n-----------  ----... [truncated]
```

</details>

<details>
<summary>Verification command at step 28 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 176
Step: 28

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.561
stdout:
......                                                                   [100%]
6 passed in 0.03s
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 32 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 196
Step: 32

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -35

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.616
stdout:
        app.config["SERVER_NAME"] = "test.local"
    
        @app.route("/admin", subdomain="admin")
        def admin():
            pass
    
        @app.route("/test", subdomain="test")
        def test():
            pass
    
        cli = FlaskGroup(create_app=lambda: app)
        result = runner.invoke(cli, ["routes", "-s", "domain"])
        assert result.exit_code == 0
>       self.expect_order(["admin", "static", "test"], ... [truncated]
```

</details>

<details>
<summary>Verification command at step 34 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 206
Step: 34

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -30

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.593
stdout:
=================================== FAILURES ===================================
_________________________ TestRoutes.test_sort_domain __________________________

self = <test_cli.TestRoutes object at 0x7ffffd7964d0>, app = <Flask 'test_cli'>
runner = <click.testing.CliRunner object at 0x7ffffd6e1b50>

    def test_sort_domain(self, app, runner):
        app.config["SERVER_NAME"] = "test.local"
    
        @app.route("/admin", subdom... [truncated]
```

</details>

<details>
<summary>Verification command at step 36 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 216
Step: 36

COMMAND
python3 -m pytest tests/test_cli.py::TestRoutes -q 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.580
stdout:
.......                                                                  [100%]
7 passed in 0.07s
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 37 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 221
Step: 37

COMMAND
python3 -m pytest tests/test_cli.py -q 2>&1 | tail -10; python3 -m flake8 src/flask/cli.py tests/test_cli.py 2>&1 | head -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.720
stdout:
..........................................................               [100%]
58 passed in 0.13s
/opt/miniconda3/envs/testbed/bin/python3: No module named flake8
stderr:
[empty]
```

</details>


[Return to main report](../README.md)
