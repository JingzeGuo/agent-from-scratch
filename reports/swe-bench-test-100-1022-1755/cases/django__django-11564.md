# django__django-11564: Distributed URL changes introduce a circular import

**Primary category:** I — Implementation error or introduced regression. **Confidence:** high.

**Recorded score:** F2P unavailable; P2P unavailable. **Steps:** 40/40. Counts are harness classifications, subject to the report's parser caveats.

The submitted patch cannot import Django; it also prefixes root-relative URLs instead of implementing the settings-level contract.

## Issue requirement

Support dynamic SCRIPT_NAME in static/media URLs; the issue discusses template tags and storage as possible change sites.

## Agent change

Adds prefix logic in storage, template context processors and static tags, with module-level imports from django.urls. It prefixes strings starting with a slash.

## Golden change

Centralizes prefixing in LazySettings.STATIC_URL and MEDIA_URL, leaves absolute paths/URLs alone, and imports get_script_prefix lazily.

## Observed failure

run_instance.log explicitly says the source patch applied cleanly. The test runner then fails on ImportError: cannot import name get_script_prefix, through urls -> forms -> templatetags.static -> urls. No reliable per-test grading is available.

## Scope and attribution

There is ambiguity about which URLs should be prefixed and where the feature belongs. Nevertheless, the immediate blocker is an agent-introduced circular import, independently of the golden design.

## Verification in the trajectory

At steps 38–39, pytest is unavailable and the native Django runner hits the import cycle. Both piped commands report exit_code 0. The step-40 final response explicitly admits the tree is broken.

## What would improve the outcome

Move the shared behavior to the lowest common configuration layer, use a lazy import where required, and reserve time for a fresh-process import smoke test and URL classification cases.

## Source evidence

Raw harness logs and trajectories are local-only (not published); the excerpts below were taken from them. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) (test split, `django__django-11564`) | — |
| prediction | [predictions.jsonl](../data/predictions.jsonl) | 7 |
| metrics + per-test results | [results.jsonl](../data/results.jsonl) | 7 |
| test_output | `test_output.txt` (local only) | — |
| harness_log | `run_instance.log` (local only) | — |
| trajectory | `django__django-11564.jsonl` (local only) | — |

<details>
<summary>Supplied problem statement</summary>

```text
Add support for SCRIPT_NAME in STATIC_URL and MEDIA_URL
Description
	 
		(last modified by Rostyslav Bryzgunov)
	 
By default, {% static '...' %} tag just appends STATIC_URL in the path. When running on sub-path, using SCRIPT_NAME WSGI param, it results in incorrect static URL - it doesn't prepend SCRIPT_NAME prefix.
This problem can be solved with prepending SCRIPT_NAME to STATIC_URL in settings.py but that doesn't work when SCRIPT_NAME is a dynamic value.
This can be easily added into default Django static tag and django.contrib.staticfiles tag as following:
def render(self, context):
	url = self.url(context)
	# Updating url here with request.META['SCRIPT_NAME'] 
	if self.varname is None:
		return url
	context[self.varname] = url
		return ''
On more research I found that FileSystemStorage and StaticFilesStorage ignores SCRIPT_NAME as well. 
We might have to do a lot of changes but I think it's worth the efforts.
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
This change doesn't seem correct to me (for one, it seems like it could break existing sites). Why not include the appropriate prefix in your STATIC_URL and MEDIA_URL settings?
This is not a patch. This is just an idea I got about the patch for {% static %} only. The patch will (probably) involve FileSystemStorage and StaticFileSystemStorage classes. The main idea behind this feature was that Django will auto detect script_name header and use that accordingly for creating static and media urls. This will reduce human efforts for setting up sites in future. This patch will also take time to develop so it can be added in Django2.0 timeline.
What I meant was that I don't think Django should automatically use SCRIPT_NAME in generating those URLs. If you're running your site on a subpath, then you should set your STATIC_URL to '​http://example.com/subpath/static/' or whatever. However, you might not even be hosting static and uploaded files on the same domain as your site (in fact, for user-uploaded files, you shouldn't do that ​for security reasons) in which case SCRIPT_URL is irrelevant in constructing the static/media URLs. How would the change make it easier to setup sites?
I think that the idea basically makes sense. Ideally, a Django instance shouldn't need to know at which subpath it is being deployed, as this can be considered as purely sysadmin stuff. It would be a good separation of concerns. For example, the Web administrator may change the WSGIScriptAlias from /foo to /bar and the application should continue working. Of course, this only applies when *_URL settings are not full URIs. In practice, it's very likely that many running instances are adapting their *_URL settings to include the base script path, hence the behavior change would be backwards incompatible. The question is whether the change is worth the incompatibility.
I see. I guess the idea would be to use get_script_prefix() like reverse() does as I don't think we have access to request everywhere we need it. It seems like some public APIs like get_static_url() and get_media_url() would replace accessing the settings directly whenever building URLs. For backwards compatibility, possibly these functions could try to detect if the setting is already prefixed appropriately. Removing the prefix from the settings, however, means that the URLs are no longer correct when generated outside of a request/response cycle though (#16734). I'm not sure if it might create any practical problems, but we might think about addressing that issue first.
I'm here at DjangoCon US 2016 will try to create a patch for this ticket ;) Why? But before I make the patch, here are some reasons to do it. The first reason is consistency inside Django core: {% url '...' %} template tag does respect SCRIPT_NAME but {% static '...' %} does not reverse(...) function does respect SCRIPT_NAME but static(...) does not And the second reason is that there is no way to make it work in case when SCRIPT_NAME is a dynamic value - see an example below. Of course we shouldn't modify STATIC_URL when it's an absolute URL, with domain & protocol. But if it starts with / - it's relative to our Django project and we need to add SCRIPT_NAME prefix. Real life example You have Django running via WSGI behind reverse proxy (let's call it back-end server), and another HTTP server on the front (let's call it front-end server). Front-end server URL is http://some.domain.com/sub/path/, back-end server URL is http://1.2.3.4:5678/. You want them both to work. You pass SCRIPT_NAME = '/sub/path/' from front-end server to back-end one. But when you access back-end server directly - there is no SCRIPT_NAME passed to WSGI/Django. So we cannot hard-code SCRIPT_NAME in Django settings because it's dynamic.
Pull-request created: ​https://github.com/django/django/pull/7000
At least documentation and additional tests look like they are required.
Absolutely agree with your remarks, Tim. I'll add tests. Could you point to docs that need to be updated?
I would like to take this ticket on and have a new PR for it: ​https://github.com/django/django/pull/10724
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/django/core/files/storage.py b/django/core/files/storage.py
index 4c27fce..65b2168 100755
--- a/django/core/files/storage.py
+++ b/django/core/files/storage.py
@@ -7,6 +7,7 @@ from django.core.exceptions import SuspiciousFileOperation
 from django.core.files import File, locks
 from django.core.files.move import file_move_safe
 from django.core.signals import setting_changed
+from django.urls import get_script_prefix
 from django.utils import timezone
 from django.utils._os import safe_join
 from django.utils.crypto import get_random_string
@@ -332,7 +333,11 @@ class FileSystemStorage(Storage):
         url = filepath_to_uri(name)
         if url is not None:
             url = url.lstrip('/')
-        return urljoin(self.base_url, url)
+        url = urljoin(self.base_url, url)
+        script_prefix = get_script_prefix().rstrip('/')
+        if script_prefix and url.startswith('/'):
+            url = script_prefix + url
+        return url
 
     def _datetime_from_timestamp(self, ts):
         """

diff --git a/django/template/context_processors.py b/django/template/context_processors.py
index 0e9efb2..9088cf4 100755
--- a/django/template/context_processors.py
+++ b/django/template/context_processors.py
@@ -63,18 +63,29 @@ def tz(request):
     return {'TIME_ZONE': timezone.get_current_timezone_name()}
 
 
+def _add_script_prefix(url):
+    """
+    Prepend the currently active script prefix to root-relative URLs.
+    """
+    from django.urls import get_script_prefix
+    script_prefix = get_script_prefix().rstrip('/')
+    if script_prefix and url.startswith('/'):
+        url = script_prefix + url
+    return url
+
+
 def static(request):
     """
     Add static-related context variables to the context.
     """
-    return {'STATIC_URL': settings.STATIC_URL}
+    return {'STATIC_URL': _add_script_prefix(settings.STATIC_URL)}
 
 
 def media(request):
     """
     Add media-related context variables to the context.
     """
-    return {'MEDIA_URL': settings.MEDIA_URL}
+    return {'MEDIA_URL': _add_script_prefix(settings.MEDIA_URL)}
 
 
 def request(request):

diff --git a/django/templatetags/static.py b/django/templatetags/static.py
index 39712aa..370b289 100755
--- a/django/templatetags/static.py
+++ b/django/templatetags/static.py
@@ -2,6 +2,7 @@ from urllib.parse import quote, urljoin
 
 from django import template
 from django.apps import apps
+from django.urls import get_script_prefix
 from django.utils.encoding import iri_to_uri
 from django.utils.html import conditional_escape
 
@@ -44,7 +45,17 @@ class PrefixNode(template.Node):
             prefix = ''
         else:
             prefix = iri_to_uri(getattr(settings, name, ''))
-        return prefix
+        return cls.add_script_prefix(prefix)
+
+    @staticmethod
+    def add_script_prefix(url):
+        """
+        Prepend the currently active script prefix to root-relative URLs.
+        """
+        script_prefix = get_script_prefix().rstrip('/')
+        if script_prefix and url.startswith('/'):
+            url = script_prefix + url
+        return url
 
     def render(self, context):
         prefix = self.handle_simple(self.name)
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/file_storage/tests.py b/tests/file_storage/tests.py
index 1c41760..16fcbf3 100755
--- a/tests/file_storage/tests.py
+++ b/tests/file_storage/tests.py
@@ -20,7 +20,7 @@ from django.db.models.fields.files import FileDescriptor
 from django.test import (
     LiveServerTestCase, SimpleTestCase, TestCase, override_settings,
 )
-from django.test.utils import requires_tz_support
+from django.test.utils import override_script_prefix, requires_tz_support
 from django.urls import NoReverseMatch, reverse_lazy
 from django.utils import timezone
 
@@ -325,6 +325,14 @@ class FileStorageTests(SimpleTestCase):
         """
         self.assertEqual(self.storage.url('test.file'), self.storage.base_url + 'test.file')
 
+    def test_file_url_with_script_prefix(self):
+        with override_script_prefix('/prefix/'):
+            self.assertEqual(
+                self.storage.url('test.file'),
+                '/prefix' + self.storage.base_url + 'test.file',
+            )
+
+    def test_file_url_special_chars(self):
         # should encode special chars except ~!*()'
         # like encodeURIComponent() JavaScript function do
         self.assertEqual(

diff --git a/tests/template_tests/syntax_tests/test_static.py b/tests/template_tests/syntax_tests/test_static.py
index 00f8cdb..d81891d 100755
--- a/tests/template_tests/syntax_tests/test_static.py
+++ b/tests/template_tests/syntax_tests/test_static.py
@@ -3,6 +3,7 @@ from urllib.parse import urljoin
 from django.conf import settings
 from django.template import TemplateSyntaxError
 from django.test import SimpleTestCase, override_settings
+from django.test.utils import override_script_prefix
 
 from ..utils import setup
 
@@ -64,6 +65,27 @@ class StaticTagTests(SimpleTestCase):
         output = self.engine.render_to_string('static-statictag05')
         self.assertEqual(output, urljoin(settings.STATIC_URL, '/static/special%3Fchars%26quoted.html'))
 
+    @setup({'static-statictag06': '{% load static %}{% static "admin/base.css" %}'})
+    def test_static_statictag_with_script_prefix(self):
+        with override_script_prefix('/script_name/'):
+            output = self.engine.render_to_string('static-statictag06')
+        self.assertEqual(
+            output,
+            '/script_name' + urljoin(settings.STATIC_URL, 'admin/base.css'),
+        )
+
+    @setup({'static-prefixtag06': '{% load static %}{% get_static_prefix %}'})
+    def test_static_prefixtag_with_script_prefix(self):
+        with override_script_prefix('/script_name/'):
+            output = self.engine.render_to_string('static-prefixtag06')
+        self.assertEqual(output, '/script_name' + settings.STATIC_URL)
+
+    @setup({'static-prefixtag07': '{% load static %}{% get_media_prefix %}'})
+    def test_media_prefixtag_with_script_prefix(self):
+        with override_script_prefix('/script_name/'):
+            output = self.engine.render_to_string('static-prefixtag07')
+        self.assertEqual(output, '/script_name' + settings.MEDIA_URL)
+
     @setup({'t': '{% load static %}{% static %}'})
     def test_static_statictag_without_path(self):
         msg = "'static' takes at least one argument (path to file)"
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/django/conf/__init__.py b/django/conf/__init__.py
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -15,7 +15,8 @@
 
 import django
 from django.conf import global_settings
-from django.core.exceptions import ImproperlyConfigured
+from django.core.exceptions import ImproperlyConfigured, ValidationError
+from django.core.validators import URLValidator
 from django.utils.deprecation import RemovedInDjango40Warning
 from django.utils.functional import LazyObject, empty
 
@@ -109,6 +110,26 @@ def configure(self, default_settings=global_settings, **options):
             setattr(holder, name, value)
         self._wrapped = holder
 
+    @staticmethod
+    def _add_script_prefix(value):
+        """
+        Add SCRIPT_NAME prefix to relative paths.
+
+        Useful when the app is being served at a subpath and manually prefixing
+        subpath to STATIC_URL and MEDIA_URL in settings is inconvenient.
+        """
+        # Don't apply prefix to valid URLs.
+        try:
+            URLValidator()(value)
+            return value
+        except (ValidationError, AttributeError):
+            pass
+        # Don't apply prefix to absolute paths.
+        if value.startswith('/'):
+            return value
+        from django.urls import get_script_prefix
+        return '%s%s' % (get_script_prefix(), value)
+
     @property
     def configured(self):
         """Return True if the settings have already been configured."""
@@ -128,6 +149,14 @@ def PASSWORD_RESET_TIMEOUT_DAYS(self):
             )
         return self.__getattr__('PASSWORD_RESET_TIMEOUT_DAYS')
 
+    @property
+    def STATIC_URL(self):
+        return self._add_script_prefix(self.__getattr__('STATIC_URL'))
+
+    @property
+    def MEDIA_URL(self):
+        return self._add_script_prefix(self.__getattr__('MEDIA_URL'))
+
 
 class Settings:
     def __init__(self, settings_module):
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/file_storage/tests.py b/tests/file_storage/tests.py
--- a/tests/file_storage/tests.py
+++ b/tests/file_storage/tests.py
@@ -521,7 +521,7 @@ def test_setting_changed(self):
         defaults_storage = self.storage_class()
         settings = {
             'MEDIA_ROOT': 'overridden_media_root',
-            'MEDIA_URL': 'overridden_media_url/',
+            'MEDIA_URL': '/overridden_media_url/',
             'FILE_UPLOAD_PERMISSIONS': 0o333,
             'FILE_UPLOAD_DIRECTORY_PERMISSIONS': 0o333,
         }
diff --git a/tests/settings_tests/tests.py b/tests/settings_tests/tests.py
--- a/tests/settings_tests/tests.py
+++ b/tests/settings_tests/tests.py
@@ -12,6 +12,7 @@
     override_settings, signals,
 )
 from django.test.utils import requires_tz_support
+from django.urls import clear_script_prefix, set_script_prefix
 
 
 @modify_settings(ITEMS={
@@ -567,3 +568,51 @@ def decorated_function():
         signals.setting_changed.disconnect(self.receiver)
         # This call shouldn't raise any errors.
         decorated_function()
+
+
+class MediaURLStaticURLPrefixTest(SimpleTestCase):
+    def set_script_name(self, val):
+        clear_script_prefix()
+        if val is not None:
+            set_script_prefix(val)
+
+    def test_not_prefixed(self):
+        # Don't add SCRIPT_NAME prefix to valid URLs, absolute paths or None.
+        tests = (
+            '/path/',
+            'http://myhost.com/path/',
+            None,
+        )
+        for setting in ('MEDIA_URL', 'STATIC_URL'):
+            for path in tests:
+                new_settings = {setting: path}
+                with self.settings(**new_settings):
+                    for script_name in ['/somesubpath', '/somesubpath/', '/', '', None]:
+                        with self.subTest(script_name=script_name, **new_settings):
+                            try:
+                                self.set_script_name(script_name)
+                                self.assertEqual(getattr(settings, setting), path)
+                            finally:
+                                clear_script_prefix()
+
+    def test_add_script_name_prefix(self):
+        tests = (
+            # Relative paths.
+            ('/somesubpath', 'path/', '/somesubpath/path/'),
+            ('/somesubpath/', 'path/', '/somesubpath/path/'),
+            ('/', 'path/', '/path/'),
+            # Invalid URLs.
+            ('/somesubpath/', 'htp://myhost.com/path/', '/somesubpath/htp://myhost.com/path/'),
+            # Blank settings.
+            ('/somesubpath/', '', '/somesubpath/'),
+        )
+        for setting in ('MEDIA_URL', 'STATIC_URL'):
+            for script_name, path, expected_path in tests:
+                new_settings = {setting: path}
+                with self.settings(**new_settings):
+                    with self.subTest(script_name=script_name, **new_settings):
+                        try:
+                            self.set_script_name(script_name)
+                            self.assertEqual(getattr(settings, setting), expected_path)
+                        finally:
+                            clear_script_prefix()
```

</details>

<details>
<summary>Verification command at step 38 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 262
Step: 38

COMMAND
python3 -m pytest tests/template_tests/syntax_tests/test_static.py tests/file_storage/tests.py -q 2>&1 | tail -20

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.351
stdout:
/opt/miniconda3/envs/testbed/bin/python3: No module named pytest
stderr:
[empty]
```

</details>

<details>
<summary>Verification command at step 39 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 267
Step: 39

COMMAND
cd tests && python3 runtests.py template_tests.syntax_tests.test_static file_storage 2>&1 | tail -25

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.641
stdout:
  File "/testbed/django/http/__init__.py", line 5, in <module>
    from django.http.response import (
  File "/testbed/django/http/response.py", line 15, in <module>
    from django.core.serializers.json import DjangoJSONEncoder
  File "/testbed/django/core/serializers/__init__.py", line 23, in <module>
    from django.core.serializers.base import SerializerDoesNotExist
  File "/testbed/django/core/serializers/base.py", line 7, in <mo... [truncated]
```

</details>


[Return to main report](../README.md)
