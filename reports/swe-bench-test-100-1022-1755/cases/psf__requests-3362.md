# psf__requests-3362: A JSON header default bypasses the failing streaming-decoder branch

**Primary category:** R — Symptom fix; underlying cause remains. **Confidence:** high.

**Recorded score:** F2P 0/1; P2P 75/75. **Steps:** 15/40. Counts are harness classifications, subject to the report's parser caveats.

The fix is specific to the example MIME type and leaves encoding=None streaming behavior unchanged.

## Issue requirement

iter_content(decode_unicode=True) returns bytes while response.text returns Unicode; application/json is the motivating example.

## Agent change

Defaults application/json headers to UTF-8 in get_encoding_from_headers.

## Golden change

Changes stream_decode_response_unicode to fall back to apparent_encoding when encoding is None and handle decoder lookup failures.

## Observed failure

A Response with raw bytes and encoding=None still yields bytes under decode_unicode=True. The single F2P test fails; all 75 P2P tests pass.

## Scope and attribution

This is a symptom-specific fix in the same file as gold, demonstrating why file overlap is not enough to assess root-cause localization. The behavioral invariant belongs to decoding, not only JSON header interpretation.

## Verification in the trajectory

The agent adds a header-parsing test and runs the old streaming test, which only covered known encodings. It never adds the decisive unknown-encoding response case.

## What would improve the outcome

Trace the difference between text and streaming decoding, then test known, absent and invalid encodings plus split multibyte input at the public API.

## Source evidence

The excerpts below come from local run artifacts. Dataset hints are included for retrospective context only; the solving objective contained the problem statement, not hints or golden patches.

| Artifact | Source | JSONL row |
|---|---|---:|
| dataset | [swe_bench_lite_test_100.jsonl](../../../swe_bench_lite_test_100.jsonl) | 53 |
| prediction | [predictions.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.jsonl) | 53 |
| metrics | [predictions.metrics.jsonl](../../../.agents/evals/test-100-1022-1755/predictions.metrics.jsonl) | 53 |
| report | [report.json](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/psf__requests-3362/report.json) | — |
| test_output | [test_output.txt](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/psf__requests-3362/test_output.txt) | — |
| harness_log | [run_instance.log](../../../.agents/evals/test-100-1022-1755/logs/run_evaluation/test-100-1022-1755/deepseek__deepseek-v4-flash/psf__requests-3362/run_instance.log) | — |
| trajectory | [psf__requests-3362.jsonl](../../../.agents/evals/test-100-1022-1755/trajectories/events/psf__requests-3362.jsonl) | — |

<details>
<summary>Supplied problem statement</summary>

```text
Uncertain about content/text vs iter_content(decode_unicode=True/False)
When requesting an application/json document, I'm seeing `next(r.iter_content(16*1024, decode_unicode=True))` returning bytes, whereas `r.text` returns unicode. My understanding was that both should return a unicode object. In essence, I thought "iter_content" was equivalent to "iter_text" when decode_unicode was True. Have I misunderstood something? I can provide an example if needed.

For reference, I'm using python 3.5.1 and requests 2.10.0.

Thanks!
```

</details>

<details>
<summary>Dataset hints — not supplied in this run</summary>

```text
what does (your response object).encoding return?

There's at least one key difference: `decode_unicode=True` doesn't fall back to `apparent_encoding`, which means it'll never autodetect the encoding. This means if `response.encoding` is None it is a no-op: in fact, it's a no-op that yields bytes.

That behaviour seems genuinely bad to me, so I think we should consider it a bug. I'd rather we had the same logic as in `text` for this.

`r.encoding` returns `None`.

On a related note, `iter_text` might be clearer/more consistent than `iter_content(decode_unicode=True)` if there's room for change in the APIs future (and `iter_content_lines` and `iter_text_lines` I guess), assuming you don't see that as bloat.

@mikepelley The API is presently frozen so I don't think we'll be adding those three methods. Besides, `iter_text` likely wouldn't provide much extra value outside of calling `iter_content(decode_unicode=True)`.
```

</details>

<details>
<summary>Submitted source patch</summary>

```text

diff --git a/requests/utils.py b/requests/utils.py
index 8d17b6b..23f7848 100755
--- a/requests/utils.py
+++ b/requests/utils.py
@@ -355,6 +355,11 @@ def get_encoding_from_headers(headers):
     if 'text' in content_type:
         return 'ISO-8859-1'
 
+    if 'application/json' in content_type:
+        # Assume UTF-8 based on RFC 4627: https://www.ietf.org/rfc/rfc4627.txt
+        # since the charset was unset.
+        return 'utf-8'
+
 
 def stream_decode_response_unicode(iterator, r):
     """Stream decodes a iterator."""
```

</details>

<details>
<summary>Submitted test patch</summary>

```text
diff --git a/tests/test_utils.py b/tests/test_utils.py
index 17149d2..4e511a5 100755
--- a/tests/test_utils.py
+++ b/tests/test_utils.py
@@ -364,6 +364,10 @@ def test_parse_dict_header(value, expected):
             CaseInsensitiveDict({'content-type': 'application/json; charset=utf-8'}),
             'utf-8'
         ),
+        (
+            CaseInsensitiveDict({'content-type': 'application/json'}),
+            'utf-8'
+        ),
         (
             CaseInsensitiveDict({'content-type': 'text/plain'}),
             'ISO-8859-1'
```

</details>

<details>
<summary>Golden source patch</summary>

```diff
diff --git a/requests/utils.py b/requests/utils.py
--- a/requests/utils.py
+++ b/requests/utils.py
@@ -358,13 +358,20 @@ def get_encoding_from_headers(headers):
 
 def stream_decode_response_unicode(iterator, r):
     """Stream decodes a iterator."""
+    encoding = r.encoding
 
-    if r.encoding is None:
-        for item in iterator:
-            yield item
-        return
+    if encoding is None:
+        encoding = r.apparent_encoding
+
+    try:
+        decoder = codecs.getincrementaldecoder(encoding)(errors='replace')
+    except (LookupError, TypeError):
+        # A LookupError is raised if the encoding was not found which could
+        # indicate a misspelling or similar mistake.
+        #
+        # A TypeError can be raised if encoding is None
+        raise UnicodeError("Unable to decode contents with encoding %s." % encoding)
 
-    decoder = codecs.getincrementaldecoder(r.encoding)(errors='replace')
     for chunk in iterator:
         rv = decoder.decode(chunk)
         if rv:
```

</details>

<details>
<summary>Golden test patch</summary>

```diff
diff --git a/tests/test_requests.py b/tests/test_requests.py
--- a/tests/test_requests.py
+++ b/tests/test_requests.py
@@ -980,6 +980,13 @@ def test_response_decode_unicode(self):
         chunks = r.iter_content(decode_unicode=True)
         assert all(isinstance(chunk, str) for chunk in chunks)
 
+        # check for encoding value of None
+        r = requests.Response()
+        r.raw = io.BytesIO(b'the content')
+        r.encoding = None
+        chunks = r.iter_content(decode_unicode=True)
+        assert all(isinstance(chunk, str) for chunk in chunks)
+
     def test_response_chunk_size_int(self):
         """Ensure that chunk_size is passed as an integer, otherwise
         raise a TypeError.
```

</details>

<details>
<summary>Verification command at step 8 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 62
Step: 8

COMMAND
python -m pytest tests/test_utils.py::test_get_encoding_from_headers -q

STORED OUTPUT PREVIEW
exit_code: 1
timed_out: false
duration_seconds: 0.664
stdout:
..F.                                                                     [100%]
=================================== FAILURES ===================================
_________________ test_get_encoding_from_headers[value2-utf-8] _________________

value = {'content-type': 'application/json'}, expected = 'utf-8'

    @pytest.mark.parametrize(
        'value, expected', (
            (
                CaseInsensitiveDict(),
                N... [truncated]
```

</details>

<details>
<summary>Verification command at step 11 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 81
Step: 11

COMMAND
python -m pytest tests/test_utils.py::test_get_encoding_from_headers -q

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.492
stdout:
....                                                                     [100%]
=============================== warnings summary ===============================
requests/packages/urllib3/_collections.py:2
requests/packages/urllib3/_collections.py:2
  /testbed/requests/packages/urllib3/_collections.py:2: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, ... [truncated]
```

</details>

<details>
<summary>Verification command at step 11 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 83
Step: 11

COMMAND
python -m pytest tests/test_requests.py::TestRequests::test_response_decode_unicode -q

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.550
stdout:
.                                                                        [100%]
=============================== warnings summary ===============================
requests/packages/urllib3/_collections.py:2
requests/packages/urllib3/_collections.py:2
  /testbed/requests/packages/urllib3/_collections.py:2: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, ... [truncated]
```

</details>

<details>
<summary>Verification command at step 12 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 89
Step: 12

COMMAND
python -m pytest tests/test_utils.py -q

STORED OUTPUT PREVIEW
exit_code: 2
timed_out: false
duration_seconds: 0.555
stdout:

==================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_utils.py _____________________
/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pluggy/_hooks.py:265: in __call__
    return self._hookexec(self.name, self.get_hookimpls(), kwargs, firstresult)
/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pluggy/_manager.py:80: in _hookexec
    return ... [truncated]
```

</details>

<details>
<summary>Verification command at step 13 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 99
Step: 13

COMMAND
python -m pytest tests/test_requests.py -q -k "response_decode_unicode or response_chunk_size_int"

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.501
stdout:
..                                                                       [100%]
=============================== warnings summary ===============================
requests/packages/urllib3/_collections.py:2
requests/packages/urllib3/_collections.py:2
  /testbed/requests/packages/urllib3/_collections.py:2: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, ... [truncated]
```

</details>

<details>
<summary>Verification command at step 14 (stored preview may be truncated)</summary>

```text
Trajectory JSONL row: 104
Step: 14

COMMAND
python -m py_compile requests/utils.py && python -m pytest tests/test_utils.py::test_get_encoding_from_headers tests/test_requests.py::TestRequests::test_response_decode_unicode tests/test_requests.py::TestRequests::test_response_chunk_size_int -q

STORED OUTPUT PREVIEW
exit_code: 0
timed_out: false
duration_seconds: 0.580
stdout:
......                                                                   [100%]
=============================== warnings summary ===============================
requests/packages/urllib3/_collections.py:2
requests/packages/urllib3/_collections.py:2
  /testbed/requests/packages/urllib3/_collections.py:2: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, ... [truncated]
```

</details>


[Return to main report](../README.md)
