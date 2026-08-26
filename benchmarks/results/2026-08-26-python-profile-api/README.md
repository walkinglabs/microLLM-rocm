# Python profile decorator and context manager

`@profile(output=...)` and `profile_scope(...)` append schema-versioned JSONL spans.
Sync, async, nested, metadata, and error paths use only the Python standard library
and work without loading the C shared library.

![Python profile API](python-profile.svg)
