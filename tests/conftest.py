"""Test wiring: skip whole files whose optional extras are not installed.

Why this file exists
--------------------
``tests/test_inference.py`` imports ``edge_trt_llm.inference`` at module level,
and that module in turn imports ``onnxruntime`` and ``PIL`` at module level.
With a bare ``uv sync`` (no extras: only pydantic + numpy) that import fails
DURING COLLECTION — before pytest executes a single test body — so the whole
file is reported as an error and the suite goes red for a reason that has
nothing to do with a real bug.

Ignoring the file here turns that into a smaller-but-green suite: those tests
simply do not exist until the matching extra is installed
(``uv sync --extra api`` or ``--extra export``).

Note: the ``try/except ImportError`` blocks inside the test files solve a
*different* problem (individual tests guarded by ``skipif``). They cannot
help here, because the failure happens before any test code can run.
"""

collect_ignore: list[str] = []

try:
    import onnxruntime  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError:
    collect_ignore.append("test_inference.py")

# test_api.py imports fastapi + the detector (hence onnxruntime): without the
# `api` extra it would fail at collection for the same reason as above.
try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
except ImportError:
    collect_ignore.append("test_api.py")
