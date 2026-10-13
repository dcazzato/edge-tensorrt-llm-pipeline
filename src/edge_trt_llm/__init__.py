"""Edge AI pipeline: HF detection model -> ONNX/TensorRT -> FastAPI -> VLM layer.

This package follows the ``src/`` layout: it is only importable once
installed (``uv sync``), so tests always exercise the installed code,
never a stray directory on ``sys.path``.
"""

# NOTE: duplicated with [project].version in pyproject.toml — single-source
# wiring (importlib.metadata) is deferred on purpose.
__version__ = "0.1.0"

__all__ = ["__version__"]
