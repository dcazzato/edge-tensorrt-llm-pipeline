# CPU serving image — ONNX Runtime backend only.
# TensorRT is deliberately NOT in this image: engines are per-GPU build
# artifacts and need a CUDA base + a GPU at runtime (docs/RESULTS.md).
FROM python:3.12-slim

# uv creates /app/.venv; the uv binary stays available for container debugging.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Layer order follows dependency weight: metadata first (cache-friendly),
# then code, then the heavy, rarely-changing model artifact.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra api --no-dev

# The model is not in git (*.onnx is gitignored): build this image AFTER
# running scripts/export_onnx.py locally.
COPY artifacts/model.onnx artifacts/model.onnx.data artifacts/manifest.json ./artifacts/

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

# /health exists exactly for this: probes cost one stdlib import, no curl needed.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "edge_trt_llm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
