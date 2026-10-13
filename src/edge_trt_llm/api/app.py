"""FastAPI serving layer: the whole API in one file.

Why one file
------------
There is deliberately no service layer, no repository, no DI framework: the
contract (``Prediction``) and the inference logic (``create_detector``'s
backends) already live elsewhere, are tested there, and are the interesting
part. What is left here is *wiring*:

    lifespan  ->  build ONE detector at startup (TensorRT if this GPU can run
                  the engine, else ONNX — with a loud UserWarning)
    /predict          ->  bytes -> PIL -> detector.predict() -> Prediction
    /predict/preview  ->  same detections, boxes drawn, PNG back
    /describe         ->  /predict + optional VLM cross-check
    /health           ->  liveness + which model is loaded

Operational decisions encoded below:

* the session is built inside the lifespan, so a missing/corrupt artifact
  fails the server at boot instead of answering 500 to every request;
* the upload size is capped before Pillow is allowed to touch the bytes —
  a 2 GB body called "image.png" must not be able to exhaust our memory;
* a body that Pillow cannot decode is a client error (422), not a 500;
* the forward pass runs in a worker thread (ONNX CPU ~155 ms, TRT GPU ~4 ms),
  so a slow request cannot block the event loop for every other client;
* no VLM configured (no VLM_BASE_URL) is a NORMAL state: /describe still
  detects and reports vlm_status="disabled" — while a configured-but-broken
  VLM answers 502, never a silent "ok".
"""

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from edge_trt_llm.inference import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_ONNX_PATH,
    OnnxDetector,
    TrtDetector,
    create_detector,
    draw_boxes,
)
from edge_trt_llm.models.types import DescribeResponse, Prediction
from edge_trt_llm.vlm.providers import VlmError, create_vlm_provider

# Reject oversized uploads before Pillow allocates anything for them.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — enormous for a 640x480 JPEG


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model BEFORE serving: fail at boot, not on the first request.

    ``create_detector`` prefers the TensorRT engine and falls back to ONNX —
    loudly (UserWarning) — when engine/sidecar are missing or built for
    another GPU + TensorRT version: the common case on a dev box. Either way
    ONE instance is built and warmed here; per request it would cost more
    than the inference itself.
    """
    app.state.detector = create_detector(
        onnx_path=DEFAULT_ONNX_PATH, manifest_path=DEFAULT_MANIFEST_PATH
    )
    # None (no VLM_BASE_URL) is a normal, documented state — /describe then
    # answers vlm_status="disabled" instead of pretending the layer exists.
    app.state.vlm_provider = create_vlm_provider()
    try:
        yield
    finally:
        app.state.detector = None
        app.state.vlm_provider = None


app = FastAPI(
    title="edge-tensorrt-llm-pipeline",
    version="0.1.0",
    description="RF-DETR detection: TensorRT if engine fits this GPU, else ONNX Runtime.",
    lifespan=lifespan,
)


def _decode(payload: bytes) -> Image.Image:
    """Bytes -> decoded PIL image, or a 4xx. Decode now, not mid-inference."""
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image larger than {MAX_UPLOAD_BYTES // 1024 * 1024} MB",
        )
    if not payload:
        raise HTTPException(status_code=422, detail="empty body")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()  # force the decode: truncated/corrupt files fail HERE
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="body is not a decodable image") from exc
    return image


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe + which model is loaded (the benchmark table needs it)."""
    detector: OnnxDetector | TrtDetector = app.state.detector
    return {"status": "ok", "model_id": detector.model_id, "backend": detector.backend}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile) -> Prediction:
    """Detect objects in an uploaded image (multipart field name: ``file``)."""
    payload = await file.read()  # async I/O: off the event loop
    image = _decode(payload)

    detector: OnnxDetector | TrtDetector = app.state.detector
    # Thread-safety: an ONNX session allows concurrent run(); TrtDetector
    # serializes forward() with an internal lock — and postprocess only
    # reads state — so one shared detector serves every request.
    return await run_in_threadpool(detector.predict, image)


@app.post("/predict/preview")
async def predict_preview(file: UploadFile) -> Response:
    """Same detections as ``/predict``, but rendered on the image (PNG).

    Useful to eyeball an export from a client that cannot interpret JSON —
    and to compare two backends visually without downloading both JSONs.

    Encoding note: ``Image.tobytes()`` returns the RAW pixel buffer with no
    PNG header, so serving it as ``image/png`` gives the client a corrupt
    file. The only correct way is ``save(BytesIO, format=...)`` + ``getvalue``
    (no tempfile: the detector takes a PIL image in memory, not a path).
    """
    payload = await file.read()  # async I/O: off the event loop
    image = _decode(payload)

    detector: OnnxDetector | TrtDetector = app.state.detector
    prediction = await run_in_threadpool(detector.predict, image)
    # draw_boxes never mutates its input, so the JSON and the PNG are built
    # from the same pixels — and in either order.
    annotated = draw_boxes(image, prediction.detections)

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"X-Latency-Ms": str(prediction.latency_ms)},
    )


@app.post("/describe", response_model=DescribeResponse)
async def describe(file: UploadFile) -> DescribeResponse:
    """/predict's answer PLUS the VLM's opinion — the multimodal entry point.

    Flow (who calls whom): this route only calls ``provider.describe(...)``;
    prompt building and reply parsing are provider-internal details — the
    same way /predict never sees the detector's preprocessing.
    """
    payload = await file.read()  # async I/O: off the event loop
    image = _decode(payload)

    detector: OnnxDetector | TrtDetector = app.state.detector
    prediction = await run_in_threadpool(detector.predict, image)

    provider = app.state.vlm_provider
    if provider is None:
        return DescribeResponse(prediction=prediction, scene=None, vlm_status="disabled")

    # The VLM must SEE the boxes it is asked to judge, so annotate first.
    annotated = draw_boxes(image, prediction.detections)
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    try:
        scene = await run_in_threadpool(provider.describe, buffer.getvalue(), prediction)
    except VlmError as exc:
        # A configured-but-broken cross-check is loud: 502, never "ok".
        raise HTTPException(status_code=502, detail=f"VLM call failed: {exc}") from exc
    return DescribeResponse(prediction=prediction, scene=scene, vlm_status="ok")
