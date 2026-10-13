"""The API layer is wiring, so these tests pin the wiring.

Inference correctness is NOT re-tested here — ``test_inference.py`` owns that.
What matters at this layer: boot fails loudly when the artifact is missing,
``/predict`` returns a valid ``Prediction``, ``/predict/preview`` returns a
real PNG with the boxes drawn, and both answer 4xx for bad input instead of
500.
"""

import io
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import edge_trt_llm.api.app as app_module
from edge_trt_llm.api.app import MAX_UPLOAD_BYTES, app
from edge_trt_llm.inference import DEFAULT_MANIFEST_PATH, DEFAULT_ONNX_PATH
from edge_trt_llm.models.types import Prediction, SceneSummary
from edge_trt_llm.vlm.providers import VlmError

MODEL_ID = "Roboflow/rf-detr-nano"
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def _sample_jpeg() -> bytes:
    """The docs sample image as JPEG bytes — the same one export/parity use."""
    image = Image.open(io.BytesIO(urllib.request.urlopen(SAMPLE_URL).read()))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Booted app: entering the context runs the lifespan (loads the model).

    VLM env vars are cleared so the disabled-vs-configured behaviour under
    test cannot depend on whatever the developer machine happens to export.
    """
    if not DEFAULT_ONNX_PATH.exists() or not DEFAULT_MANIFEST_PATH.exists():
        pytest.skip("run `uv run python scripts/export_onnx.py` first")
    saved = {key: os.environ.pop(key, None) for key in ("VLM_BASE_URL", "VLM_MODEL")}
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        os.environ.update({key: value for key, value in saved.items() if value is not None})


def test_boot_fails_loudly_when_manifest_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing artifact must kill the boot, not 500 every request forever."""
    monkeypatch.setattr(app_module, "DEFAULT_MANIFEST_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        with TestClient(app):
            pass


def test_health_reports_loaded_model(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "model_id": MODEL_ID,
        "backend": "onnxruntime",
    }


def test_predict_returns_valid_prediction(client: TestClient) -> None:
    response = client.post(
        "/predict", files={"file": ("sample.jpg", _sample_jpeg(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["backend"] == "onnxruntime"
    assert body["model_id"] == MODEL_ID
    assert body["image_size"] == [640, 480]  # (width, height), JSON turns it into a list
    assert body["latency_ms"] > 0
    # the sample is known to hold 2 cats + 1 remote above the 0.5 threshold
    assert len(body["detections"]) >= 3
    assert {"cat", "remote"} <= {d["label"] for d in body["detections"]}
    for det in body["detections"]:
        assert 0.0 <= det["score"] <= 1.0
        x1, y1, x2, y2 = det["bbox"]
        assert x1 <= x2 and y1 <= y2


def test_predict_requires_file_field(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_predict_rejects_non_image_body(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 422
    assert "decodable" in response.json()["detail"]


def test_predict_rejects_empty_body(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("x.png", b"", "image/png")})
    assert response.status_code == 422


def test_predict_rejects_oversized_body(client: TestClient) -> None:
    """The size cap fires BEFORE Pillow allocates anything for the body."""
    response = client.post(
        "/predict", files={"file": ("big.png", b"0" * (MAX_UPLOAD_BYTES + 1), "image/png")}
    )
    assert response.status_code == 413


# --- /predict/preview: same detections, rendered ---------------------------


def test_preview_returns_a_real_png(client: TestClient) -> None:
    response = client.post(
        "/predict/preview", files={"file": ("sample.jpg", _sample_jpeg(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    # PNG signature: proves an ENCODED image, not Image.tobytes() raw pixels
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert float(response.headers["x-latency-ms"]) > 0

    annotated = Image.open(io.BytesIO(response.content))
    assert annotated.format == "PNG"
    assert annotated.size == (640, 480)  # preview keeps the original geometry


def test_preview_actually_draws_the_boxes(client: TestClient) -> None:
    """Pure red (255, 0, 0) comes only from draw_boxes' outline='red'.

    JPEG pixels are lossy and essentially never land on exact red, so an
    exact match proves the rectangles were drawn rather than guessed.
    """
    response = client.post(
        "/predict/preview", files={"file": ("sample.jpg", _sample_jpeg(), "image/jpeg")}
    )
    pixels = np.asarray(Image.open(io.BytesIO(response.content)))
    exact_red = (pixels[..., 0] == 255) & (pixels[..., 1] == 0) & (pixels[..., 2] == 0)
    assert exact_red.sum() >= 100, "no box outline found — draw_boxes did not run"


def test_preview_rejects_non_image_body(client: TestClient) -> None:
    response = client.post(
        "/predict/preview", files={"file": ("x.png", b"not an image", "image/png")}
    )
    assert response.status_code == 422
    assert "decodable" in response.json()["detail"]


def test_preview_rejects_oversized_body(client: TestClient) -> None:
    """The preview shares _decode, hence the same cap as /predict."""
    response = client.post(
        "/predict/preview", files={"file": ("big.png", b"0" * (MAX_UPLOAD_BYTES + 1), "image/png")}
    )
    assert response.status_code == 413


# --- /describe: detect + optional VLM cross-check -----------------------------


class _FakeVlm:
    """Two-line stub honoring the VlmProvider Protocol — no model needed."""

    def describe(self, annotated_png: bytes, prediction: Prediction) -> SceneSummary:
        # the endpoint must send a REAL annotated PNG, not the raw upload
        assert annotated_png.startswith(b"\x89PNG")
        assert prediction.detections, "stub expects the real detector output"
        return SceneSummary(
            description="Two cats and a remote on a couch.",
            checks_out=True,
            provider="fake",
            model_id="fake-model",
            latency_ms=1.0,
        )


class _BrokenVlm:
    def describe(self, annotated_png: bytes, prediction: Prediction) -> SceneSummary:
        raise VlmError("connection refused")


def test_describe_without_vlm_reports_disabled(client: TestClient) -> None:
    """No provider configured: detection stays fully valid, scene=null."""
    response = client.post(
        "/describe", files={"file": ("sample.jpg", _sample_jpeg(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["vlm_status"] == "disabled"
    assert body["scene"] is None
    assert len(body["prediction"]["detections"]) >= 3


def test_describe_returns_scene_when_provider_present(client: TestClient) -> None:
    app.state.vlm_provider = _FakeVlm()
    try:
        response = client.post(
            "/describe", files={"file": ("sample.jpg", _sample_jpeg(), "image/jpeg")}
        )
    finally:
        app.state.vlm_provider = None
    assert response.status_code == 200
    body = response.json()
    assert body["vlm_status"] == "ok"
    assert body["scene"]["checks_out"] is True
    assert body["scene"]["model_id"] == "fake-model"
    # /describe is a strict superset: the /predict payload is embedded whole
    assert len(body["prediction"]["detections"]) >= 3


def test_describe_maps_broken_vlm_to_502(client: TestClient) -> None:
    """Configured-but-broken must be loud: 502, never a fake 'ok'."""
    app.state.vlm_provider = _BrokenVlm()
    try:
        response = client.post(
            "/describe", files={"file": ("sample.jpg", _sample_jpeg(), "image/jpeg")}
        )
    finally:
        app.state.vlm_provider = None
    assert response.status_code == 502
    assert "connection refused" in response.json()["detail"]
