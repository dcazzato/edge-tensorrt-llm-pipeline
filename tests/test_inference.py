"""Pin ``edge_trt_llm.inference`` against the Hugging Face reference.

Two groups, deliberately kept apart:

* **Parity tests** (need the ``export`` extra): run the SAME input through our
  numpy implementation and through ``transformers``, then assert the numbers
  agree. This is what makes dropping ``transformers`` from the API safe — if a
  future edit breaks the resize convention or the box maths, these fail
  instead of silently serving shifted detections.
* **Contract tests** (no ML deps): shapes, dtypes, the ONNX feed contract,
  empty-detection semantics, error paths.
* **Engine validation tests** (no GPU needed): the compatibility
  chain must reject a wrong engine on a CPU-only dev box, and the ONNX
  fallback must be loud — a silent backend downgrade burned one benchmark
  run (docs/RESULTS.md, ONNX Runtime's CPU fallback).
"""

import io
import json
import sys
import urllib.request
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from edge_trt_llm.inference import (
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_SIZE,
    OUTPUT_NAMES,
    draw_boxes,
    engine_incompatibility,
    engine_meta_path,
    postprocess,
    preprocess,
)
from edge_trt_llm.models.types import Detection

try:  # pragma: no cover - depends on the installed extra
    import onnxruntime

    HAS_ORT = True
except ImportError:  # pragma: no cover
    HAS_ORT = False

try:  # pragma: no cover
    import torch
    from transformers import AutoImageProcessor, RfDetrForObjectDetection

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAS_TRANSFORMERS = False

MODEL_ID = "Roboflow/rf-detr-nano"
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

needs_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="needs the export extra (transformers, torch)"
)
needs_ort = pytest.mark.skipif(not HAS_ORT, reason="needs the api/export extra (onnxruntime)")


def _sample_image() -> Image.Image:
    """The docs sample image (cats + remote)."""
    return Image.open(io.BytesIO(urllib.request.urlopen(SAMPLE_URL).read()))


# --- contract tests: no ML dependencies -------------------------------------


def test_preprocess_shapes_and_dtypes() -> None:
    """The ONNX feed contract: NCHW float32 values + int64 mask, batch of 1."""
    feed = preprocess(Image.new("RGB", (64, 48), (10, 20, 30)))

    assert set(feed) == {"pixel_values", "pixel_mask"}
    assert feed["pixel_values"].dtype == np.float32
    assert feed["pixel_values"].shape == (1, 3, *INPUT_SIZE)
    assert feed["pixel_mask"].dtype == np.int64
    assert feed["pixel_mask"].shape == (1, *INPUT_SIZE)
    # single image, no padding -> every pixel is valid
    assert feed["pixel_mask"].all()


def test_preprocess_converts_non_rgb_input() -> None:
    """FastAPI can hand us RGBA/grayscale; the graph only accepts 3 channels."""
    feed = preprocess(Image.new("L", (32, 32), 128))
    assert feed["pixel_values"].shape == (1, 3, *INPUT_SIZE)


def test_preprocess_normalisation_constants_match_imagenet() -> None:
    """Guard against a typo in the module constants (they feed the model)."""
    assert IMAGE_MEAN == (0.485, 0.456, 0.406)
    assert IMAGE_STD == (0.229, 0.224, 0.225)
    assert INPUT_SIZE == (384, 384)


def test_postprocess_returns_empty_list_below_threshold() -> None:
    """An empty detection list is a valid prediction, not an error."""
    logits = np.full((300, 91), -10.0, dtype=np.float32)  # sigmoid ~ 0
    boxes = np.zeros((300, 4), dtype=np.float32)
    boxes[:, 2:] = 0.1  # w, h > 0 keeps corners ordered

    assert postprocess(logits, boxes, (640, 480), labels={0: "N/A"}) == []


def test_postprocess_threshold_is_strict_and_maps_labels() -> None:
    """``score > threshold`` (strict, like HF) and class_id -> label lookup."""
    logits = np.full((300, 91), -10.0, dtype=np.float32)
    logits[7, 17] = 0.0  # sigmoid(0) == 0.5 exactly -> NOT > 0.5
    boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.1], dtype=np.float32), (300, 1))

    strict = postprocess(logits, boxes, (640, 480), labels={17: "cat"}, threshold=0.5)
    assert strict == [], "score == threshold must be excluded (HF uses strict >)"

    relaxed = postprocess(logits, boxes, (640, 480), labels={17: "cat"}, threshold=0.4)
    assert len(relaxed) == 1
    assert relaxed[0].class_id == 17
    assert relaxed[0].label == "cat"
    assert relaxed[0].score == pytest.approx(0.5)


def test_postprocess_scales_boxes_to_original_pixels() -> None:
    """pred_boxes are normalised to the resized square; output is in pixels."""
    logits = np.full((300, 91), -10.0, dtype=np.float32)
    logits[0, 5] = 10.0
    # cx=0.5, cy=0.5, w=0.25, h=0.5 on a 640x480 image -> centred box
    boxes = np.tile(np.array([0.5, 0.5, 0.25, 0.5], dtype=np.float32), (300, 1))

    [det] = postprocess(logits, boxes, (640, 480), labels={5: "car"})

    assert det.bbox == pytest.approx((240.0, 120.0, 400.0, 360.0))
    # sigmoid(10) = 0.9999546 — abs=1e-4 accommodates it without going limp
    assert det.score == pytest.approx(1.0, abs=1e-4)


# --- draw_boxes: shared by export_onnx and /predict/preview ----------------

_DETECTION = Detection(bbox=(10.0, 20.0, 100.0, 80.0), score=0.9, label="cat", class_id=17)


def test_draw_boxes_does_not_mutate_its_input() -> None:
    """The API infers on the very image it later renders — keep it pristine."""
    image = Image.new("RGB", (160, 120), (7, 8, 9))
    before = image.tobytes()

    annotated = draw_boxes(image, [_DETECTION])

    assert annotated is not image, "must return a copy, not the input"
    assert image.tobytes() == before, "input pixels were modified"
    assert annotated.size == image.size


def test_draw_boxes_returns_rgb_for_non_rgb_input() -> None:
    """Yellow text on mode ``L`` is meaningless: normalise to RGB first."""
    for mode, size, colour in (("L", (160, 120), 42), ("RGBA", (160, 120), (1, 2, 3, 4))):
        annotated = draw_boxes(Image.new(mode, size, colour), [_DETECTION])
        assert annotated.mode == "RGB", f"mode {mode} was not converted to RGB"
        assert annotated.size == size


def test_draw_boxes_handles_an_empty_detection_list() -> None:
    """No detections is valid — still returns a usable copy of the image."""
    image = Image.new("RGB", (64, 48), (0, 0, 0))
    annotated = draw_boxes(image, [])
    assert annotated.mode == "RGB"
    assert annotated.size == image.size


def test_draw_boxes_renders_box_and_label() -> None:
    """Both the outline and the text must land on the pixels.

    NOTE: the label is NOT counted as pure (255, 255, 0) — Pillow's default
    font is a FreeType font, so the glyphs are anti-aliased and come out as a
    blend. Assert on "close enough" instead, or this test lies about passing.
    """
    annotated = np.asarray(draw_boxes(Image.new("RGB", (160, 120), (0, 0, 0)), [_DETECTION]))

    pure_red = (annotated[..., 0] == 255) & (annotated[..., 1] == 0) & (annotated[..., 2] == 0)
    near_yellow = (
        (annotated[..., 0] > 150) & (annotated[..., 1] > 150) & (annotated[..., 2] < 80)
    )
    assert pure_red.sum() > 100, "box outline missing"
    assert near_yellow.sum() > 20, "label text missing"


def test_draw_boxes_output_is_a_valid_encodable_image() -> None:
    """save(BytesIO) must round-trip: this is the tobytes() trap, tested."""
    buffer = io.BytesIO()
    draw_boxes(Image.new("RGB", (64, 48), (3, 3, 3)), [_DETECTION]).save(
        buffer, format="PNG"
    )
    reloaded = Image.open(io.BytesIO(buffer.getvalue()))
    assert reloaded.format == "PNG"
    assert reloaded.size == (64, 48)


# --- parity tests: our numpy code vs the Hugging Face reference -------------


@needs_transformers
def test_preprocess_matches_huggingface_reference() -> None:
    """Same PIL image through both pipelines -> tensors agree within 1e-5."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    image = _sample_image()

    reference = processor(images=image, return_tensors="np")
    ours = preprocess(image)

    delta = float(np.abs(ours["pixel_values"] - reference["pixel_values"]).max())
    assert delta < 1e-5, f"pixel_values diverge from HF reference: {delta:.3e}"
    assert np.array_equal(ours["pixel_mask"], reference["pixel_mask"]), "pixel_mask differs"


@needs_transformers
def test_postprocess_matches_huggingface_reference() -> None:
    """Same model outputs through both decoders -> identical detections."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = RfDetrForObjectDetection.from_pretrained(MODEL_ID).eval().to("cpu")
    image = _sample_image()
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    reference = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=torch.tensor([image.size[::-1]])
    )[0]
    ours = postprocess(
        outputs.logits[0].numpy(),
        outputs.pred_boxes[0].numpy(),
        image.size,
        labels=model.config.id2label,
    )

    assert len(ours) == len(reference["scores"]) > 0, "detection count diverges"
    np.testing.assert_allclose(
        [d.score for d in ours], reference["scores"].numpy(), atol=1e-5, rtol=1e-4
    )
    assert [d.class_id for d in ours] == reference["labels"].tolist()
    np.testing.assert_allclose(
        [d.bbox for d in ours], reference["boxes"].numpy(), atol=0.01, rtol=1e-5
    )


@needs_transformers
def test_end_to_end_detection_parity() -> None:
    """The number that actually matters: our preprocess -> model -> same boxes."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = RfDetrForObjectDetection.from_pretrained(MODEL_ID).eval().to("cpu")
    image = _sample_image()
    reference_inputs = processor(images=image, return_tensors="pt")
    ours = preprocess(image)

    with torch.inference_mode():
        out_ref = model(**reference_inputs)
        out_ours = model(
            pixel_values=torch.from_numpy(ours["pixel_values"]),
            pixel_mask=torch.from_numpy(ours["pixel_mask"]),
        )

    assert (out_ref.logits - out_ours.logits).abs().max() < 5e-3
    assert (out_ref.pred_boxes - out_ours.pred_boxes).abs().max() < 5e-3

    target = torch.tensor([image.size[::-1]])
    det_ref = processor.post_process_object_detection(
        out_ref, threshold=0.5, target_sizes=target
    )[0]
    det_ours = processor.post_process_object_detection(
        out_ours, threshold=0.5, target_sizes=target
    )[0]
    assert len(det_ref["scores"]) == len(det_ours["scores"])
    np.testing.assert_allclose(det_ours["scores"], det_ref["scores"], atol=1e-4)


# --- OnnxDetector: needs the exported artifact ------------------------------


@needs_ort
def test_detector_requires_manifest(tmp_path: Path) -> None:
    """Missing manifest must fail at construction, not at first predict."""
    from edge_trt_llm.inference import OnnxDetector

    with pytest.raises(FileNotFoundError, match="export_onnx"):
        OnnxDetector(Path("artifacts/model.onnx"), tmp_path / "missing.json", warmup=False)


@needs_ort
def test_detector_predict_returns_contract() -> None:
    """Full path on the real artifact: a valid Prediction with our backend."""
    from edge_trt_llm.inference import DEFAULT_MANIFEST_PATH, DEFAULT_ONNX_PATH, OnnxDetector

    if not DEFAULT_ONNX_PATH.exists() or not DEFAULT_MANIFEST_PATH.exists():
        pytest.skip("run `uv run python scripts/export_onnx.py` first")

    image = _sample_image()
    detector = OnnxDetector(DEFAULT_ONNX_PATH, DEFAULT_MANIFEST_PATH, warmup=False)
    prediction = detector.predict(image)

    assert prediction.backend == "onnxruntime"
    assert prediction.model_id == MODEL_ID
    assert prediction.image_size == image.size
    assert prediction.latency_ms > 0
    # the docs sample is known to contain cats + remote above 0.5
    assert len(prediction.detections) >= 3
    assert all(0.0 <= d.score <= 1.0 for d in prediction.detections)
    assert {d.label for d in prediction.detections} >= {"cat", "remote"}


@needs_ort
def test_detector_output_names_match_export() -> None:
    """The session must expose exactly the outputs the pipeline consumes."""
    from edge_trt_llm.inference import DEFAULT_ONNX_PATH

    if not DEFAULT_ONNX_PATH.exists():
        pytest.skip("run `uv run python scripts/export_onnx.py` first")

    session = onnxruntime.InferenceSession(
        str(DEFAULT_ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    available = {o.name for o in session.get_outputs()}
    assert set(OUTPUT_NAMES) <= available


def test_postprocess_rejects_shape_mismatch() -> None:
    """Loud failure at the boundary instead of a silent wrong box."""
    with pytest.raises(ValueError, match="shape mismatch"):
        postprocess(
            np.zeros((300, 91), dtype=np.float32),
            np.zeros((299, 4), dtype=np.float32),
            (640, 480),
            labels={},
        )


def test_postprocess_unknown_class_fails_loudly() -> None:
    """A class id absent from the manifest means a stale manifest, not a skip."""
    logits = np.full((300, 91), -10.0, dtype=np.float32)
    logits[0, 42] = 10.0
    boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32), (300, 1))

    with pytest.raises(KeyError, match="missing from manifest"):
        postprocess(logits, boxes, (640, 480), labels={17: "cat"})


# --- Engine validation chain (no GPU, no tensorrt) ----------------------------
# Everything before the CUDA hand-off must run on a CPU-only dev box, and the
# fallback must warn: a silent backend downgrade burned one benchmark run
# (docs/RESULTS.md, ONNX Runtime's CPU fallback).


def _write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model_id": MODEL_ID, "labels": {"0": "N/A"}}))
    return manifest


def _write_engine(tmp_path: Path) -> Path:
    engine = tmp_path / "model_fp16.engine"
    engine.write_bytes(b"pretend-this-is-an-engine")
    return engine


def _write_sidecar(engine: Path, **overrides: object) -> None:
    meta = {
        "gpu": "Tesla T4",
        "tensorrt": "11.4.0.106",
        "model_id": MODEL_ID,
        "inputs": {"pixel_values": [1, 3, 384, 384], "pixel_mask": [1, 384, 384]},
        "outputs": {"logits": [1, 576, 91], "pred_boxes": [1, 576, 4]},
    }
    meta.update(overrides)
    engine_meta_path(engine).write_text(json.dumps(meta))


def test_engine_meta_path_glues_sidecar_to_engine() -> None:
    """One config file less: the sidecar name derives from the engine name."""
    sidecar = engine_meta_path(Path("artifacts/model_fp16.engine"))
    assert sidecar.name == "model_fp16.engine.meta.json"


def test_engine_incompatibility_missing_engine(tmp_path: Path) -> None:
    """Check 1: absent engine resolves in the fast path, no tensorrt import."""
    reason = engine_incompatibility(tmp_path / "absent.engine")
    assert reason is not None and reason.startswith("engine file not found")


def test_engine_incompatibility_missing_sidecar(tmp_path: Path) -> None:
    """An engine without build provenance is not usable — sidecar travels."""
    engine = _write_engine(tmp_path)
    reason = engine_incompatibility(engine)
    assert reason is not None and "build metadata missing" in reason


def test_engine_incompatibility_corrupt_sidecar(tmp_path: Path) -> None:
    """Broken metadata -> readable reason, not an unhandled JSONDecodeError."""
    engine = _write_engine(tmp_path)
    engine_meta_path(engine).write_text("{not json")
    reason = engine_incompatibility(engine)
    assert reason is not None and "unreadable" in reason


def test_engine_incompatibility_wrong_resolution(tmp_path: Path) -> None:
    """An engine built at 512² must never serve 384² feeds (and vice versa)."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine, inputs={"pixel_values": [1, 3, 512, 512], "pixel_mask": [1, 512, 512]})
    reason = engine_incompatibility(engine, _write_manifest(tmp_path))
    assert reason is not None and "pixel_values" in reason


def test_engine_incompatibility_wrong_outputs(tmp_path: Path) -> None:
    """postprocess reads exactly OUTPUT_NAMES — a different graph is a new model."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine, outputs={"det_out": [1, 300, 6]})
    reason = engine_incompatibility(engine, _write_manifest(tmp_path))
    assert reason is not None and "output names mismatch" in reason


def test_engine_incompatibility_model_id_mismatch(tmp_path: Path) -> None:
    """An engine built from another checkpoint dies before any CUDA call."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine, model_id="Roboflow/rf-detr-s")
    reason = engine_incompatibility(engine, _write_manifest(tmp_path))
    assert reason is not None and "model_id mismatch" in reason


def test_engine_incompatibility_no_gpu_is_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No nvidia-smi (dev box / CPU CI) -> refuse; never 'assume compatible'."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine)
    monkeypatch.setattr("edge_trt_llm.inference._current_gpu_name", lambda: None)
    reason = engine_incompatibility(engine, _write_manifest(tmp_path))
    assert reason is not None and "no NVIDIA GPU" in reason


def test_engine_incompatibility_gpu_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built on T4, running elsewhere -> refuse, naming both boards."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine)
    monkeypatch.setattr("edge_trt_llm.inference._current_gpu_name", lambda: "GeForce RTX 4090")
    reason = engine_incompatibility(engine, _write_manifest(tmp_path))
    assert reason is not None and "GPU mismatch" in reason and "Tesla T4" in reason


def test_engine_incompatibility_trt_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact TRT version required — engines are per-build-env (see docs/RESULTS.md)."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine)
    monkeypatch.setattr("edge_trt_llm.inference._current_gpu_name", lambda: "Tesla T4")
    fake_trt = ModuleType("tensorrt")
    fake_trt.__version__ = "11.0.0.123"
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    reason = engine_incompatibility(engine, _write_manifest(tmp_path))
    assert reason is not None and "TensorRT mismatch" in reason and "11.4.0.106" in reason


def test_engine_incompatibility_full_match_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Golden path: sidecar + GPU + TRT all match -> None (TrtDetector proceeds)."""
    engine = _write_engine(tmp_path)
    _write_sidecar(engine)
    monkeypatch.setattr("edge_trt_llm.inference._current_gpu_name", lambda: "Tesla T4")
    fake_trt = ModuleType("tensorrt")
    fake_trt.__version__ = "11.4.0.106"
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    assert engine_incompatibility(engine, _write_manifest(tmp_path)) is None


# --- create_detector: fallback must be LOUD -----------------------------------


@needs_ort
def test_create_detector_auto_falls_back_loudly(tmp_path: Path) -> None:
    """auto + unusable engine -> ONNX detector AND a visible UserWarning."""
    from edge_trt_llm.inference import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_ONNX_PATH,
        OnnxDetector,
        create_detector,
    )

    if not DEFAULT_ONNX_PATH.exists() or not DEFAULT_MANIFEST_PATH.exists():
        pytest.skip("run `uv run python scripts/export_onnx.py` first")

    with pytest.warns(UserWarning, match="serving with ONNX Runtime"):
        detector = create_detector(engine_path=tmp_path / "absent.engine", warmup=False)
    assert isinstance(detector, OnnxDetector)


def test_create_detector_strict_tensorrt_never_degrades(tmp_path: Path) -> None:
    """backend='tensorrt' -> EngineUnavailable instead of a silent ONNX swap."""
    from edge_trt_llm.inference import EngineUnavailable, create_detector

    with pytest.raises(EngineUnavailable, match="engine file not found"):
        create_detector(engine_path=tmp_path / "absent.engine", backend="tensorrt", warmup=False)


@needs_ort
def test_create_detector_onnxruntime_is_deterministic() -> None:
    """backend='onnxruntime' -> portable path, no attempt, no fallback warning."""
    from edge_trt_llm.inference import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_ONNX_PATH,
        OnnxDetector,
        create_detector,
    )

    if not DEFAULT_ONNX_PATH.exists() or not DEFAULT_MANIFEST_PATH.exists():
        pytest.skip("run `uv run python scripts/export_onnx.py` first")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector = create_detector(backend="onnxruntime", warmup=False)
    assert isinstance(detector, OnnxDetector)
    assert not [w for w in caught if "serving with ONNX Runtime" in str(w.message)]


def test_create_detector_rejects_unknown_backend() -> None:
    """A typo in the backend string fails at startup, not at first request."""
    from edge_trt_llm.inference import create_detector

    with pytest.raises(ValueError, match="unknown backend"):
        create_detector(backend="cuda_magic")


def test_trt_detector_refuses_at_construction(tmp_path: Path) -> None:
    """No engine -> constructor raises with the chain's own reason string."""
    from edge_trt_llm.inference import EngineUnavailable, TrtDetector

    with pytest.raises(EngineUnavailable, match="engine file not found"):
        TrtDetector(tmp_path / "absent.engine")


# --- Notebook sidecar writer <-> loader contract ------------------------------
# The notebook run on Colab is MANUAL and must never need repeating for a
# schema mismatch: execute the real §3 cell with tensorrt mocked out and feed
# the exact JSON it writes into the exact function that will read it.


def test_notebook_sidecar_writer_satisfies_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = json.loads(Path("notebooks/build_engine_bench.ipynb").read_text(encoding="utf-8"))
    cell = next(
        c
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def build_engine" in "".join(c["source"])
    )
    src = "".join(cell["source"])
    definition = src[: src.index("\nbuild_engine(")]  # drop the immediate invocation
    definition = "\n".join(
        line for line in definition.splitlines() if not line.startswith(("import ", "from "))
    )

    class FakeLogger:  # the notebook only touches Logger(VERBOSE/WARNING)
        VERBOSE = WARNING = 0

        def __init__(self, *args: object) -> None:
            pass

    class FakeEngine:  # mirrors the REAL downloaded sidecar: 2 inputs, 9 outputs
        _tensors = {
            "pixel_values": ([1, 3, 384, 384], "INPUT"),
            "pixel_mask": ([1, 384, 384], "INPUT"),
            "logits": ([1, 300, 91], "OUTPUT"),
            "pred_boxes": ([1, 300, 4], "OUTPUT"),
            # intermediates the exporter keeps — the loader must tolerate them:
            "init_reference_points": ([1, 300, 4], "OUTPUT"),
            "last_hidden_state": ([1, 300, 256], "OUTPUT"),
            "intermediate_hidden_states": ([2, 1, 300, 256], "OUTPUT"),
            "intermediate_reference_points": ([1, 1, 300, 4], "OUTPUT"),
            "enc_outputs_class": ([1, 300, 91], "OUTPUT"),
            "enc_outputs_coord_logits": ([1, 300, 4], "OUTPUT"),
            "backbone_features": ([1, 256, 24, 24], "OUTPUT"),
        }

        def __iter__(self):
            return iter(self._tensors)

        def get_tensor_shape(self, name: str) -> list[int]:
            return self._tensors[name][0]

        def get_tensor_mode(self, name: str) -> str:
            return self._tensors[name][1]

    class FakeRuntime:
        def __init__(self, *args: object) -> None:
            pass

        def deserialize_cuda_engine(self, blob: bytes) -> FakeEngine:
            return FakeEngine()

    class FakeConfig:
        def set_memory_pool_limit(self, *args: object) -> None:
            pass

    class FakeBuilder:
        def __init__(self, *args: object) -> None:
            pass

        def create_network(self) -> object:
            return object()

        def create_builder_config(self) -> FakeConfig:
            return FakeConfig()

        def build_serialized_network(self, network: object, config: object) -> bytes:
            return b"pretend-serialized-engine"

    class FakeParser:
        def __init__(self, *args: object) -> None:
            pass

        def parse_from_file(self, path: str) -> bool:
            return True

    fake_trt = ModuleType("tensorrt")
    fake_trt.Logger = FakeLogger
    fake_trt.Builder = FakeBuilder
    fake_trt.OnnxParser = FakeParser
    fake_trt.Runtime = FakeRuntime
    fake_trt.MemoryPoolType = type("Pool", (), {"WORKSPACE": 0})
    fake_trt.TensorIOMode = type("Mode", (), {"INPUT": "INPUT", "OUTPUT": "OUTPUT"})
    fake_trt.__version__ = "11.4.0.106"

    fake_subprocess = SimpleNamespace(
        run=lambda *a, **k: SimpleNamespace(stdout="Tesla T4\n")
    )
    namespace = {
        "trt": fake_trt,
        "subprocess": fake_subprocess,
        "Path": Path,
        "json": json,
        "MODEL_ID": MODEL_ID,
    }
    exec(compile(definition, "<notebook §3>", "exec"), namespace)  # noqa: S102
    engine = tmp_path / "model_fp16.engine"
    namespace["build_engine"](tmp_path / "model_fp32.onnx", engine)

    sidecar = json.loads(engine_meta_path(engine).read_text(encoding="utf-8"))
    assert sidecar["gpu"] == "Tesla T4"
    assert sidecar["tensorrt"] == "11.4.0.106"
    assert sidecar["model_id"] == MODEL_ID
    # THE contract: the loader must accept exactly what the notebook wrote —
    # including a dynamic (-1) batch dim, which the loader deliberately ignores.
    monkeypatch.setattr("edge_trt_llm.inference._current_gpu_name", lambda: "Tesla T4")
    runtime_trt = ModuleType("tensorrt")
    runtime_trt.__version__ = "11.4.0.106"
    monkeypatch.setitem(sys.modules, "tensorrt", runtime_trt)
    assert engine_incompatibility(engine, _write_manifest(tmp_path)) is None
