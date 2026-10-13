"""Inference core: image in → ``Prediction`` out, no transformers.

Why this module exists
----------------------
The shortcut would be keeping ``AutoImageProcessor`` inside FastAPI and
calling it per request: it works and there is no geometry to get wrong. We
deliberately do not, for three reasons:

1. **Dependencies.** The ``api`` extra ships ``onnxruntime`` + ``pillow`` but
   NOT ``transformers``/``torch`` (see pyproject.toml) — an edge container
   should not carry a training framework just to resize a JPEG.
2. **It is verifiable.** ``scripts/export_onnx.py`` proved the discipline:
   pin our numbers against the reference with a parity check.
   ``tests/test_inference.py`` does the same for *this* module, tensor by
   tensor, so "trust transformers" is no longer the only way to be safe.
3. **Backend swaps.** TensorRT will consume the very same feed dict. Keeping
   preprocessing a pure function means swapping the backend does not touch it.

Layout: :func:`preprocess` / :func:`postprocess` are pure (no I/O, no
session, no globals); the detector classes hold state and load their runtime
ONCE — FastAPI creates one at startup and reuses it.

The TensorRT backend :class:`TrtDetector` sits behind the very same
``predict(image) -> Prediction`` contract. The contract is what makes the
fallback safe — an engine file is a build artifact of ONE GPU + ONE TensorRT
version, so on most machines it is missing or incompatible. Before touching
CUDA, :func:`engine_incompatibility` validates a build-metadata sidecar
(``<engine>.meta.json``) and raises :class:`EngineUnavailable`;
:func:`create_detector` converts that into a fallback to :class:`OnnxDetector`.
The fallback warns on purpose: ONNX Runtime's silent CPU fallback cost us a
whole benchmark run (see ``docs/RESULTS.md``) — this layer refuses to serve
with a different backend than requested without saying so.
:func:`draw_boxes` is the one presentation helper: the export script and the
API's ``/predict/preview`` render detections the same way because they call
the same function.

Class names do not live in the ``.onnx`` (the graph carries tensors, not
intent), so ``scripts/export_onnx.py`` writes ``artifacts/manifest.json``
beside it and :func:`load_manifest` reads it.
"""

import json
import subprocess
import threading
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from edge_trt_llm.models.types import BackendName, Detection, Prediction

# --- RF-DETR preprocessing contract (verbatim from preprocessor_config.json).
# NOTE: INPUT_SIZE is (height, width) — HF's convention — NOT PIL's (w, h).
INPUT_SIZE = (384, 384)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
RESCALE_FACTOR = 0.00392156862745098  # 1/255 as written in the config file
SCORE_THRESHOLD = 0.5

OUTPUT_NAMES = ("logits", "pred_boxes")
DEFAULT_ONNX_PATH = Path("artifacts/model.onnx")
DEFAULT_MANIFEST_PATH = Path("artifacts/manifest.json")
# FP16 engine (notebook §4) — never committed (*.engine → Release assets),
# so on a fresh checkout this path is empty and "auto" serves ONNX.
DEFAULT_ENGINE_PATH = Path("artifacts/model_fp16.engine")


def preprocess(
    image: Image.Image,
    *,
    input_size: tuple[int, int] = INPUT_SIZE,
    mean: Sequence[float] = IMAGE_MEAN,
    std: Sequence[float] = IMAGE_STD,
) -> dict[str, np.ndarray]:
    """RGB image → the exact feed dict the exported graph declares.

    Mirrors ``RfDetrImageProcessor`` step by step — and the order matters:

    1. **rescale to [0, 1] BEFORE resizing.** RF-DETR is the opposite of
       classic DETR here (their own source comments it: *"rescale then
       resize"*); doing it after would silently change every pixel value.
    2. **resize by stretching to a square** — no letterbox, aspect ratio is
       intentionally discarded; ``postprocess`` scales boxes back to the
       original pixels, which is exact because the mapping is linear per axis.
    3. **ImageNet normalisation**, then NCHW + batch dimension.

    The ``pixel_mask`` marks valid pixels; a single fully-resized image has no
    padding, so it is all ones (int64 — what the graph declares; onnxruntime
    does not silently cast).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")  # FastAPI can hand us grayscale/RGBA

    pixels = np.asarray(image, dtype=np.float32) * np.float32(RESCALE_FACTOR)
    pixels = _resize_bilinear(pixels, input_size[0], input_size[1])
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

    pixel_values = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])  # (1, 3, H, W)
    pixel_mask = np.ones((1, *input_size), dtype=np.int64)
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask}


def _resize_bilinear(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Float32 HWC image → (out_h, out_w) bilinear, torch's ``align_corners=False``.

    Coordinate map (same as ``F.interpolate(mode="bilinear", align_corners=False)``):
    ``src = (dst + 0.5) * scale - 0.5``, clamped to the source range. Verified
    empirically against the HF processor: for a 4px left-white image the
    white→black edge lands on pixel 192 of 384, exactly as this formula
    predicts. A wrong convention here shifts every box by a few pixels while
    everything still "works" — the classic silent detection bug.

    NOTE — future refactoring candidate: ``cv2.resize(..., INTER_LINEAR)``
    would replace this entire function (same half-pixel convention, no
    antialiasing, float32 in/out), but opencv is not in the `api` extra and
    its wheel weighs tens of MB. 15 lines of tested numpy are cheaper than a
    new dependency. Revisit only if profiling puts preprocessing on the
    critical path; the parity test is the verdict either way.
    """
    in_h, in_w = image.shape[:2]
    # float32 coordinates on purpose: torch's upsample kernel does the same
    # arithmetic in float32, and matching it drops the delta vs the HF
    # reference from 1.0e-4 to 5.5e-7. float64 here would be "more precise"
    # but *less* faithful — we want the same numbers the reference produced.
    scale_y = np.float32(in_h / out_h)
    scale_x = np.float32(in_w / out_w)
    ys = (np.arange(out_h, dtype=np.float32) + np.float32(0.5)) * scale_y - np.float32(0.5)
    xs = (np.arange(out_w, dtype=np.float32) + np.float32(0.5)) * scale_x - np.float32(0.5)
    ys = np.clip(ys, np.float32(0.0), np.float32(in_h - 1))
    xs = np.clip(xs, np.float32(0.0), np.float32(in_w - 1))

    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    # Cast the integer indices BEFORE subtracting: float32 - int64 promotes to
    # float64 in NumPy, which would silently hand onnxruntime a double tensor
    # (it rejects it) instead of the float32 the graph declares.
    wy = (ys - y0.astype(np.float32))[:, None, None]
    wx = (xs - x0.astype(np.float32))[None, :, None]
    one = np.float32(1.0)

    top = image[np.ix_(y0, x0)] * (one - wx) + image[np.ix_(y0, x1)] * wx
    bottom = image[np.ix_(y1, x0)] * (one - wx) + image[np.ix_(y1, x1)] * wx
    return top * (one - wy) + bottom * wy


def postprocess(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    image_size: tuple[int, int],
    *,
    labels: Mapping[int, str],
    threshold: float = SCORE_THRESHOLD,
) -> list[Detection]:
    """Raw graph outputs → detections in ORIGINAL pixel space (batch = 1).

    Faithful port of ``RfDetrImageProcessor.post_process_object_detection``:

    * sigmoid over classes, then **global top-k over the flattened
      (query × class) matrix** — k defaults to ``num_queries``, which is
      NOT the same as taking the best class per query;
    * ``pred_boxes`` are (cx, cy, w, h) normalised to the resized square →
      corners, then scaled by ``(width, height)`` of the original image;
    * ``score > threshold`` is **strict**, exactly like HF.

    Returns ``[]`` when nothing clears the threshold — a valid prediction
    ("nothing found"), not an error.
    """
    if logits.shape[0] != pred_boxes.shape[0]:
        raise ValueError(f"shape mismatch: logits {logits.shape} vs pred_boxes {pred_boxes.shape}")
    num_queries, num_classes = logits.shape

    prob = _sigmoid(logits)
    flat = prob.reshape(-1)
    k = min(num_queries, flat.size)  # HF: top_k defaults to num_queries
    top = np.argsort(-flat, kind="stable")[:k]  # descending, like torch.topk
    scores = flat[top]
    query_idx = top // num_classes
    class_idx = top % num_classes

    cx, cy, w, h = pred_boxes[query_idx].T
    width, height = image_size  # PIL order: (width, height)
    x1 = (cx - 0.5 * w) * np.float32(width)
    y1 = (cy - 0.5 * h) * np.float32(height)
    x2 = (cx + 0.5 * w) * np.float32(width)
    y2 = (cy + 0.5 * h) * np.float32(height)

    keep = scores > threshold
    kept_classes = class_idx[keep]
    unknown = sorted(set(kept_classes.tolist()) - labels.keys())
    if unknown:
        raise KeyError(
            f"classes {unknown} missing from manifest — regenerate artifacts/manifest.json"
        )

    return [
        Detection(
            bbox=(float(a), float(b), float(c), float(d)),
            score=float(s),
            label=labels[int(cls)],
            class_id=int(cls),
        )
        for s, cls, a, b, c, d in zip(
            scores[keep], kept_classes, x1[keep], y1[keep], x2[keep], y2[keep], strict=True
        )
    ]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """``1/(1+e^-x)`` without overflow: ``e^{-log(1+e^{-x})}`` via logaddexp.

    Equivalent to ``torch.sigmoid`` within float32 epsilon; the naive form
    would emit RuntimeWarnings for strongly negative logits.
    """
    return np.exp(-np.logaddexp(0.0, -x))


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> tuple[str, dict[int, str]]:
    """Read the sidecar written by ``scripts/export_onnx.py`` → (model_id, labels)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python scripts/export_onnx.py` first"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data["model_id"]), {int(k): v for k, v in data["labels"].items()}


def _blank_feed(input_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Zero image feed — a legal input used only to warm the session up."""
    return {
        "pixel_values": np.zeros((1, 3, *input_size), dtype=np.float32),
        "pixel_mask": np.ones((1, *input_size), dtype=np.int64),
    }


class OnnxDetector:
    """ONNX Runtime backend: session loaded once, ``predict()`` per request.

    Loading ``model.onnx`` + its 111 MB external weights takes ~1 s — far too
    slow to repeat per call — so FastAPI will hold ONE instance created at
    startup. ``providers`` is a parameter so TensorRT/CUDA
    execution providers without changing this contract:
    ``predict(image) -> Prediction``.
    """

    # Reported by /health; always matches Prediction.backend for this class.
    backend: BackendName = "onnxruntime"

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX_PATH,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        threshold: float = SCORE_THRESHOLD,
        warmup: bool = True,
    ) -> None:
        self.model_id, self.labels = load_manifest(manifest_path)
        self.threshold = threshold
        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers))

        # Input geometry comes from the graph itself, so re-exporting at
        # another resolution does not require editing this file. The batch
        # dim is dynamic (str) — H and W are static for RF-DETR.
        dims = next(i.shape for i in self.session.get_inputs() if i.name == "pixel_values")
        self.input_size = (int(dims[2]), int(dims[3])) if isinstance(dims[2], int) else INPUT_SIZE

        if warmup:
            # First run pays arena/kernel init; do it now so the first real
            # request is not always the slowest sample in the benchmark.
            self.session.run(list(OUTPUT_NAMES), _blank_feed(self.input_size))

    def predict(self, image: Image.Image) -> Prediction:
        """Full request path: preprocess → forward → postprocess → contract."""
        feed = preprocess(image, input_size=self.input_size)
        start = time.perf_counter()
        logits, pred_boxes = self.session.run(list(OUTPUT_NAMES), feed)
        latency_ms = (time.perf_counter() - start) * 1000  # forward only, as in the export

        detections = postprocess(
            logits[0], pred_boxes[0], image.size, labels=self.labels, threshold=self.threshold
        )
        return Prediction(
            detections=detections,
            image_size=image.size,
            backend="onnxruntime",
            model_id=self.model_id,
            latency_ms=round(latency_ms, 2),
        )


class EngineUnavailable(RuntimeError):
    """TensorRT engine cannot run here; :func:`create_detector` falls back to ONNX."""


def engine_meta_path(engine_path: Path) -> Path:
    """Build-metadata sidecar: travels glued to the engine file itself.

    Engines ship as Release assets — the sidecar must ship with them,
    because "which GPU + which TensorRT built this" is part of the artifact.
    """
    return engine_path.with_name(engine_path.name + ".meta.json")


def _current_gpu_name() -> str | None:
    """GPU name via the exact ``nvidia-smi`` query the notebook records at build time.

    ``None`` means "no NVIDIA GPU visible" (dev box, CPU CI) — never guess.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    names = proc.stdout.strip().splitlines()
    return names[0] if names else None


def _norm_gpu(name: str) -> str:
    """Collapse whitespace + case: ``'Tesla  T4'`` and ``'tesla t4'`` are one board."""
    return " ".join(name.split()).casefold()


def engine_incompatibility(
    engine_path: Path, manifest_path: Path = DEFAULT_MANIFEST_PATH
) -> str | None:
    """``None`` = the engine can run here; otherwise the human-readable reason.

    Checks run cheapest-first and WITHOUT importing tensorrt: importing it on
    a machine without CUDA can abort the process (notebook §1 saw it), and the
    common case "no engine on this box" must return in microseconds. Order:

    1. engine file present?
    2. sidecar present and parseable?
    3. I/O spec matches what ``preprocess`` feeds / ``postprocess`` reads?
    4. engine's model_id == manifest's model_id (built from THIS model)?
    5. GPU visible and equal to the build GPU?
    6. TensorRT installed and equal to the build version (exact)?

    Step 6 is deliberately strict: TensorRT documents serialized engines as
    portable only to the exact build environment — a mismatch means rebuild,
    not "try anyway and hope output is not garbage".
    """
    engine_path = Path(engine_path)
    if not engine_path.is_file():
        return f"engine file not found: {engine_path}"

    meta_path = engine_meta_path(engine_path)
    if not meta_path.is_file():
        return f"build metadata missing: {meta_path} (sidecar must travel with the engine)"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"build metadata unreadable: {exc}"

    inputs = meta.get("inputs") or {}
    expected_shapes = {"pixel_values": (3, *INPUT_SIZE), "pixel_mask": tuple(INPUT_SIZE)}
    for name, tail in expected_shapes.items():
        shape = inputs.get(name)
        if shape is None or tuple(shape[1:]) != tail:
            return f"input {name!r}: engine {shape}, expected [batch, {list(tail)}]"

    # The real graph exposes MORE outputs than we consume (the exporter keeps
    # intermediates — the downloaded sidecar lists 9): demand the ones
    # postprocess reads, tolerate the extras.
    outputs = set(meta.get("outputs") or {})
    if not set(OUTPUT_NAMES) <= outputs:
        return f"output names mismatch: engine {sorted(outputs)}, need {sorted(OUTPUT_NAMES)}"

    model_id, _labels = load_manifest(manifest_path)
    if meta.get("model_id") != model_id:
        return f"model_id mismatch: engine {meta.get('model_id')!r} vs manifest {model_id!r}"

    runtime_gpu = _current_gpu_name()
    if runtime_gpu is None:
        return "no NVIDIA GPU visible (nvidia-smi unavailable)"
    build_gpu = str(meta.get("gpu", ""))
    if _norm_gpu(build_gpu) != _norm_gpu(runtime_gpu):
        return f"GPU mismatch: built on {build_gpu!r}, running on {runtime_gpu!r}"

    build_trt = str(meta.get("tensorrt", ""))
    if not build_trt:
        return "build metadata has no 'tensorrt' version"
    try:
        import tensorrt as trt
    except ImportError:
        return "tensorrt not installed (uv sync --extra trt)"
    if trt.__version__ != build_trt:
        return f"TensorRT mismatch: built {build_trt}, runtime {trt.__version__} — rebuild (§3)"
    return None


class TrtDetector:
    """TensorRT backend behind the same ``predict(image) -> Prediction`` contract.

    Same contract, different machinery — and deliberately not a subclass of
    :class:`OnnxDetector`: an ORT session and a TRT context+buffer pair share
    no code, so inheritance would only hide the difference. What callers may
    rely on is exactly the contract (plus ``model_id``/``labels``).

    Lifecycle mirrors the benchmark lesson (docs/RESULTS.md): everything
    expensive — deserialize, ONE execution context, device buffers, tensor
    addresses, warmup — happens in ``__init__``; ``predict`` only copies
    inputs, executes and reads back. The first benchmark measured context
    creation inside the timer; the serving path has no such excuse.

    Thread-safety: unlike an ORT session, one TRT context with shared buffers
    is NOT safe to use concurrently — FastAPI runs ``predict`` in a threadpool,
    so the forward pass is serialized with a lock.
    """

    # Reported by /health; always matches Prediction.backend for this class.
    backend: BackendName = "tensorrt"

    def __init__(
        self,
        engine_path: Path = DEFAULT_ENGINE_PATH,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        threshold: float = SCORE_THRESHOLD,
        warmup: bool = True,
    ) -> None:
        engine_path = Path(engine_path)
        issue = engine_incompatibility(engine_path, manifest_path)
        if issue is not None:
            raise EngineUnavailable(issue)

        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise EngineUnavailable(f"tensorrt/torch not importable: {exc}") from exc

        meta = json.loads(engine_meta_path(engine_path).read_text(encoding="utf-8"))
        self.model_id, self.labels = load_manifest(manifest_path)
        self.threshold = threshold
        dims = meta["inputs"]["pixel_values"]
        self.input_size = (int(dims[2]), int(dims[3]))

        # Keep runtime AND engine alive for the context's whole lifetime;
        # WARNING is enum-only in TRT 11 (notebook §3).
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise EngineUnavailable(f"deserialize_cuda_engine returned None for {engine_path}")
        self._engine = engine

        context = engine.create_execution_context()
        if context is None:
            raise EngineUnavailable(
                "create_execution_context() returned None (driver/GPU mismatch)"
            )
        self._context = context
        self._torch = torch
        self._lock = threading.Lock()
        self._tensors: dict[str, torch.Tensor] = {}
        self._dtypes: dict[str, np.dtype] = {}
        torch_dtypes = {
            np.dtype("float32"): torch.float32,
            np.dtype("float16"): torch.float16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
        }
        for name in engine:  # iterating an engine yields its tensor names (TRT 10+)
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            self._dtypes[name] = dtype
            buf = torch.empty(
                tuple(engine.get_tensor_shape(name)), dtype=torch_dtypes[dtype], device="cuda"
            )
            self._tensors[name] = buf
            context.set_tensor_address(name, buf.data_ptr())

        if warmup:
            # First real request must not pay cuBLAS/cuda init (same reason
            # OnnxDetector warms its session).
            self._forward(_blank_feed(self.input_size))

    def _forward(self, feed: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """H2D copy → execute → sync → D2H; no allocation per call."""
        torch = self._torch
        for name, array in feed.items():
            buf = self._tensors.get(name)
            if buf is None:
                raise KeyError(f"engine exposes no tensor named {name!r}")
            src = np.ascontiguousarray(array.astype(self._dtypes[name], copy=False))
            buf.copy_(torch.from_numpy(src))
        # Async launch: measure executed work, not queue depth (notebook §7).
        self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return tuple(self._tensors[name].cpu().numpy() for name in OUTPUT_NAMES)

    def predict(self, image: Image.Image) -> Prediction:
        """Full request path — same stages as ``OnnxDetector.predict``, other runtime."""
        feed = preprocess(image, input_size=self.input_size)
        start = time.perf_counter()
        with self._lock:
            logits, pred_boxes = self._forward(feed)
        latency_ms = (time.perf_counter() - start) * 1000  # forward only, as everywhere else

        detections = postprocess(
            logits[0], pred_boxes[0], image.size, labels=self.labels, threshold=self.threshold
        )
        return Prediction(
            detections=detections,
            image_size=image.size,
            backend="tensorrt",
            model_id=self.model_id,
            latency_ms=round(latency_ms, 2),
        )


DetectorBackend = Literal["auto", "onnxruntime", "tensorrt"]


def create_detector(
    onnx_path: Path = DEFAULT_ONNX_PATH,
    engine_path: Path = DEFAULT_ENGINE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    threshold: float = SCORE_THRESHOLD,
    backend: DetectorBackend = "auto",
    warmup: bool = True,
) -> OnnxDetector | TrtDetector:
    """Backend picker: ``"auto"`` prefers TensorRT, falls back to ONNX Runtime.

    The fallback warns loudly on purpose — a silent downgrade is exactly what
    hid ONNX Runtime's CPU fallback for a whole benchmark run
    (``docs/RESULTS.md``). A serving box must say what it is serving with.

    ``backend="tensorrt"`` disables the fallback (re-raise instead): use it
    where a missing engine is an error, e.g. a GPU CI runner. ``"onnxruntime"``
    pins the portable path — what CPU-only machines always get.
    """
    if backend not in {"auto", "onnxruntime", "tensorrt"}:
        raise ValueError(f"unknown backend {backend!r}: expected 'auto', 'onnxruntime', 'tensorrt'")
    if backend in {"auto", "tensorrt"}:
        try:
            return TrtDetector(engine_path, manifest_path, threshold=threshold, warmup=warmup)
        except EngineUnavailable as exc:
            if backend == "tensorrt":
                raise
            warnings.warn(f"TensorRT unavailable: {exc} — serving with ONNX Runtime", stacklevel=2)
    return OnnxDetector(onnx_path, manifest_path, threshold=threshold, warmup=warmup)


def draw_boxes(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    """Copy of ``image`` with boxes + labelled scores drawn on it.

    Shared on purpose: ``scripts/export_onnx.py`` (visual regression artifact)
    and the API's ``/predict/preview`` must look identical, so they call this
    one function instead of keeping two copies that drift apart.

    Guarantees callers rely on:

    * **never mutates the input** — the API infers on the very image it then
      renders, and the export script prints its JSON from the same pixels;
    * **always returns RGB** — an upload can arrive as ``L``, ``P`` or ``RGBA``
      and a colour like ``"yellow"`` is meaningless on mode ``L``;
    * **label gets a dark plate** — yellow text alone disappears on a bright
      background (white wall, snow), which is exactly where people look first.

    Output is a normal PIL image: encode it with ``image.save(buffer,
    format="PNG")``, never with ``Image.tobytes()`` (raw pixels, no header).
    """
    annotated = image.convert("RGB") if image.mode != "RGB" else image.copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{det.label}: {det.score:.2f}"
        _, _, text_w, text_h = draw.textbbox((0, 0), text)
        # Anchor inside the box's top-left corner, pulled back on-frame when
        # the box hugs an edge — otherwise the label is drawn off the image.
        tx = min(max(x1 + 4, 0), max(annotated.width - text_w, 0))
        ty = min(max(y1 + 4, 0), max(annotated.height - text_h, 0))
        pad = 2
        draw.rectangle(
            [tx - pad, ty - pad, tx + text_w + pad, ty + text_h + pad],
            fill="black",
        )
        draw.text((tx, ty), text, fill="yellow")

    return annotated
