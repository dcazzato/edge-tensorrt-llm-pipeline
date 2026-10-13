"""Load RF-DETR nano, export to ONNX, and run a parity check.

Usage:
  python scripts/export_onnx.py                    # normal run (export + parity)
  python scripts/export_onnx.py --process-example  # also save example outputs
"""

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoImageProcessor, RfDetrForObjectDetection

from edge_trt_llm.inference import draw_boxes
from edge_trt_llm.models.types import Detection, Prediction

MODEL_ID = "Roboflow/rf-detr-nano"
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # docs sample: cats + remote
SCORE_THRESHOLD = 0.5
ONNX_PATH = Path("artifacts/model.onnx")
KEY_OUTPUTS = ["logits", "pred_boxes"]  # the two tensors the pipeline actually consumes
PARITY_RTOL = 1e-3  # relative tolerance (allclose-style): FP32 kernels reorder ops
PARITY_ATOL = 1e-4  # absolute floor for values near zero


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RF-DETR nano to ONNX and run parity checks."
    )
    parser.add_argument(
        "--process-example",
        action="store_true",
        help=(
            "Save visual outputs (annotated images) and JSON predictions "
            "for both PyTorch and ONNX for visual debugging."
        ),
    )
    args = parser.parse_args()

    # 1) Model + processor (first run downloads the checkpoint to ~/.cache/huggingface)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = RfDetrForObjectDetection.from_pretrained(MODEL_ID).eval().to("cpu")

    id2label = model.config.id2label
    print(f"model:    {MODEL_ID}")
    print(f"classes:  {len(id2label)} COCO categories")
    print(f"params:   {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 2) Sample image straight from the official docs example (stdlib, no extra deps)
    image = Image.open(io.BytesIO(urllib.request.urlopen(SAMPLE_URL).read()))

    # 3) Preprocess -> forward -> postprocess (no NMS: DETR-family is end-to-end)
    inputs = image_processor(images=image, return_tensors="pt")
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    latency_ms = (time.perf_counter() - start) * 1000  # forward only; stage breakdown later

    target_sizes = torch.tensor([image.size[::-1]])  # (height, width), as HF expects
    result = image_processor.post_process_object_detection(
        outputs, threshold=SCORE_THRESHOLD, target_sizes=target_sizes
    )[0]

    # 4) Map to OUR contract — the Prediction type gets its first real data
    detections = [
        Detection(
            bbox=tuple(box.tolist()),  # HF returns (x1, y1, x2, y2) in original pixels
            score=score.item(),
            label=id2label[label.item()],
            class_id=label.item(),
        )
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"])
    ]
    prediction = Prediction(
        detections=detections,
        image_size=image.size,  # (width, height)
        backend="pytorch",
        model_id=MODEL_ID,
        latency_ms=round(latency_ms, 2),
    )
    print(prediction.model_dump_json(indent=2))

    # 5) v2: trace the forward graph into a portable ONNX artifact
    #    (what ONNX Runtime and, later, TensorRT will consume)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=1, max=64)  # free batch dimension
    torch.onnx.export(
        model,
        args=(inputs["pixel_values"], inputs["pixel_mask"]),  # signature order, not dict order
        f=str(ONNX_PATH),
        input_names=["pixel_values", "pixel_mask"],
        output_names=list(outputs.keys()),
        dynamic_shapes=({0: batch}, {0: batch}),  # dynamo spelling of dynamic_axes
        opset_version=18,  # Resize.antialias needs >=18 (where bicubic_aa lands)
        dynamo=True,  # current exporter: translates via onnxscript's op registry
    )
    print(f"onnx:     {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")

    # 5b) The graph answers with 91 raw class INDICES; the only place "17"
    #     becomes "cat" is the HF checkpoint's id2label. Reading it at serving
    #     time would mean importing transformers — which the `api` extra
    #     deliberately excludes (an edge container should not carry a training
    #     framework). So the exporter, the one step that already has
    #     transformers loaded, dumps it beside the artifact. This sidecar is
    #     what lets FastAPI and the future LLM layer speak class NAMES without
    #     the HF stack: no manifest = no labels = no Prediction.
    manifest = {"model_id": MODEL_ID, "labels": {str(k): v for k, v in id2label.items()}}
    manifest_path = Path("artifacts/manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path} ({len(id2label)} classes)")

    # 6) v3: parity check — same inputs through ONNX Runtime vs the PyTorch reference.
    #    Raw tensors compared BEFORE post-processing: if these match, everything downstream does.
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "pixel_values": inputs["pixel_values"].numpy(),
        "pixel_mask": inputs["pixel_mask"].numpy(),
    }
    start = time.perf_counter()
    ort_outputs = dict(zip(KEY_OUTPUTS, session.run(KEY_OUTPUTS, ort_inputs)))
    ort_latency_ms = (time.perf_counter() - start) * 1000  # forward only, mirrors the torch timer
    for name in KEY_OUTPUTS:
        expected = outputs[name].numpy()
        actual = ort_outputs[name]
        if expected.shape != actual.shape:
            raise ValueError(f"shape mismatch {name}: {expected.shape} vs {actual.shape}")
        delta = float(abs(expected - actual).max())
        scale = float(abs(expected).max())
        if not (abs(expected - actual) <= PARITY_ATOL + PARITY_RTOL * abs(expected)).all():
            raise ValueError(f"parity FAILED {name}: Δ={delta:.2e}, scale={scale:.1e}")
        print(f"parity {name:10s} OK — max|Δ| = {delta:.2e} (values up to {scale:.1e})")
    print("parity:   OK — ONNX matches the PyTorch reference")
    print(
        f"latency:  pytorch {latency_ms:.1f} ms — onnxruntime {ort_latency_ms:.1f} ms "
        f"(single cold run, CPU)"
    )

    # 7) [Optional Dev Feature] End-to-end visual and JSON comparison
    if args.process_example:
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Mock object to allow attribute access (outputs.logits) for HF post-processing
        class ModelOutputMock:
            def __init__(self, outputs_dict: dict):
                self.logits = outputs_dict["logits"]
                self.pred_boxes = outputs_dict["pred_boxes"]

        ort_outputs_torch = {
            k: torch.from_numpy(v) for k, v in ort_outputs.items()
        }
        mock_outputs = ModelOutputMock(ort_outputs_torch)

        onnx_result_raw = image_processor.post_process_object_detection(
            mock_outputs, threshold=SCORE_THRESHOLD, target_sizes=target_sizes
        )[0]

        onnx_detections = [
            Detection(
                bbox=tuple(box.tolist()),
                score=score.item(),
                label=id2label[label.item()],
                class_id=label.item(),
            )
            for score, label, box in zip(
                onnx_result_raw["scores"],
                onnx_result_raw["labels"],
                onnx_result_raw["boxes"],
            )
        ]

        pytorch_json_path = artifacts_dir / "pytorch_predictions.json"
        onnx_json_path = artifacts_dir / "onnx_predictions.json"
        
        pytorch_json_path.write_text(prediction.model_dump_json(indent=2))
        onnx_json_path.write_text(
            Prediction(
                detections=onnx_detections,
                image_size=image.size,
                backend="onnxruntime",
                model_id=MODEL_ID,
                latency_ms=round(ort_latency_ms, 2),
            ).model_dump_json(indent=2)
        )

        pt_image_out = draw_boxes(image, detections)
        ort_image_out = draw_boxes(image, onnx_detections)

        pt_image_out.save(artifacts_dir / "pytorch_rendered.jpg")
        ort_image_out.save(artifacts_dir / "onnx_rendered.jpg")

        print(f"debug:   Saved visual debug artifacts and JSONs to {artifacts_dir}/")


if __name__ == "__main__":
    main()