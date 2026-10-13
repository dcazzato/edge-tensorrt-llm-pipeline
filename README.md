# edge-tensorrt-llm-pipeline

Take an object-detection model from Hugging Face, make it fast (TensorRT FP16),
portable (ONNX CPU fallback) and servable (FastAPI) — then make it honest: a
vision-language model describes each scene and cross-checks the detector's own answer.

[![CI](https://github.com/dcazzato/edge-tensorrt-llm-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/dcazzato/edge-tensorrt-llm-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

Most "edge AI" demos stop at the model running once. This pipeline treats the model as
one stage of a service, and answers the questions that stage alone cannot: does the ONNX
export actually match the original PyTorch model (parity gate), does FP16 acceleration
change the answer (two parity gates — tensor-level and end-to-end), which backend is
serving right now (`/health`), and what did the model actually see (a vision-language
cross-check on `/describe`).

Result: 4.28 ms per frame on a Tesla T4 (3.4× vs PyTorch FP16), with an ONNX CPU
fallback that keeps the exact same API on GPU-less machines.

## Key Results

Measured on **Tesla T4 (Colab free), TensorRT 11.4.0.106**, RF-DETR nano
(static input `1×3×384×384`), forward pass only, GPU-synced — full methodology
in [docs/RESULTS.md](docs/RESULTS.md).

| Backend | p50 (ms) | p99 (ms) | Notes |
|---|---|---|---|
| PyTorch FP16 CUDA (eager) | 14.59 | 15.77 | baseline, no fusion |
| TensorRT FP32 | 12.69 | 15.17 | fusion alone already beats eager FP16 |
| **TensorRT FP16** | **4.28** | **4.45** | **3.4× vs PyTorch FP16** (ModelOpt AutoCast) |
| ONNX Runtime (CPU, local dev) | ~154 | — | GPU-less fallback, same API |

Accuracy parity gates both pass: the FP32 engine matches ONNX at tensor level
(max|Δ| = 2.09e-04); the FP16 engine returns the same detections (4/4,
Δconf = 0.0053, Δbox = 0.0011).

## Architecture

```text
image upload (multipart "file")
    │
    ▼
FastAPI (src/edge_trt_llm/api/app.py) ── _decode: bytes → PIL image, 10 MB cap
    │
    ▼
create_detector() ── engine + .meta.json valid for THIS gpu+trt? ──yes──► TensorRT FP16 (~4 ms)
    │                          no → loud fallback warning (never silent)
    └────────────────────────────────────────────────────────► ONNX Runtime CPU (~154 ms)
    │
    ▼
preprocess (bilinear 384×384 + ImageNet) → forward → postprocess (sigmoid → top-k → rescale)
    │
    ▼
Prediction {detections: [{label, confidence, box}]}        # 91 classes
    │
    ├─ /predict          → JSON
    ├─ /predict/preview  → draw_boxes → annotated PNG
    └─ /describe         → draw_boxes → VLM provider
                             │  build_messages → POST {base}/chat/completions → parse_reply
                             ▼
                     DescribeResponse {prediction, scene, vlm_status}
```

## API

Every endpoint below is documented as a flow — who calls whom, in order.
Inference and VLM calls run in worker threads, so one slow request can never
block the event loop for the other clients.

### `GET /health`

```text
boot → app.state → {"status": "ok", "model_id": "Roboflow/rf-detr-nano",
                     "backend": "tensorrt" | "onnxruntime"}
```

### `POST /predict` — multipart field `file`

```text
bytes → _decode (10 MB cap → 413, empty/undecodable → 422)
      → detector.predict()
          → preprocess (bilinear 384×384 + ImageNet norm)
          → forward (TRT FP16 | ONNX Runtime)
          → postprocess (sigmoid → top-k → rescale to original pixels)
      → Prediction JSON
```

### `POST /predict/preview`

```text
/predict's flow → draw_boxes(copy) → annotated PNG (same detections, drawn)
```

### `POST /describe` — the multimodal entry point

```text
/predict's flow → draw_boxes (the VLM must SEE the boxes it is asked to judge)
                → provider.describe(png, prediction)
                      ├─ build_messages(...)               (vlm/prompts.py)
                      ├─ POST {VLM_BASE_URL}/chat/completions
                      └─ parse_reply(text)                 (vlm/parsing.py)
                → DescribeResponse {prediction, scene, vlm_status}
```

| VLM state | Response |
|---|---|
| not configured (`VLM_BASE_URL` unset) | `200` — detection valid, `scene: null`, `vlm_status: "disabled"` |
| configured and healthy | `200` — `scene.description` + `checks_out: true / false / null` |
| configured but broken | `502` — a half-dead cross-check is loud, never a silent `"ok"` |

**The input is the image only — there is no text field.** The QA protocol (role,
rules, output shape) is fixed server-side in `src/edge_trt_llm/vlm/prompts.py`, and the
per-request text is generated from the detector's own output, so the written evidence
and the drawn boxes always describe the same prediction. Changing the protocol is a
code edit plus a restart — deliberately not a request parameter.

**Try it live.** `/describe` needs a running OpenAI-compatible server, and a
**valid API key** when the service requires one (OpenRouter/OpenAI do, Ollama
doesn't). Copy `.env.example` → `.env` (**gitignored — keys never enter git**),
fill it in, and start with `--env-file .env`. Real response from a live run
(OpenRouter + `openai/gpt-4o-mini`, 1200×800 street photo, detections trimmed):

```json
{
  "prediction": {
    "detections": [
      {"bbox": [1026.26, 450.10, 1200.66, 620.97], "score": 0.914, "label": "car", "class_id": 3},
      {"bbox": [186.76, 194.90, 267.57, 483.57], "score": 0.847, "label": "person", "class_id": 1}
    ],
    "image_size": [1200, 800],
    "backend": "onnxruntime",
    "model_id": "Roboflow/rf-detr-nano",
    "latency_ms": 163.45
  },
  "scene": {
    "description": "The scene depicts a market area with several people and cars visible...",
    "checks_out": true,
    "raw": null,
    "provider": "openai-compat",
    "model_id": "openai/gpt-4o-mini",
    "latency_ms": 9498.89
  },
  "vlm_status": "ok"
}
```

Reading it: `scene.description` is composed live from your image — the prompt
fixes the *shape* (JSON with description + tri-state `checks_out`), never the
words; `raw: null` means the reply parsed cleanly (it only carries text as a
parse-fallback); `provider` is our adapter's name, `model_id` the model that
actually answered.

## Quickstart

```bash
git clone <repo-url> && cd edge-tensorrt-llm-pipeline

# 1. API + CPU fallback backend
uv sync --extra api

# 2. HF RF-DETR nano → artifacts/model.onnx + manifest (labels, preprocess config)
uv run python scripts/export_onnx.py

# 3. serve
uv run uvicorn edge_trt_llm.api.app:app --reload

# 4. smoke-test
curl -F "file=@photo.jpg" http://127.0.0.1:8000/predict | jq '.detections[0]'
```

**GPU machine** (TensorRT FP16, same API):

```bash
uv sync --extra api --extra trt
# build the engine with notebooks/build_engine_bench.ipynb →
# create_detector() picks it up automatically, else falls back to ONNX
```

**Multimodal layer** (optional — any OpenAI-compatible `/chat/completions`
server works: OpenRouter, OpenAI, Ollama, vLLM, LM Studio):

```bash
uv sync --extra api --extra vlm
cp .env.example .env       # fill it in — .env is gitignored, NEVER commit keys
uv run uvicorn edge_trt_llm.api.app:app --env-file .env

curl -F "file=@photo.jpg" http://127.0.0.1:8000/describe | jq .scene
```

## Docker

The image serves the **CPU/ONNX path** — TensorRT needs a GPU and a CUDA base
image, so it is deliberately not in here. The model is not in git (`*.onnx` is
gitignored), so build **after** exporting it:

```bash
uv run python scripts/export_onnx.py    # once: writes artifacts/
docker build -t edge-trt-llm .
docker run --rm -p 8000:8000 edge-trt-llm
```

```bash
curl http://localhost:8000/health       # the image healthcheck runs the same probe
curl -F "file=@photo.jpg" localhost:8000/predict
```

The VLM layer is env-driven exactly like the bare-metal server:
`docker run --env-file .env -p 8000:8000 edge-trt-llm`.

## TensorRT optimization workflow

- `notebooks/build_engine_bench.ipynb` — build the FP32/FP16 engines + benchmark on a Colab T4/L4
- `docs/RESULTS.md` — the full numbers, methodology and accuracy parity check

`.engine` files are GPU- and TensorRT-version-specific and are **not committed**;
they ship as GitHub Release assets with a build manifest. The ONNX model is the
portable artifact.

## Project structure

```text
├── artifacts/                      # committed: manifest.json + benchmark.json;
│                                    # model.onnx & engines → Release assets (too heavy for git)
├── docs/RESULTS.md                 # benchmark numbers, methodology, parity gates
├── notebooks/build_engine_bench.ipynb   # Colab: build FP32/FP16 engines + benchmark
├── scripts/export_onnx.py          # HF RF-DETR → ONNX + parity check + manifest
├── src/edge_trt_llm/
│   ├── api/app.py                  # FastAPI: /health /predict /predict/preview /describe
│   ├── inference.py                # preprocess · postprocess · OnnxDetector · TrtDetector · create_detector
│   ├── models/types.py             # Detection / Prediction / SceneSummary / DescribeResponse
│   └── vlm/                        # the /describe chain: prompts · parsing · providers
└── tests/                          # 56 tests: parity vs HF, contracts, API, notebook cell
```

## Status & next steps

**Working and tested** (56 tests, `uv run pytest`):
- export RF-DETR nano → ONNX with a PyTorch ↔ ONNX Runtime parity check
- inference layer: numpy preprocessing/postprocessing, ONNX Runtime backend
- TensorRT FP16 build + benchmark on a Colab T4 (4.28 ms p50, see Key Results)
- serving-time engine validation with a loud fallback to ONNX
- FastAPI service: `/health`, `/predict`, `/predict/preview`, `/describe`
- multimodal cross-check, verified end-to-end against a real VLM (OpenRouter)
- CPU serving image (Docker, ONNX path)

**Possible next steps:** publishing the T4-built engine as a GitHub Release
asset, and a C++ TensorRT runtime.

## Tech stack

`TensorRT 11` · `ONNX Runtime` · `FastAPI` · `Hugging Face` · `ModelOpt` · `Pydantic` · `pytest` · `uv`

## AI disclosure

Built incrementally with an agentic AI coding assistant (Claude, driven via
Cline), under continuous human direction: requirements, architecture, every
benchmark run (local CPU + Colab T4) and the verification of each published
number in `docs/RESULTS.md` are the author's. AI pairing here is the point,
not a shortcut: the repo is about the verification gates that make the output
trustworthy.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Dario.
