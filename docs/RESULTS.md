# Benchmark results — TensorRT FP16 on Colab T4

Measured with `notebooks/build_engine_bench.ipynb` (§0–§7), 2026-10-09,
re-run 2026-10-10 after fixing the benchmark harness (see Method).
Every figure names the hardware + TensorRT version it was measured on —
an engine file runs only on the GPU architecture + TRT major it was built with.

## Environment

- GPU: `Tesla T4` (Colab, from `nvidia-smi`)
- TensorRT: `11.4.0.106` (from `trt.__version__`)
- CUDA driver: from `nvidia-smi` in notebook §0
- Model: `Roboflow/rf-detr-nano` → `artifacts/model.onnx` (static input `1×3×384×384`)
- Method: warmup=10 (discarded), timed=50, forward-pass only, GPU-synced,
  median (p50) + p99 — engines **and execution contexts and device buffers**
  allocated once, outside the timer (the first pass created the execution
  context per call, inside the timer, inflating TRT rows ~8×; the recorded
  `benchmark.json` carries the fixed `method` string).
- Data: `artifacts/benchmark.json` (machine-readable, committed).

## Latency (forward pass, same pixels)

| Backend | p50 (ms) | p99 (ms) | Notes |
|---|---|---|---|
| PyTorch FP16 CUDA (eager) | 14.59 | 15.77 | baseline, no fusion |
| ONNX Runtime (CUDA EP requested → ran CPU) | 390.76 | 593.45 | confirmed via `get_providers()`, see note |
| TensorRT FP32 | 12.69 | 15.17 | `create_network()` without flags |
| TensorRT FP16 | **4.28** | **4.45** | ModelOpt AutoCast, `--keep_io_types` |
| ONNX Runtime CPU (local machine) | ~154 | — | reference from `export_onnx.py` parity run (different CPU) |

**Notes**

- **TensorRT FP16 wins: 4.28 ms p50 = 3.4× vs PyTorch FP16 (14.59) and
  3.0× vs TRT FP32 (12.69)** — the expected T4 signature (tensor cores +
  kernel fusion), with a tight p99 (4.45). An earlier run showed the
  opposite ordering; that was a benchmark bug (execution context created
  per call, inside the timer), retracted here.
- **TRT FP32 (12.69) also beats eager PyTorch FP16 (14.59)**: graph-level
  fusion pays even without half precision.
- **ONNX Runtime: CUDA was requested, CPU was used.** The session is created
  with `providers=["CUDAExecutionProvider"]`, but `ort_sess.get_providers()`
  returned `['CPUExecutionProvider']` — ORT silently falls back to CPU when
  a requested EP cannot initialize (Colab preinstalls the CPU build; the GPU
  wheel collides with it or its CUDA EP failed to load). So 390.76 ms is a
  **CPU measurement on Colab's CPU**, vs ~154 ms CPU on the local machine —
  two different CPUs, not a GPU result. There is no ORT-CUDA number in this
  report; performance claims rest on TensorRT only, ONNX Runtime serves as
  the portable correctness reference.

## Accuracy parity (same pixels, `logits` + `pred_boxes`)

Two gates — the FP16 gate is **end-to-end, not tensor-level**, and the
reason is structural: the graph's internal TopK (300-of-576) flips
indices whenever score gaps fall below FP16 noise — 549/575 gaps do —
so tensor parity on `logits` is unattainable by construction, not by bug.

| Engine | Gate | Result | Verdict |
|---|---|---|---|
| FP32 vs ONNX/CUDA | tensor `allclose(rtol=1e-3, atol=1e-4)` | max\|Δ\| = 2.09e-04 | PASS |
| FP16 vs ONNX/CUDA | e2e: same detections @0.5, Δconf ≤ 0.05, Δbox ≤ 0.01 | 4/4 same, Δconf = 0.0053, Δbox = 0.0011 | PASS |

Per-node probe located the FP16 divergence at `node_topk__1` (index
flips), not overflow: `inf16 = 0` on every tensor.

## Layer precision (notebook §5)

TRT 11 has no per-layer engine query (`ICudaEngine.get_layer` removed);
precision is declared in the AutoCast ONNX graph instead — counted from
initializer dtypes:

```text
{'INT64 (non-compute constants)': 327, 'FLOAT': 95, 'FLOAT16': 333}
```

Of the 428 compute-relevant initializers, **333 (78%) are FP16**; the 95
still-FP32 are the explicitly excluded nodes (norm/softmax/detection head).
The 327 INT64 entries are shape/index constants — integer by nature, not a
precision decision. Pattern matches the AutoCast flags: `--keep_io_types`
(FP32 I/O), `--op_types_to_exclude LayerNormalization Softmax ReduceMean`,
`--nodes_to_exclude .*class.*|.*logit.*|.*bbox.*|.*head.*`.

## Artifacts

- `artifacts/benchmark.json` — machine-readable numbers (committed; the
  `.gitignore` re-includes exactly this file). The serving layer reads it.
- `model_fp16.engine` — stays on Colab/Drive → GitHub Release asset in
  the release step. NEVER committed: tied to this GPU + TRT major.

