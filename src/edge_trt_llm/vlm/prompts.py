"""VLM prompt construction: the instruction half of the pipeline.

Pure module — no network, no model, no file I/O — because a prompt to a VLM
plays the role that code plays for a detector: it *defines* the behavior.
Defining it here, as data, is what makes it reviewable and unit-testable.

Design decisions encoded below:

* The VLM receives the ANNOTATED image (boxes already drawn) plus the same
  detections as text: vision alone cannot read confidences, and text alone
  cannot judge box *placement* — the cross-check needs both.
* The answer is requested as a compact JSON object (English prose inside it):
  machines parse it, humans can still read it. The parser does NOT trust it
  (see vlm/parsing.py): models add fences, preambles and translations.
* ``checks_out`` is tri-state in the prompt as well: "I cannot tell" is a
  real answer and must never be coerced into ``false``.
* Output language is pinned to English: the same model happily replies in the
  user's language, and mixed-language JSON is a parsing bug waiting to happen.
* The empty-detection case gets an explicit instruction: "detector saw
  nothing" is exactly when a cross-check matters most.
"""

import base64
from typing import Any

from edge_trt_llm.models.types import Detection, Prediction

SYSTEM_PROMPT = """\
You are a QA reviewer for an object-detection pipeline.
You receive (1) an image with the detector's bounding boxes drawn on it and
(2) the same detections as text: label, confidence score and pixel box.

Describe the scene, then judge whether the detector's output fits what you see.
Answer in English with ONLY this JSON object — no markdown fences, no prose:
{"description": "2-4 sentences", "checks_out": true | false | null}

checks_out rules:
- true  : the boxes match the image (right objects, roughly right places) and
          no clearly visible object of a listed kind is missing.
- false : a box sits on empty space or on the wrong object, or a clearly
          visible object of a listed kind has no box at all.
- null  : you cannot tell (too small, occluded, or the evidence is ambiguous).

Never invent detections that are not in the list: you are reviewing them,
not performing the detection yourself.
"""


def format_detections(detections: list[Detection]) -> str:
    """The detection list as evidence text — one line per box.

    The empty case is instructed, not blank: an image the detector found
    nothing in could be genuinely empty or a missed-detection bug — the VLM
    must say which, instead of staring at an empty list.
    """
    if not detections:
        return (
            "(no detections — the detector found nothing). "
            "State whether the image is genuinely empty or an obvious object was missed."
        )
    lines = []
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        lines.append(f"{i}. {det.label} — score {det.score:.2f} at ({x1}, {y1})-({x2}, {y2})")
    return "\n".join(lines)


def build_messages(annotated_png: bytes, prediction: Prediction) -> list[dict[str, Any]]:
    """Assemble the OpenAI-compatible chat messages the VLM will receive.

    ``annotated_png`` must be the image drawn with THESE detections
    (``draw_boxes``) — the whole cross-check rests on the VLM seeing the same
    boxes it is asked to judge. The PNG rides as a base64 data URL: accepted
    by every OpenAI-compatible server (Ollama included) and keeps this
    builder free of temp files.
    """
    evidence = format_detections(prediction.detections)
    user_text = (
        f"Detector output ({prediction.backend} / {prediction.model_id}):\n"
        f"{evidence}\n\n"
        "The annotated image is attached. Reply with the JSON object only."
    )
    image_url = f"data:image/png;base64,{base64.b64encode(annotated_png).decode('ascii')}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
