"""Typed contract shared by every stage of the pipeline.

One definition consumed by the inference backends (TensorRT / ONNX Runtime),
the FastAPI layer and the LLM prompt builder — so all boundaries speak the
same, validated language instead of ad-hoc dicts.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

BackendName = Literal["pytorch", "onnxruntime", "tensorrt"]


class Detection(BaseModel):
    """A single detected object, in pixel coordinates of the ORIGINAL image."""

    bbox: tuple[float, float, float, float] = Field(
        description="(x1, y1, x2, y2) top-left / bottom-right corner, pixels, "
        "relative to the original (pre-resize) image size"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in [0, 1]",
    )
    label: str = Field(
        description="Human-readable class name, e.g. 'person' (what the LLM sees)",
    )
    class_id: int = Field(
        description="Raw class index from the model head (what programs see)",
    )

    @field_validator("bbox")
    @classmethod
    def _check_corner_order(
        cls, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Fail loudly at the boundary: a reversed box is a postprocessing bug."""
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(
                f"bbox corners out of order: {bbox} — expected (x1, y1, x2, y2) with x2 >= x1"
            )
        return bbox


class Prediction(BaseModel):
    """Result of one detector call — THE artifact the rest of the system consumes.

    An empty ``detections`` list is a valid prediction ("nothing found"),
    not an error: distinguishing the two matters when the VLM layer later
    cross-checks the scene.
    """

    detections: list[Detection] = Field(
        default_factory=list,
        description="Detected objects; [] means the model found nothing",
    )
    image_size: tuple[int, int] = Field(
        description="(width, height) of the original input image — bboxes are in its pixel space",
    )
    backend: BackendName = Field(
        description="Which runtime produced this prediction (benchmark comparisons rely on it)",
    )
    model_id: str = Field(
        description="Model identity, e.g. 'rfdetr-nano' (precision/engine info added later)",
    )
    latency_ms: float = Field(
        ge=0.0,
        description="End-to-end inference latency in milliseconds (split per stage later)",
    )


class SceneSummary(BaseModel):
    """What the VLM layer said about the scene — the "describe + cross-check" half.

    ``checks_out`` is tri-state ON PURPOSE: ``None`` means "the model gave no
    parseable opinion", not "the model disagrees". Collapsing the two would
    manufacture a disagreement signal out of a parsing failure — the loudest
    kind of bug in a monitoring pipeline.
    """

    description: str = Field(
        description="Natural-language description of the scene, in English",
    )
    checks_out: bool | None = Field(
        default=None,
        description="Does the image confirm the detector's findings? "
        "None = no parseable opinion (not a disagreement)",
    )
    raw: str | None = Field(
        default=None,
        description="Model's raw text when it did not return the JSON we asked for",
    )
    provider: str = Field(
        description="Provider adapter that served the call, e.g. 'openai-compat'",
    )
    model_id: str = Field(
        description="VLM identity, e.g. 'smolvlm:2b' or 'gpt-4o-mini'",
    )
    latency_ms: float = Field(
        ge=0.0,
        description="End-to-end VLM call latency in milliseconds",
    )


class DescribeResponse(BaseModel):
    """``/describe`` = ``/predict`` + optional scene summary.

    The prediction is embedded whole rather than flattened: /describe stays a
    strict superset of /predict, so one client parser serves both endpoints.
    """

    prediction: Prediction
    scene: SceneSummary | None = Field(
        default=None,
        description="VLM output; null when no VLM is configured (vlm_status='disabled')",
    )
    vlm_status: Literal["ok", "disabled"] = Field(
        description="'disabled' (scene=null) means no VLM configured — the detection "
        "is still fully valid; a configured-but-broken VLM is a 502, never this",
    )
