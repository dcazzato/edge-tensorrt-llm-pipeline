"""Tests for the typed prediction contract — the pipeline's executable specification.

Each test pins ONE behavior of the contract, so any future change that
breaks a boundary (backends, API, LLM layer) fails here first.
"""

import pytest
from pydantic import ValidationError

from edge_trt_llm.models.types import Detection, Prediction


def _make_prediction(**kwargs) -> Prediction:
    """Build a Prediction with defaults; individual tests override what they exercise."""
    base = dict(image_size=(640, 480), backend="pytorch", model_id="rfdetr-nano", latency_ms=1.0)
    base.update(kwargs)
    return Prediction(**base)


def test_detection_accepts_valid_input() -> None:
    """Happy path: a well-formed detection keeps its values (ints coerced to float)."""
    det = Detection(bbox=(10, 20, 110, 220), score=0.97, label="person", class_id=0)
    assert det.bbox == (10.0, 20.0, 110.0, 220.0)
    assert det.score == 0.97


def test_detection_rejects_reversed_bbox() -> None:
    """Semantic layer: shape is fine (4 floats), meaning is not (x2 < x1)."""
    with pytest.raises(ValidationError, match="corners out of order"):
        Detection(bbox=(110, 20, 10, 220), score=0.9, label="person", class_id=0)


def test_detection_rejects_out_of_range_score() -> None:
    """Shape layer: score=1.42 is a float but violates the ge/le bounds."""
    with pytest.raises(ValidationError):
        Detection(bbox=(0, 0, 10, 10), score=1.42, label="person", class_id=0)


def test_prediction_rejects_unknown_backend() -> None:
    """Literal closed set: 'TensorRT' (wrong case) must not pass as a backend name."""
    with pytest.raises(ValidationError):
        _make_prediction(backend="TensorRT")


def test_prediction_empty_detections_is_valid() -> None:
    """An empty list is a legitimate answer ('nothing found'), not an error."""
    prediction = _make_prediction()
    assert prediction.detections == []


def test_prediction_instances_do_not_share_detections() -> None:
    """Regression guard: each instance owns its list (default_factory, not = [])."""
    first = _make_prediction()
    second = _make_prediction()
    first.detections.append(Detection(bbox=(0, 0, 5, 5), score=0.5, label="cat", class_id=15))
    assert second.detections == []
