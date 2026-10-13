"""The only module that talks to a VLM, and the only one that
knows both prompt-building and reply-parsing.

Call flow of ``POST /describe`` (who calls whom):

    app.py  ──►  detector.detect(image)              (existing)
            ──►  draw_boxes(...)                     (existing)
            ──►  provider.describe(png, prediction)  ◄── this module
                     ├─ build_messages(...)          (vlm/prompts.py)
                     ├─ POST {base_url}/chat/completions
                     └─ parse_reply(text)            (vlm/parsing.py)
            ◄──  SceneSummary

``app.py`` never imports prompts/parsing directly: changing the prompt text
or the reply format touches exactly one file, and tests fake the provider
with a two-line stub (the ``VlmProvider`` Protocol) instead of a live model.

Configuration comes from the environment — fail loudly, no silent defaults:

    VLM_BASE_URL    e.g. http://localhost:11434/v1   UNSET = VLM disabled
    VLM_MODEL       e.g. qwen2.5vl:7b               required if base set
    VLM_API_KEY     optional; falls back to OPENAI_API_KEY, then "not-needed"
    VLM_TIMEOUT_S   seconds, default 60 (local CPU VLMs are slow)

Two very different failure modes, deliberately:

* no ``VLM_BASE_URL``     -> a NORMAL state: /describe answers 200 with
  ``vlm_status="disabled"`` — detection still fully works;
* configured but broken   -> raises ``VlmError`` -> endpoint answers 502 —
  a half-dead cross-check must be loud, never silently "fine".
"""

import os
import time
from typing import Protocol

from edge_trt_llm.models.types import Prediction, SceneSummary
from edge_trt_llm.vlm.parsing import parse_reply
from edge_trt_llm.vlm.prompts import build_messages


class VlmProvider(Protocol):
    """The seam: the endpoint depends only on this shape, tests stub it."""

    def describe(self, annotated_png: bytes, prediction: Prediction) -> SceneSummary:
        """Build prompt -> call model -> parse reply, as one opinion about a scene."""
        ...


class VlmError(RuntimeError):
    """Transport or shape failure — app.py maps this to HTTP 502."""


class OpenAiCompatProvider:
    """Adapter for any /chat/completions-compatible server (Ollama, vLLM, OpenAI).

    The ``openai`` SDK is imported lazily inside ``__init__``: without
    ``uv sync --extra vlm`` the API still boots — it just cannot create this
    class (and says so loudly) when a base URL is actually configured.
    """

    name = "openai-compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "not-needed",
        timeout_s: float = 60.0,
        http_client: object | None = None,  # httpx.Client (injected by tests via MockTransport)
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VlmError("openai package not installed — run: uv sync --extra vlm") from exc
        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            http_client=http_client,
        )

    def describe(self, annotated_png: bytes, prediction: Prediction) -> SceneSummary:
        """Full chain: prompts -> network -> parsing -> SceneSummary (never partial)."""
        messages = build_messages(annotated_png, prediction)
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # the QA answer must be reproducible
                max_tokens=400,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            # The SDK raises a whole family of errors (timeouts, 4xx/5xx, auth).
            # One failure language for the endpoint is worth one broad catch.
            raise VlmError(f"VLM call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(content, str) or not content.strip():
            raise VlmError("VLM returned no text content")

        answer = parse_reply(content)
        raw_text = content.strip()
        return SceneSummary(
            description=answer.description,
            checks_out=answer.checks_out,
            # The parser keeps the raw text AS description when it fails —
            # identity check detects exactly that fallback case.
            raw=raw_text if answer.description == raw_text else None,
            provider=self.name,
            model_id=self.model,
            latency_ms=latency_ms,
        )


def create_vlm_provider() -> VlmProvider | None:
    """Environment -> provider, called ONCE at app startup. ``None`` = disabled."""
    base_url = os.getenv("VLM_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    model = os.getenv("VLM_MODEL", "").strip()
    if not model:
        raise VlmError("VLM_BASE_URL is set but VLM_MODEL is empty — set both or neither")
    return OpenAiCompatProvider(
        base_url=base_url,
        model=model,
        api_key=os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed",
        timeout_s=float(os.getenv("VLM_TIMEOUT_S", "60")),
    )
