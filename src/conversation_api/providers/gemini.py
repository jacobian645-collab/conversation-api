"""Google Gemini through the google-genai SDK: Gemini API (GEMINI_API_KEY) or Vertex AI (ADC).

In Google Cloud Shell, Vertex AI works with no key: Application Default Credentials and
GOOGLE_CLOUD_PROJECT are already set.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from .base import (
    Completion,
    ErrorKind,
    GenerationRequest,
    Provider,
    ProviderError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
    kind_for_status,
)


def to_gemini_contents(req: GenerationRequest) -> list[dict[str, Any]]:
    # Gemini calls the assistant role "model".
    return [
        {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
        for m in req.messages
    ]


def build_gemini_config(req: GenerationRequest) -> dict[str, Any]:
    cfg: dict[str, Any] = {"max_output_tokens": req.max_tokens}
    if req.system:
        cfg["system_instruction"] = req.system
    if req.temperature is not None:
        cfg["temperature"] = req.temperature
    return cfg


def _usage(resp: Any) -> Usage:
    meta = getattr(resp, "usage_metadata", None)
    if not meta:
        return Usage()
    return Usage(getattr(meta, "prompt_token_count", 0) or 0, getattr(meta, "candidates_token_count", 0) or 0)


def _stop(resp: Any) -> str | None:
    cands = getattr(resp, "candidates", None) or []
    reason = getattr(cands[0], "finish_reason", None) if cands else None
    return getattr(reason, "name", None) or (str(reason) if reason else None)


def parse_gemini_response(resp: Any) -> Completion:
    return Completion(text=getattr(resp, "text", None) or "", usage=_usage(resp), stop_reason=_stop(resp))


class GeminiProvider(Provider):
    name = "gemini"
    sdk_module = "google.genai"

    def _make_client(self) -> Any:
        from google import genai

        opts = self.spec.options
        if opts.get("vertexai"):
            project = opts.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            return genai.Client(vertexai=True, project=project or None, location=opts.get("location") or None)
        return genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        if spec.options.get("vertexai"):
            if not (spec.options.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")):
                return False, "GOOGLE_CLOUD_PROJECT is not set"
            return True, "ok (Application Default Credentials)"
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return False, "GEMINI_API_KEY is not set"
        return True, "ok"

    def _translate(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            return self.error(f"gemini {code}: {getattr(exc, 'message', exc)}", kind_for_status(code), code)
        if isinstance(exc, ValueError) and "api key" in str(exc).lower():
            return self.error(f"gemini: {exc}", ErrorKind.UNAVAILABLE)
        return self.error(f"gemini: {type(exc).__name__}: {exc}", ErrorKind.RETRYABLE)

    async def complete(self, req: GenerationRequest) -> Completion:
        try:
            resp = await self.client.aio.models.generate_content(
                model=req.model, contents=to_gemini_contents(req), config=build_gemini_config(req)
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        return parse_gemini_response(resp)

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        usage, stop = Usage(), None
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=req.model, contents=to_gemini_contents(req), config=build_gemini_config(req)
            )
            async for chunk in stream:
                if text := getattr(chunk, "text", None):
                    yield TextDelta(text)
                if getattr(chunk, "usage_metadata", None):
                    usage = _usage(chunk)
                stop = _stop(chunk) or stop
        except Exception as exc:
            raise self._translate(exc) from exc
        yield StreamEnd(usage, stop)
