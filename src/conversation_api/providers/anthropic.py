"""Anthropic API (ANTHROPIC_API_KEY)."""

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


def build_anthropic_kwargs(req: GenerationRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
    }
    if req.system:
        kwargs["system"] = req.system
    if req.temperature is not None:
        # Newer SDKs no longer type `temperature` (and newer models may reject it), so it is sent only
        # when a catalog entry or request explicitly sets it.
        kwargs["extra_body"] = {"temperature": req.temperature}
    return kwargs


def parse_anthropic_message(msg: Any) -> Completion:
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
    u = msg.usage
    return Completion(
        text=text,
        usage=Usage(getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0),
        stop_reason=getattr(msg, "stop_reason", None),
    )


class AnthropicProvider(Provider):
    name = "anthropic"
    sdk_module = "anthropic"

    def _make_client(self) -> Any:
        import anthropic

        kwargs: dict[str, Any] = {"timeout": self.timeout_s, "max_retries": 2}
        if base_url := self.spec.options.get("base_url"):
            kwargs["base_url"] = base_url
        return anthropic.AsyncAnthropic(**kwargs)

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set"
        return True, "ok"

    def _translate(self, exc: Exception) -> ProviderError:
        import anthropic

        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, anthropic.APIStatusError):
            return self.error(
                f"anthropic {exc.status_code}: {exc.message}",
                kind_for_status(exc.status_code),
                exc.status_code,
            )
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return self.error(f"anthropic: {exc}", ErrorKind.RETRYABLE)
        if isinstance(exc, anthropic.AnthropicError) and "api_key" in str(exc).lower():
            return self.error(f"anthropic: {exc}", ErrorKind.UNAVAILABLE)
        return self.error(f"anthropic: {type(exc).__name__}: {exc}", ErrorKind.RETRYABLE)

    async def complete(self, req: GenerationRequest) -> Completion:
        try:
            msg = await self.client.messages.create(**build_anthropic_kwargs(req))
        except Exception as exc:
            raise self._translate(exc) from exc
        return parse_anthropic_message(msg)

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        try:
            async with self.client.messages.stream(**build_anthropic_kwargs(req)) as s:
                async for text in s.text_stream:
                    yield TextDelta(text)
                final = parse_anthropic_message(await s.get_final_message())
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc
        yield StreamEnd(final.usage, final.stop_reason)
