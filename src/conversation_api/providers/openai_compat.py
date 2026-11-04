"""OpenAI Chat Completions, and everything that speaks it.

One adapter, three providers:
  openai        api.openai.com (OPENAI_API_KEY)
  azure_openai  an Azure OpenAI resource: API key, or Entra ID (DefaultAzureCredential) when no key is set,
                which is what you have after `az login` or inside Azure Cloud Shell
  ollama        any OpenAI-compatible server (Ollama, vLLM, LM Studio) at `base_url`
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


def to_openai_messages(req: GenerationRequest) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": req.system}] if req.system else []
    msgs += [{"role": m.role, "content": m.content} for m in req.messages]
    return msgs


def parse_chat_completion(resp: Any) -> Completion:
    choice = resp.choices[0]
    u = getattr(resp, "usage", None)
    return Completion(
        text=choice.message.content or "",
        usage=Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0),
        stop_reason=choice.finish_reason,
    )


class OpenAIProvider(Provider):
    name = "openai"
    sdk_module = "openai"
    # Newer OpenAI models take max_completion_tokens; most OpenAI-compatible servers still take max_tokens.
    default_token_param = "max_completion_tokens"
    supports_stream_usage = True

    def _make_client(self) -> Any:
        import openai

        kwargs: dict[str, Any] = {"timeout": self.timeout_s, "max_retries": 2}
        if base_url := self.spec.options.get("base_url"):
            kwargs["base_url"] = base_url
        return openai.AsyncOpenAI(**kwargs)

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set"
        return True, "ok"

    def _kwargs(self, req: GenerationRequest) -> dict[str, Any]:
        token_param = self.spec.options.get("token_param", self.default_token_param)
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": to_openai_messages(req),
            token_param: req.max_tokens,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        return kwargs

    def _translate(self, exc: Exception) -> ProviderError:
        import openai

        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, openai.APIStatusError):
            return self.error(
                f"{self.name} {exc.status_code}: {exc.message}",
                kind_for_status(exc.status_code),
                exc.status_code,
            )
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return self.error(f"{self.name}: {exc}", ErrorKind.RETRYABLE)
        if isinstance(exc, openai.OpenAIError):  # e.g. missing api key at client construction
            return self.error(f"{self.name}: {exc}", ErrorKind.UNAVAILABLE)
        return self.error(f"{self.name}: {type(exc).__name__}: {exc}", ErrorKind.RETRYABLE)

    async def complete(self, req: GenerationRequest) -> Completion:
        try:
            resp = await self.client.chat.completions.create(**self._kwargs(req))
        except Exception as exc:
            raise self._translate(exc) from exc
        return parse_chat_completion(resp)

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        kwargs = self._kwargs(req) | {"stream": True}
        if self.supports_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        usage, stop = Usage(), None
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        yield TextDelta(choice.delta.content)
                    if choice.finish_reason:
                        stop = choice.finish_reason
                if getattr(chunk, "usage", None):
                    usage = Usage(chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc
        yield StreamEnd(usage, stop)


class AzureOpenAIProvider(OpenAIProvider):
    name = "azure_openai"

    @classmethod
    def extra_name(cls) -> str:
        return "azure"

    def _make_client(self) -> Any:
        import openai

        endpoint = self.spec.options.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise self.error("azure_openai: AZURE_OPENAI_ENDPOINT is not set", ErrorKind.UNAVAILABLE)
        kwargs: dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": self.spec.options.get("api_version") or "2024-10-21",
            "timeout": self.timeout_s,
            "max_retries": 2,
        }
        if key := os.environ.get("AZURE_OPENAI_API_KEY"):
            kwargs["api_key"] = key
        else:  # keyless: Entra ID token from az login / managed identity / Cloud Shell
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        return openai.AsyncAzureOpenAI(**kwargs)

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        if not (spec.options.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")):
            return False, "AZURE_OPENAI_ENDPOINT is not set"
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            import importlib.util

            if importlib.util.find_spec("azure.identity") is None:
                return False, "no AZURE_OPENAI_API_KEY and azure-identity not installed"
            return True, "ok (Entra ID)"
        return True, "ok"


class OllamaProvider(OpenAIProvider):
    name = "ollama"
    default_token_param = "max_tokens"
    supports_stream_usage = False

    @classmethod
    def extra_name(cls) -> str:
        return "openai"

    def _make_client(self) -> Any:
        import openai

        return openai.AsyncOpenAI(
            base_url=self.spec.options.get("base_url", "http://localhost:11434/v1"),
            api_key=os.environ.get(
                "OLLAMA_API_KEY", "ollama"
            ),  # Ollama ignores the key; the SDK requires one
            timeout=self.timeout_s,
            max_retries=1,
        )

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        return True, f"not checked (needs a server at {spec.options.get('base_url')})"
