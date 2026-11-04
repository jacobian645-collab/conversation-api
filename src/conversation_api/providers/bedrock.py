"""Amazon Bedrock through the Converse API.

Converse gives one request/response shape across Bedrock model families (Anthropic Claude, Amazon Nova,
Meta Llama, Mistral, ...), so swapping between them is a catalog change, not a code change.
Credentials come from the standard AWS chain: in AWS CloudShell they are already there.

Throttling is handled in two layers:
  1. botocore retries in **adaptive** mode: exponential backoff plus a client-side token bucket that slows
     the client down when Bedrock throttles. Clients are shared per region and retry config, so the rate
     limiter sees all traffic this process sends to a region, not one model's share of it.
  2. If a ThrottlingException still comes back after those retries, the request is sent once to a smaller
     model (`throttle_fallback_model` in the catalog, which can be an `ssm:` reference). The reply records
     which model answered.
Catalog options: retry_mode (legacy | standard | adaptive, default adaptive), max_attempts (retries after
the first call, botocore semantics; default 3 = up to 4 calls), throttle_fallback_model.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any

from .base import (
    ChatMessage,
    Completion,
    ErrorKind,
    GenerationRequest,
    ModelSwitch,
    Provider,
    ProviderError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
    iterate_in_thread,
)

log = logging.getLogger("conversation_api")

THROTTLING = "ThrottlingException"
_RETRYABLE = {
    THROTTLING,
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ServiceQuotaExceededException",
}
_UNAVAILABLE = {
    "AccessDeniedException",
    "ResourceNotFoundException",
    "UnrecognizedClientException",
    "ExpiredTokenException",
}
# ConverseStream reports some failures as events inside the stream instead of raising.
_STREAM_ERRORS = {
    "throttlingException": THROTTLING,
    "serviceUnavailableException": "ServiceUnavailableException",
    "internalServerException": "InternalServerException",
    "modelStreamErrorException": "ModelStreamErrorException",
    "validationException": "ValidationException",
}
RETRY_MODES = ("legacy", "standard", "adaptive")
DEFAULT_RETRY_MODE = "adaptive"
DEFAULT_MAX_ATTEMPTS = 3

_clients: dict[tuple[Any, ...], Any] = {}
_clients_lock = threading.Lock()


class BedrockThrottled(ProviderError):
    """ThrottlingException that survived the SDK's adaptive retries."""


def region_for(model_id: str, configured: str | None) -> str | None:
    """An ARN model ID (inference profile, provisioned throughput, custom model) only works in its own region,
    so its region wins over the configured one."""
    if model_id.startswith("arn:"):
        parts = model_id.split(":")
        if len(parts) > 3 and parts[3]:
            return parts[3]
    return configured or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


def retry_config(options: dict[str, Any]) -> dict[str, Any]:
    mode = str(options.get("retry_mode") or DEFAULT_RETRY_MODE).lower()
    if mode not in RETRY_MODES:
        raise ValueError(f"retry_mode must be one of {', '.join(RETRY_MODES)}, got {mode!r}")
    attempts = int(options.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    if attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    return {"mode": mode, "max_attempts": attempts}


def shared_client(region: str | None, retries: dict[str, Any], timeout_s: float) -> Any:
    """One bedrock-runtime client per (region, retry config, timeout). boto3 clients are thread-safe, and
    sharing one lets adaptive mode's rate limiter see every request this process sends to that region."""
    key = (region, retries["mode"], retries["max_attempts"], timeout_s)
    with _clients_lock:
        if key not in _clients:
            import boto3
            from botocore.config import Config

            cfg = Config(
                read_timeout=timeout_s,
                connect_timeout=10,
                retries=dict(retries),
                user_agent_extra="conversation-api",
            )
            _clients[key] = boto3.client("bedrock-runtime", region_name=region or None, config=cfg)
        return _clients[key]


def to_converse_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": [{"text": m.content}]} for m in messages]


def build_converse_kwargs(req: GenerationRequest, model_id: str | None = None) -> dict[str, Any]:
    inference: dict[str, Any] = {"maxTokens": req.max_tokens}
    if req.temperature is not None:
        inference["temperature"] = req.temperature
    kwargs: dict[str, Any] = {
        "modelId": model_id or req.model,
        "messages": to_converse_messages(req.messages),
        "inferenceConfig": inference,
    }
    if req.system:
        kwargs["system"] = [{"text": req.system}]
    return kwargs


def parse_converse_response(resp: dict[str, Any]) -> Completion:
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    text = "".join(b["text"] for b in blocks if "text" in b)
    usage = resp.get("usage", {})
    return Completion(
        text=text,
        usage=Usage(usage.get("inputTokens", 0), usage.get("outputTokens", 0)),
        stop_reason=resp.get("stopReason"),
    )


class BedrockProvider(Provider):
    name = "bedrock"
    sdk_module = "boto3"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retries = retry_config(self.spec.options)

    @classmethod
    def validate_options(cls, spec: Any) -> None:
        retry_config(spec.options)

    # ---- clients ---------------------------------------------------------------------------------
    def _make_client(self) -> Any:
        return self._client_for(self.spec.model)

    def _client_for(self, model_id: str) -> Any:
        if self._client is not None:  # injected (tests)
            return self._client
        return shared_client(
            region_for(model_id, self.spec.options.get("region")), self.retries, self.timeout_s
        )

    @property
    def region(self) -> str | None:
        return region_for(self.spec.model, self.spec.options.get("region"))

    @property
    def smaller_model(self) -> str | None:
        return self.spec.options.get("throttle_fallback_model") or None

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        import boto3

        if boto3.Session().get_credentials() is None:
            return False, "no AWS credentials found (run in AWS CloudShell, or set AWS_PROFILE / keys)"
        return True, "ok"

    # ---- errors --------------------------------------------------------------------------------------
    def _coded_error(self, code: str, detail: str, model_id: str, status: int | None = None) -> ProviderError:
        region = region_for(model_id, self.spec.options.get("region"))
        msg = f"bedrock {code} (model {model_id!r}, region {region}): {detail}"
        if code == "ValidationException" and "model identifier" in detail.lower():
            msg += (
                f". Check the ID exists in {region}: aws bedrock list-inference-profiles "
                f"--region {region}. Cross-region profile prefixes (us., eu., apac., au., global.) "
                "must match the region, and an ARN must be from the same account."
            )
        if code == THROTTLING:
            return BedrockThrottled(
                msg, kind=ErrorKind.RETRYABLE, provider=self.name, status=status, code=code
            )
        kind = (
            ErrorKind.RETRYABLE
            if code in _RETRYABLE
            else ErrorKind.UNAVAILABLE
            if code in _UNAVAILABLE
            else ErrorKind.INVALID_REQUEST
        )
        return ProviderError(msg, kind=kind, provider=self.name, status=status, code=code)

    def _translate(self, exc: Exception, model_id: str | None = None) -> ProviderError:
        from botocore.exceptions import (
            ClientError,
            EndpointConnectionError,
            NoCredentialsError,
            ReadTimeoutError,
        )

        if isinstance(exc, ProviderError):
            return exc
        model_id = model_id or self.spec.model
        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {})
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            return self._coded_error(err.get("Code", ""), err.get("Message", str(exc)), model_id, status)
        if isinstance(exc, NoCredentialsError):
            return self.error("bedrock: no AWS credentials", ErrorKind.UNAVAILABLE)
        if isinstance(exc, (EndpointConnectionError, ReadTimeoutError)):
            return self.error(f"bedrock: {exc}", ErrorKind.RETRYABLE)
        return self.error(f"bedrock: {type(exc).__name__}: {exc}", ErrorKind.RETRYABLE)

    def _downshift_target(self, model_id: str) -> str | None:
        small = self.smaller_model
        return small if small and small != model_id else None

    # ---- Converse ------------------------------------------------------------------------------------
    async def _converse(self, req: GenerationRequest, model_id: str) -> Completion:
        try:
            resp = await asyncio.to_thread(
                self._client_for(model_id).converse, **build_converse_kwargs(req, model_id)
            )
        except Exception as exc:
            raise self._translate(exc, model_id) from exc
        return parse_converse_response(resp)

    async def complete(self, req: GenerationRequest) -> Completion:
        try:
            return await self._converse(req, req.model)
        except BedrockThrottled as exc:
            # Still throttled after the SDK's adaptive retries: try the smaller model once.
            small = self._downshift_target(req.model)
            if small is None:
                raise
            log.warning(
                "bedrock.throttled_downshift",
                extra={"from_model": req.model, "to_model": small, "error": str(exc)},
            )
        result = await self._converse(req, small)
        result.model_id = small
        result.fallback_reason = f"{THROTTLING} on {req.model}; answered by smaller model {small}"
        return result

    # ---- ConverseStream ----------------------------------------------------------------------------
    async def _converse_stream(self, req: GenerationRequest, model_id: str) -> AsyncIterator[StreamEvent]:
        kwargs = build_converse_kwargs(req, model_id)
        client = self._client_for(model_id)
        usage, stop = Usage(), None
        try:
            async for event in iterate_in_thread(lambda: client.converse_stream(**kwargs)["stream"]):
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"].get("delta", {}).get("text")
                    if text:
                        yield TextDelta(text)
                elif "messageStop" in event:
                    stop = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    u = event["metadata"].get("usage", {})
                    usage = Usage(u.get("inputTokens", 0), u.get("outputTokens", 0))
                else:
                    for key, code in _STREAM_ERRORS.items():
                        if key in event:
                            raise self._coded_error(code, str(event[key].get("message")), model_id)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc, model_id) from exc
        yield StreamEnd(usage, stop, model_id=model_id if model_id != req.model else None)

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        sent_text = False
        try:
            async for ev in self._converse_stream(req, req.model):
                sent_text = sent_text or isinstance(ev, TextDelta)
                yield ev
            return
        except BedrockThrottled as exc:
            # Only before the first token: once text has reached the client, switching would splice answers.
            small = self._downshift_target(req.model)
            if small is None or sent_text:
                raise
            log.warning(
                "bedrock.throttled_downshift",
                extra={"from_model": req.model, "to_model": small, "error": str(exc), "stream": True},
            )
        yield ModelSwitch(req.model, small, f"{THROTTLING} on {req.model}")
        async for ev in self._converse_stream(req, small):
            yield ev
