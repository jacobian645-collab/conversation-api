"""Amazon Bedrock through the Converse API.

Converse gives one request/response shape across Bedrock model families (Anthropic Claude, Amazon Nova,
Meta Llama, Mistral, ...), so swapping between them is a catalog change, not a code change.
Credentials come from the standard AWS chain: in AWS CloudShell they are already there.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from .base import (
    ChatMessage,
    Completion,
    ErrorKind,
    GenerationRequest,
    Provider,
    ProviderError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
    iterate_in_thread,
)

_RETRYABLE = {
    "ThrottlingException",
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


def region_for(model_id: str, configured: str | None) -> str | None:
    """An ARN model ID (inference profile, provisioned throughput, custom model) only works in its own region,
    so its region wins over the configured one."""
    if model_id.startswith("arn:"):
        parts = model_id.split(":")
        if len(parts) > 3 and parts[3]:
            return parts[3]
    return configured or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


def to_converse_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": [{"text": m.content}]} for m in messages]


def build_converse_kwargs(req: GenerationRequest) -> dict[str, Any]:
    inference: dict[str, Any] = {"maxTokens": req.max_tokens}
    if req.temperature is not None:
        inference["temperature"] = req.temperature
    kwargs: dict[str, Any] = {
        "modelId": req.model,
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

    def _make_client(self) -> Any:
        import boto3
        from botocore.config import Config

        region = self.region
        cfg = Config(
            read_timeout=self.timeout_s,
            connect_timeout=10,
            retries={"mode": "adaptive", "max_attempts": 3},
            user_agent_extra="conversation-api",
        )
        return boto3.client("bedrock-runtime", region_name=region or None, config=cfg)

    @property
    def region(self) -> str | None:
        return region_for(self.spec.model, self.spec.options.get("region"))

    @classmethod
    def credentials_hint(cls, spec: Any) -> tuple[bool, str]:
        import boto3

        if boto3.Session().get_credentials() is None:
            return False, "no AWS credentials found (run in AWS CloudShell, or set AWS_PROFILE / keys)"
        return True, "ok"

    def _translate(self, exc: Exception) -> ProviderError:
        from botocore.exceptions import (
            ClientError,
            EndpointConnectionError,
            NoCredentialsError,
            ReadTimeoutError,
        )

        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            detail = exc.response.get("Error", {}).get("Message", str(exc))
            msg = f"bedrock {code} (model {self.spec.model!r}, region {self.region}): {detail}"
            if code == "ValidationException" and "model identifier" in detail.lower():
                msg += (
                    f". Check the ID exists in {self.region}: aws bedrock list-inference-profiles "
                    f"--region {self.region}. Cross-region profile prefixes (us., eu., apac., au., global.) "
                    "must match the region, and an ARN must be from the same account."
                )
            if code in _RETRYABLE:
                return self.error(msg, ErrorKind.RETRYABLE, status)
            if code in _UNAVAILABLE:
                return self.error(msg, ErrorKind.UNAVAILABLE, status)
            return self.error(msg, ErrorKind.INVALID_REQUEST, status)
        if isinstance(exc, NoCredentialsError):
            return self.error("bedrock: no AWS credentials", ErrorKind.UNAVAILABLE)
        if isinstance(exc, (EndpointConnectionError, ReadTimeoutError)):
            return self.error(f"bedrock: {exc}", ErrorKind.RETRYABLE)
        return self.error(f"bedrock: {type(exc).__name__}: {exc}", ErrorKind.RETRYABLE)

    async def complete(self, req: GenerationRequest) -> Completion:
        try:
            resp = await asyncio.to_thread(self.client.converse, **build_converse_kwargs(req))
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc
        return parse_converse_response(resp)

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        kwargs = build_converse_kwargs(req)
        usage, stop = Usage(), None
        try:
            async for event in iterate_in_thread(lambda: self.client.converse_stream(**kwargs)["stream"]):
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
                    for key in (
                        "throttlingException",
                        "serviceUnavailableException",
                        "internalServerException",
                        "modelStreamErrorException",
                        "validationException",
                    ):
                        if key in event:
                            kind = (
                                ErrorKind.INVALID_REQUEST
                                if key == "validationException"
                                else ErrorKind.RETRYABLE
                            )
                            raise self.error(f"bedrock stream {key}: {event[key].get('message')}", kind)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc
        yield StreamEnd(usage, stop)
