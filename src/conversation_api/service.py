"""Conversation logic: model resolution, history fitting, fallback, and turn commits.

A turn is atomic: the user message and the reply are saved together, only after a model has answered.
A failed or abandoned turn leaves the conversation exactly as it was, so the history always alternates
user/assistant, which every provider requires.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .providers import (
    PROVIDERS,
    ChatMessage,
    Completion,
    ErrorKind,
    GenerationRequest,
    Provider,
    ProviderError,
    StreamEnd,
    TextDelta,
)
from .providers.mock import approx_tokens
from .registry import ModelRegistry, ModelSpec
from .schemas import (
    Attempt,
    ChatRequest,
    Conversation,
    ModelInfo,
    ModelUsed,
    SendMessage,
    StoredMessage,
    TurnResponse,
    UpdateConversation,
    UsageOut,
    utcnow,
)
from .store import ConversationStore

log = logging.getLogger("conversation_api")


class ConversationNotFound(LookupError):
    pass


class TurnFailed(Exception):
    """Every model in the chain failed, or one failed in a way another model cannot fix."""

    def __init__(self, attempts: list[Attempt], invalid_request: bool = False):
        last = attempts[-1].error if attempts else "no model attempted"
        super().__init__(last)
        self.attempts = attempts
        self.invalid_request = invalid_request


def fit_history(
    messages: list[ChatMessage], system: str | None, budget_tokens: int
) -> tuple[list[ChatMessage], int]:
    """Drop the oldest messages until the estimate fits. Keeps the newest message, and starts on a user turn."""

    def cost(m: ChatMessage) -> int:
        return approx_tokens(m.content) + 4

    total = sum(cost(m) for m in messages) + approx_tokens(system or "")
    start = 0
    while total > budget_tokens and start < len(messages) - 1:
        total -= cost(messages[start])
        start += 1
    while start < len(messages) - 1 and messages[start].role != "user":
        start += 1
    return messages[start:], start


@dataclass
class _Plan:
    conv: Conversation
    content: str
    requested: str
    chain: list[ModelSpec]
    attempts: list[Attempt] = field(default_factory=list)


class ConversationService:
    def __init__(
        self,
        registry: ModelRegistry,
        store: ConversationStore,
        *,
        timeout_s: float = 120.0,
        provider_classes: dict[str, type[Provider]] | None = None,
    ):
        self.registry = registry
        self.store = store
        self.timeout_s = timeout_s
        self.provider_classes = provider_classes or PROVIDERS
        self._providers: dict[tuple[str, str, str], Provider] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._availability: dict[str, tuple[float, tuple[bool, str]]] = {}

    # ---- providers --------------------------------------------------------------------------------
    def provider_for(self, spec: ModelSpec) -> Provider:
        key = (spec.name, spec.provider, spec.model)
        if key not in self._providers:
            self._providers[key] = self.provider_classes[spec.provider](spec, timeout_s=self.timeout_s)
        return self._providers[key]

    def availability(self, spec: ModelSpec, ttl_s: float = 30.0) -> tuple[bool, str]:
        cached = self._availability.get(spec.name)
        if cached and time.monotonic() - cached[0] < ttl_s:
            return cached[1]
        try:
            result = self.provider_classes[spec.provider].availability(spec)
        except Exception as exc:  # an availability probe must never break the listing
            result = (False, f"check failed: {exc}")
        self._availability[spec.name] = (time.monotonic(), result)
        return result

    def list_models(self) -> list[ModelInfo]:
        default = self.registry.resolve(None).name
        out = []
        for spec in self.registry.models.values():
            ok, status = self.availability(spec)
            out.append(
                ModelInfo(
                    name=spec.name,
                    provider=spec.provider,
                    model_id=spec.model,
                    model_source=self.registry.source_of(spec.name),
                    description=spec.description,
                    aliases=self.registry.aliases_for(spec.name),
                    is_default=spec.name == default,
                    available=ok,
                    status=status,
                    fallbacks=spec.fallbacks,
                    context_tokens=spec.context_tokens,
                    max_tokens=spec.max_tokens,
                )
            )
        return out

    def reload(self, new: ModelRegistry) -> list[dict[str, Any]]:
        """Swap in a freshly loaded catalog. In-flight turns finish on the provider they already hold;
        conversations resolve their model name against the new catalog on their next turn."""
        changes = self.registry.describe_changes(new)
        self.registry = new
        self._providers = {}
        self._availability = {}
        log.info("catalog.reloaded", extra={"models": len(new.models), "changes": changes})
        return changes

    # ---- conversation CRUD ---------------------------------------------------------------------
    async def create(
        self, model: str | None = None, system: str | None = None, metadata: dict[str, Any] | None = None
    ) -> Conversation:
        name = (model or self.registry.default).strip()
        self.registry.resolve(name)  # validate now, fail fast with a 400
        conv = Conversation(model=name, system=system or None, metadata=metadata or {})
        await self.store.put(conv)
        log.info("conversation.created", extra={"conversation_id": conv.id, "model": name})
        return conv

    async def get(self, conv_id: str) -> Conversation:
        conv = await self.store.get(conv_id)
        if conv is None:
            raise ConversationNotFound(conv_id)
        return conv

    async def update(self, conv_id: str, body: UpdateConversation) -> Conversation:
        async with self._locks[conv_id]:
            conv = await self.get(conv_id)
            if body.model and body.model != conv.model:
                self.registry.resolve(body.model)
                conv.model_history.append(
                    {
                        "at": utcnow().isoformat(),
                        "from": conv.model,
                        "to": body.model,
                        "after_turn": sum(1 for m in conv.messages if m.role == "user"),
                    }
                )
                log.info(
                    "conversation.model_swapped",
                    extra={"conversation_id": conv.id, "from": conv.model, "to": body.model},
                )
                conv.model = body.model
            if body.system is not None:
                conv.system = body.system or None
            conv.updated_at = utcnow()
            await self.store.put(conv)
            return conv

    async def delete(self, conv_id: str) -> None:
        if not await self.store.delete(conv_id):
            raise ConversationNotFound(conv_id)
        self._locks.pop(conv_id, None)

    # ---- turns -------------------------------------------------------------------------------------
    def _plan(self, conv: Conversation, body: SendMessage) -> _Plan:
        requested = (body.model or conv.model).strip()
        spec = self.registry.resolve(requested)
        return _Plan(
            conv=conv, content=body.content, requested=requested, chain=self.registry.fallback_chain(spec)
        )

    def _request(self, spec: ModelSpec, plan: _Plan, body: SendMessage) -> tuple[GenerationRequest, int]:
        history = [ChatMessage(m.role, m.content) for m in plan.conv.messages] + [
            ChatMessage("user", plan.content)
        ]
        max_tokens = body.max_tokens or spec.max_tokens
        fitted, trimmed = fit_history(history, plan.conv.system, max(1, spec.context_tokens - max_tokens))
        temperature = body.temperature if body.temperature is not None else spec.temperature
        req = GenerationRequest(
            model=spec.model,
            messages=fitted,
            system=plan.conv.system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return req, trimmed

    def _record_failure(self, plan: _Plan, spec: ModelSpec, exc: ProviderError) -> None:
        plan.attempts.append(
            Attempt(model=spec.name, provider=spec.provider, error=str(exc), kind=exc.kind.value)
        )
        log.warning(
            "turn.attempt_failed",
            extra={
                "conversation_id": plan.conv.id,
                "model": spec.name,
                "kind": exc.kind.value,
                "error": str(exc),
            },
        )

    async def _commit(
        self, plan: _Plan, spec: ModelSpec, result: Completion, started: float, trimmed: int
    ) -> TurnResponse:
        conv = plan.conv
        used = ModelUsed(name=spec.name, provider=spec.provider, model_id=spec.model)
        usage = UsageOut(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=spec.cost_usd(result.usage.input_tokens, result.usage.output_tokens),
        )
        reply = StoredMessage(role="assistant", content=result.text, model=used, usage=usage)
        conv.messages += [StoredMessage(role="user", content=plan.content), reply]
        conv.updated_at = utcnow()
        await self.store.put(conv)
        latency = int((time.monotonic() - started) * 1000)
        log.info(
            "turn.completed",
            extra={
                "conversation_id": conv.id,
                "model_requested": plan.requested,
                "model_used": spec.name,
                "fell_back": bool(plan.attempts),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": latency,
            },
        )
        return TurnResponse(
            conversation_id=conv.id,
            message=reply,
            model_requested=plan.requested,
            model_used=used,
            fell_back=bool(plan.attempts),
            attempts=plan.attempts,
            usage=usage,
            latency_ms=latency,
            history_trimmed=trimmed,
        )

    async def send(self, conv_id: str, body: SendMessage) -> TurnResponse:
        async with self._locks[conv_id]:
            plan = self._plan(await self.get(conv_id), body)
            started = time.monotonic()
            for spec in plan.chain:
                req, trimmed = self._request(spec, plan, body)
                try:
                    result = await asyncio.wait_for(self.provider_for(spec).complete(req), self.timeout_s)
                except asyncio.TimeoutError:
                    self._record_failure(
                        plan,
                        spec,
                        ProviderError(
                            f"timed out after {self.timeout_s:.0f}s",
                            kind=ErrorKind.RETRYABLE,
                            provider=spec.provider,
                        ),
                    )
                    continue
                except ProviderError as exc:
                    self._record_failure(plan, spec, exc)
                    if not exc.can_fall_back:
                        raise TurnFailed(plan.attempts, invalid_request=True) from exc
                    continue
                return await self._commit(plan, spec, result, started, trimmed)
            raise TurnFailed(plan.attempts)

    async def send_stream(self, conv_id: str, body: SendMessage) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yields (event, data). Falls back only before the first token; after that the turn is abandoned."""
        async with self._locks[conv_id]:
            plan = self._plan(await self.get(conv_id), body)
            started = time.monotonic()
            yield (
                "start",
                {"conversation_id": conv_id, "model_requested": plan.requested, "model": plan.chain[0].name},
            )
            for i, spec in enumerate(plan.chain):
                req, trimmed = self._request(spec, plan, body)
                parts: list[str] = []
                end = StreamEnd()
                try:
                    async for ev in self.provider_for(spec).stream(req):
                        if isinstance(ev, TextDelta):
                            parts.append(ev.text)
                            yield "delta", {"text": ev.text}
                        else:
                            end = ev
                except ProviderError as exc:
                    self._record_failure(plan, spec, exc)
                    if parts or not exc.can_fall_back or i == len(plan.chain) - 1:
                        yield (
                            "error",
                            {
                                "error": str(exc),
                                "kind": exc.kind.value,
                                "partial": bool(parts),
                                "attempts": [a.model_dump() for a in plan.attempts],
                            },
                        )
                        return
                    yield "fallback", {"from": spec.name, "to": plan.chain[i + 1].name, "error": str(exc)}
                    continue
                result = Completion(text="".join(parts), usage=end.usage, stop_reason=end.stop_reason)
                turn = await self._commit(plan, spec, result, started, trimmed)
                yield "done", turn.model_dump(mode="json")
                return

    async def chat(self, body: ChatRequest) -> tuple[str, SendMessage]:
        """Resolve /v1/chat into (conversation_id, message). `model` is sticky here: it swaps the conversation."""
        if body.conversation_id is None:
            conv = await self.create(body.model, body.system)
        else:
            conv = await self.get(body.conversation_id)
            if (body.model and body.model != conv.model) or body.system is not None:
                conv = await self.update(conv.id, UpdateConversation(model=body.model, system=body.system))
        msg = SendMessage(
            content=body.content, stream=body.stream, max_tokens=body.max_tokens, temperature=body.temperature
        )
        return conv.id, msg
