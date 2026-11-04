"""Deterministic offline model.

It answers from the conversation it is given, so it shows that history survives a model swap: ask it
"what did I say first?" after switching from Bedrock to mock and it can quote your first message.

Test hooks inside a user message:
  #fail          raises a retryable error (exercise fallback)
  #fail-invalid  raises an invalid-request error (no fallback)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from .base import (
    Completion,
    ErrorKind,
    GenerationRequest,
    Provider,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider(Provider):
    name = "mock"

    def _make_client(self) -> Any:
        return None

    def _reply(self, req: GenerationRequest) -> str:
        last = req.messages[-1].content
        if "#fail-invalid" in last:
            raise self.error("mock: request rejected (#fail-invalid)", ErrorKind.INVALID_REQUEST, 400)
        if "#fail" in last:
            raise self.error("mock: simulated overload (#fail)", ErrorKind.RETRYABLE, 529)
        user_turns = [m.content for m in req.messages if m.role == "user"]
        parts = [f"[{self.spec.name} · turn {len(user_turns)}] You said: {last!r}."]
        if len(user_turns) > 1:
            parts.append(f"Your first message in this conversation was: {user_turns[0]!r}.")
        if req.system:
            parts.append(f"(system prompt: {req.system[:60]!r})")
        text = " ".join(parts)
        words = text.split(" ")
        return " ".join(words[: max(1, req.max_tokens)])

    async def complete(self, req: GenerationRequest) -> Completion:
        text = self._reply(req)
        usage = Usage(
            input_tokens=sum(approx_tokens(m.content) for m in req.messages)
            + approx_tokens(req.system or ""),
            output_tokens=approx_tokens(text),
        )
        return Completion(text=text, usage=usage, stop_reason="end_turn")

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        result = await self.complete(req)
        words = result.text.split(" ")
        for i, word in enumerate(words):
            yield TextDelta(word if i == 0 else " " + word)
            await asyncio.sleep(0.01)
        yield StreamEnd(result.usage, result.stop_reason)
