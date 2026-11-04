"""Provider contract.

Every model backend implements the same two calls, `complete` and `stream`, over a provider-neutral
request. Conversation history is stored in this neutral shape too, which is what makes it safe to swap
the model in the middle of a conversation: nothing in the history belongs to one vendor.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from ..registry import ModelSpec

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationRequest:
    model: str  # provider model ID
    messages: list[ChatMessage]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@dataclass
class TextDelta:
    text: str


@dataclass
class StreamEnd:
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


StreamEvent = TextDelta | StreamEnd


class ErrorKind(str, Enum):
    RETRYABLE = "retryable"  # throttled, overloaded, timeout, 5xx: another model may succeed
    UNAVAILABLE = "unavailable"  # missing SDK, credentials, access or model: another provider may succeed
    INVALID_REQUEST = "invalid_request"  # the request itself is wrong: another model will not help


class ProviderError(Exception):
    def __init__(self, message: str, *, kind: ErrorKind, provider: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.status = status

    @property
    def can_fall_back(self) -> bool:
        return self.kind is not ErrorKind.INVALID_REQUEST


def kind_for_status(status: int | None) -> ErrorKind:
    if status is None or status in (408, 409, 425, 429) or status >= 500:
        return ErrorKind.RETRYABLE
    if status in (401, 403, 404):
        return ErrorKind.UNAVAILABLE
    return ErrorKind.INVALID_REQUEST


class Provider(ABC):
    """One instance per catalog entry. Clients are created lazily so a missing SDK only fails its own model."""

    name: ClassVar[str]
    sdk_module: ClassVar[str | None] = None  # import name checked by `availability`

    def __init__(self, spec: ModelSpec, client: Any | None = None, timeout_s: float = 120.0):
        self.spec = spec
        self.timeout_s = timeout_s
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    @abstractmethod
    def _make_client(self) -> Any: ...

    @abstractmethod
    async def complete(self, req: GenerationRequest) -> Completion: ...

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        """Default: one-shot completion delivered as a single delta. Providers override with real streaming."""
        result = await self.complete(req)
        yield TextDelta(result.text)
        yield StreamEnd(result.usage, result.stop_reason)

    @classmethod
    def availability(cls, spec: ModelSpec) -> tuple[bool, str]:
        """Cheap, local check (no network): is the SDK installed and is there something that looks like a credential?"""
        if cls.sdk_module:
            import importlib.util

            root = cls.sdk_module.split(".")[0]
            if importlib.util.find_spec(root) is None or importlib.util.find_spec(cls.sdk_module) is None:
                return False, f"SDK not installed (pip install '.[{cls.extra_name()}]')"
        return cls.credentials_hint(spec)

    @classmethod
    def credentials_hint(cls, spec: ModelSpec) -> tuple[bool, str]:
        return True, "ok"

    @classmethod
    def extra_name(cls) -> str:
        return cls.name

    def error(self, message: str, kind: ErrorKind, status: int | None = None) -> ProviderError:
        return ProviderError(message, kind=kind, provider=self.name, status=status)


_DONE = object()


async def iterate_in_thread(make_iter: Any) -> AsyncIterator[Any]:
    """Drive a blocking iterator (boto3 event streams) from a worker thread without blocking the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def worker() -> None:
        try:
            iterator: Iterator[Any] = make_iter()
            for item in iterator:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:  # surfaced on the event-loop side
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
