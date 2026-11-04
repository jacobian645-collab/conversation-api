from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conversation_api.app import create_app
from conversation_api.config import Settings
from conversation_api.providers import (
    PROVIDERS,
    ErrorKind,
    Provider,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)
from conversation_api.providers.base import Completion, GenerationRequest
from conversation_api.registry import ModelRegistry
from conversation_api.service import ConversationService
from conversation_api.store import MemoryStore


class ScriptedProvider(Provider):
    """Test double. Behaviour comes from catalog options: fail=retryable|unavailable|invalid, fail_mid_stream=true."""

    name = "scripted"
    calls: list[tuple[str, GenerationRequest]] = []

    def _make_client(self) -> Any:
        return None

    def _maybe_fail(self) -> None:
        fail = self.spec.options.get("fail")
        if fail:
            raise self.error(
                f"{self.spec.name} failed ({fail})",
                ErrorKind(
                    {"retryable": "retryable", "unavailable": "unavailable", "invalid": "invalid_request"}[
                        fail
                    ]
                ),
            )

    async def complete(self, req: GenerationRequest) -> Completion:
        ScriptedProvider.calls.append((self.spec.name, req))
        self._maybe_fail()
        return Completion(
            text=f"{self.spec.name} reply #{len(req.messages)}", usage=Usage(10, 5), stop_reason="end"
        )

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamEvent]:
        ScriptedProvider.calls.append((self.spec.name, req))
        self._maybe_fail()
        yield TextDelta(f"{self.spec.name} ")
        if self.spec.options.get("fail_mid_stream"):
            raise self.error("connection dropped", ErrorKind.RETRYABLE)
        yield TextDelta("streamed")
        yield StreamEnd(Usage(10, 2), "end")


CATALOG: dict[str, Any] = {
    "default": "primary",
    "models": {
        "primary": {
            "provider": "scripted",
            "model": "p-1",
            "fallbacks": ["backup"],
            "price_per_mtok": {"input": 3.0, "output": 15.0},
        },
        "backup": {"provider": "scripted", "model": "b-1"},
        "other": {"provider": "scripted", "model": "o-1"},
        "down": {"provider": "scripted", "model": "d-1", "fail": "retryable", "fallbacks": ["backup"]},
        "noauth": {"provider": "scripted", "model": "n-1", "fail": "unavailable", "fallbacks": ["backup"]},
        "picky": {"provider": "scripted", "model": "x-1", "fail": "invalid", "fallbacks": ["backup"]},
        "dead": {"provider": "scripted", "model": "z-1", "fail": "retryable"},
        "flaky-stream": {
            "provider": "scripted",
            "model": "f-1",
            "fail_mid_stream": True,
            "fallbacks": ["backup"],
        },
        "tiny": {"provider": "scripted", "model": "t-1", "context_tokens": 60, "max_tokens": 10},
        "mock": {"provider": "mock", "model": "echo-v1"},
    },
    "aliases": {"smart": "primary", "fast": "other"},
}

TEST_PROVIDERS = {**PROVIDERS, "scripted": ScriptedProvider}


@pytest.fixture(autouse=True)
def _scripted_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Registry validation checks provider names against PROVIDERS, so register the double there too.
    monkeypatch.setitem(PROVIDERS, "scripted", ScriptedProvider)
    ScriptedProvider.calls = []
    yield


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_dict(CATALOG, env={})


@pytest.fixture
def service(registry: ModelRegistry) -> ConversationService:
    return ConversationService(registry, MemoryStore(), provider_classes=TEST_PROVIDERS)


@pytest.fixture
def make_client(service: ConversationService) -> Any:
    def _make(**settings: Any) -> TestClient:
        return TestClient(create_app(Settings(_env_file=None, **settings), service=service))

    return _make


@pytest.fixture
def client(make_client: Any) -> Iterator[TestClient]:
    with make_client() as c:
        yield c
