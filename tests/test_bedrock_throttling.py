"""Adaptive retries and the ThrottlingException → smaller-model fallback in the Bedrock adapter."""

from collections.abc import Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError

from conversation_api.providers import ErrorKind, GenerationRequest, ModelSwitch, StreamEnd, TextDelta
from conversation_api.providers import bedrock as bedrock_mod
from conversation_api.providers.base import ChatMessage
from conversation_api.providers.bedrock import BedrockProvider, BedrockThrottled, retry_config, shared_client
from conversation_api.registry import CatalogError, ModelRegistry, ModelSpec
from conversation_api.schemas import SendMessage
from conversation_api.service import ConversationService, TurnFailed
from conversation_api.store import MemoryStore

from .conftest import TEST_PROVIDERS

BIG, SMALL = "us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-haiku-4-5"
REQ = GenerationRequest(model=BIG, messages=[ChatMessage("user", "hi")], max_tokens=50)


def client_error(code: str, message: str = "boom") -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": 429 if code == "ThrottlingException" else 400},
        },
        "Converse",
    )


class FakeBedrock:
    """behaviour per model ID: None (ok) | 'throttle' | 'invalid' | 'throttle_event' | 'throttle_after_text'."""

    def __init__(self, **behaviour: str | None):
        self.behaviour = {BIG: behaviour.get("big"), SMALL: behaviour.get("small")}
        self.calls: list[tuple[str, str]] = []

    def _check(self, op: str, model_id: str) -> str | None:
        self.calls.append((op, model_id))
        b = self.behaviour.get(model_id)
        if b == "throttle":
            raise client_error("ThrottlingException", "Too many requests, please wait before trying again.")
        if b == "invalid":
            raise client_error("ValidationException", "The provided model identifier is invalid.")
        return b

    def converse(self, **kw: Any) -> dict[str, Any]:
        self._check("converse", kw["modelId"])
        return {
            "output": {"message": {"content": [{"text": f"answer from {kw['modelId']}"}]}},
            "usage": {"inputTokens": 5, "outputTokens": 3},
            "stopReason": "end_turn",
        }

    def converse_stream(self, **kw: Any) -> dict[str, Any]:
        b = self._check("converse_stream", kw["modelId"])

        def events() -> Iterator[dict[str, Any]]:
            if b == "throttle_event":
                yield {"throttlingException": {"message": "Too many tokens"}}
            yield {"contentBlockDelta": {"delta": {"text": "from "}}}
            if b == "throttle_after_text":
                yield {"throttlingException": {"message": "Too many tokens"}}
            yield {"contentBlockDelta": {"delta": {"text": kw["modelId"]}}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}}

        return {"stream": events()}


def provider(fake: FakeBedrock, small: str | None = SMALL, **opts: Any) -> BedrockProvider:
    if small:
        opts["throttle_fallback_model"] = small
    return BedrockProvider(ModelSpec(name="sonnet", provider="bedrock", model=BIG, **opts), client=fake)


async def collect(gen: Any) -> list[Any]:
    return [e async for e in gen]


# ---- adaptive retry config -----------------------------------------------------------------------
def test_retry_mode_defaults_to_adaptive_and_is_configurable() -> None:
    assert retry_config({}) == {"mode": "adaptive", "max_attempts": 3}
    assert retry_config({"retry_mode": "standard", "max_attempts": "5"}) == {
        "mode": "standard",
        "max_attempts": 5,
    }
    with pytest.raises(ValueError, match="retry_mode"):
        retry_config({"retry_mode": "turbo"})


def test_bad_retry_mode_fails_catalog_load() -> None:
    with pytest.raises(CatalogError, match="retry_mode"):
        ModelRegistry.from_dict(
            {"models": {"a": {"provider": "bedrock", "model": "m", "retry_mode": "turbo"}}}, env={}
        )


@pytest.fixture
def fresh_clients(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(bedrock_mod, "_clients", {})
    yield


def test_real_boto3_client_is_adaptive_and_shared_per_region(fresh_clients: None) -> None:
    retries = retry_config({})
    a = shared_client("us-east-1", retries, 60.0)
    # botocore: max_attempts=3 retries -> 4 calls in total
    assert a.meta.config.retries == {"mode": "adaptive", "total_max_attempts": 4}
    assert shared_client("us-east-1", retries, 60.0) is a  # one rate limiter per region
    assert shared_client("ap-south-1", retries, 60.0) is not a


def test_provider_uses_arn_region_for_its_client(fresh_clients: None) -> None:
    arn = "arn:aws:bedrock:eu-west-1:111122223333:application-inference-profile/x"
    p = BedrockProvider(ModelSpec(name="x", provider="bedrock", model=arn, region="us-east-1"))
    assert p.client.meta.region_name == "eu-west-1"


# ---- Converse: try/except ThrottlingException → smaller model -------------------------------------
async def test_throttled_request_is_answered_by_smaller_model() -> None:
    fake = FakeBedrock(big="throttle")
    out = await provider(fake).complete(REQ)
    assert out.text == f"answer from {SMALL}" and out.model_id == SMALL
    assert "ThrottlingException" in (out.fallback_reason or "")
    assert fake.calls == [("converse", BIG), ("converse", SMALL)]


async def test_throttled_without_smaller_model_raises_retryable() -> None:
    with pytest.raises(BedrockThrottled) as exc:
        await provider(FakeBedrock(big="throttle"), small=None).complete(REQ)
    assert exc.value.kind is ErrorKind.RETRYABLE and exc.value.code == "ThrottlingException"


async def test_smaller_model_also_throttled_raises_with_its_id() -> None:
    with pytest.raises(BedrockThrottled, match=SMALL):
        await provider(FakeBedrock(big="throttle", small="throttle")).complete(REQ)


async def test_other_errors_do_not_downshift() -> None:
    fake = FakeBedrock(big="invalid")
    with pytest.raises(Exception) as exc:
        await provider(fake).complete(REQ)
    assert exc.value.kind is ErrorKind.INVALID_REQUEST and fake.calls == [("converse", BIG)]


# ---- ConverseStream -------------------------------------------------------------------------------
@pytest.mark.parametrize("behaviour", ["throttle", "throttle_event"])
async def test_stream_switches_to_smaller_model_before_first_token(behaviour: str) -> None:
    events = await collect(provider(FakeBedrock(big=behaviour)).stream(REQ))
    assert isinstance(events[0], ModelSwitch) and (events[0].from_model, events[0].to_model) == (BIG, SMALL)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == f"from {SMALL}"
    assert isinstance(events[-1], StreamEnd) and events[-1].model_id == SMALL


async def test_stream_throttled_after_text_does_not_switch() -> None:
    gen = provider(FakeBedrock(big="throttle_after_text")).stream(REQ)
    got: list[Any] = []
    with pytest.raises(BedrockThrottled):
        async for e in gen:
            got.append(e)
    assert [type(e) for e in got] == [TextDelta] and not any(isinstance(e, ModelSwitch) for e in got)


# ---- end to end through the service ---------------------------------------------------------------
def service_with(fake: FakeBedrock, catalog: dict[str, Any]) -> ConversationService:
    class InjectedBedrock(BedrockProvider):
        def __init__(self, spec: ModelSpec, **kw: Any):
            super().__init__(spec, client=fake, **kw)

    providers = {**TEST_PROVIDERS, "bedrock": InjectedBedrock}
    return ConversationService(
        ModelRegistry.from_dict(catalog, env={}), MemoryStore(), provider_classes=providers
    )


CATALOG = {
    "default": "sonnet",
    "models": {
        "sonnet": {
            "provider": "bedrock",
            "model": BIG,
            "throttle_fallback_model": SMALL,
            "price_per_mtok": {"input": 3, "output": 15},
        },
    },
}


async def test_turn_reports_the_smaller_model_that_answered() -> None:
    svc = service_with(FakeBedrock(big="throttle"), CATALOG)
    conv = await svc.create()
    turn = await svc.send(conv.id, SendMessage(content="hi"))
    assert turn.model_used.name == "sonnet" and turn.model_used.model_id == SMALL
    assert turn.fell_back and "ThrottlingException" in turn.attempts[0].error
    assert turn.usage.cost_usd is None  # sonnet's prices don't apply to haiku's tokens
    assert (await svc.get(conv.id)).messages[-1].model.model_id == SMALL


async def test_stream_turn_emits_fallback_event_and_records_smaller_model() -> None:
    svc = service_with(FakeBedrock(big="throttle_event"), CATALOG)
    conv = await svc.create()
    events = [e async for e in svc.send_stream(conv.id, SendMessage(content="hi"))]
    names = [e[0] for e in events]
    assert names[:2] == ["start", "fallback"] and names[-1] == "done"
    assert events[1][1]["to"] == SMALL and events[-1][1]["model_used"]["model_id"] == SMALL


async def test_misconfigured_fallback_model_does_not_mask_the_real_failure() -> None:
    """Primary throttled with no smaller model; first catalog fallback has an invalid ID; second answers."""
    catalog = {
        "default": "sonnet",
        "models": {
            "sonnet": {"provider": "bedrock", "model": BIG, "fallbacks": ["bad-haiku", "mock"]},
            "bad-haiku": {"provider": "bedrock", "model": SMALL},
            "mock": {"provider": "mock", "model": "echo-v1"},
        },
    }
    svc = service_with(FakeBedrock(big="throttle", small="invalid"), catalog)
    conv = await svc.create()
    turn = await svc.send(conv.id, SendMessage(content="hi"))
    assert turn.model_used.name == "mock"
    assert [a.kind for a in turn.attempts] == ["retryable", "invalid_request"]

    svc = service_with(
        FakeBedrock(big="throttle", small="invalid"),
        {
            **catalog,
            "models": {
                **catalog["models"],
                "sonnet": {"provider": "bedrock", "model": BIG, "fallbacks": ["bad-haiku"]},
            },
        },
    )
    conv = await svc.create()
    with pytest.raises(TurnFailed) as exc:
        await svc.send(conv.id, SendMessage(content="hi"))
    assert not exc.value.invalid_request  # reported as 502 all_models_failed, not a bad request
