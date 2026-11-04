"""Provider adapters, tested offline with fake SDK clients: request mapping, response parsing, error translation."""

from types import SimpleNamespace as NS
from typing import Any

import pytest
from botocore.exceptions import ClientError

from conversation_api.providers import ChatMessage, ErrorKind, GenerationRequest, StreamEnd, TextDelta
from conversation_api.providers.anthropic import build_anthropic_kwargs, parse_anthropic_message
from conversation_api.providers.base import kind_for_status
from conversation_api.providers.bedrock import (
    BedrockProvider,
    build_converse_kwargs,
    parse_converse_response,
    region_for,
)
from conversation_api.providers.gemini import GeminiProvider, build_gemini_config, to_gemini_contents
from conversation_api.providers.openai_compat import OllamaProvider, OpenAIProvider, to_openai_messages
from conversation_api.registry import ModelSpec

REQ = GenerationRequest(
    model="m-1",
    system="be brief",
    max_tokens=50,
    messages=[ChatMessage("user", "hi"), ChatMessage("assistant", "hello"), ChatMessage("user", "q")],
)


def spec(provider: str, **opts: Any) -> ModelSpec:
    return ModelSpec(name="t", provider=provider, model="m-1", **opts)


async def collect(gen: Any) -> list[Any]:
    return [e async for e in gen]


# ---- Bedrock ---------------------------------------------------------------------------------------
class FakeBedrock:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return {
            "output": {"message": {"content": [{"text": "Hel"}, {"text": "lo"}]}},
            "usage": {"inputTokens": 7, "outputTokens": 2},
            "stopReason": "end_turn",
        }

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {
            "stream": iter(
                [
                    {"messageStart": {"role": "assistant"}},
                    {"contentBlockDelta": {"delta": {"text": "Hel"}}},
                    {"contentBlockDelta": {"delta": {"text": "lo"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 2}}},
                ]
            )
        }


def test_converse_request_shape() -> None:
    kw = build_converse_kwargs(REQ)
    assert kw["system"] == [{"text": "be brief"}]
    assert kw["messages"][1] == {"role": "assistant", "content": [{"text": "hello"}]}
    assert kw["inferenceConfig"] == {"maxTokens": 50}  # temperature only when set
    assert parse_converse_response(FakeBedrock().converse()).text == "Hello"


async def test_bedrock_complete_and_stream() -> None:
    p = BedrockProvider(spec("bedrock"), client=FakeBedrock())
    out = await p.complete(REQ)
    assert (out.text, out.usage.input_tokens, out.stop_reason) == ("Hello", 7, "end_turn")
    events = await collect(p.stream(REQ))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert events[-1] == StreamEnd(usage=out.usage, stop_reason="end_turn")


@pytest.mark.parametrize(
    "code, kind",
    [
        ("ThrottlingException", ErrorKind.RETRYABLE),
        ("AccessDeniedException", ErrorKind.UNAVAILABLE),
        ("ValidationException", ErrorKind.INVALID_REQUEST),
    ],
)
async def test_bedrock_error_translation(code: str, kind: ErrorKind) -> None:
    err = ClientError(
        {"Error": {"Code": code, "Message": "boom"}, "ResponseMetadata": {"HTTPStatusCode": 400}}, "Converse"
    )
    with pytest.raises(Exception) as exc:
        await BedrockProvider(spec("bedrock"), client=FakeBedrock(err)).complete(REQ)
    assert exc.value.kind is kind and code in str(exc.value)


# ---- Anthropic -------------------------------------------------------------------------------------
def test_anthropic_mapping() -> None:
    kw = build_anthropic_kwargs(REQ)
    assert kw["system"] == "be brief" and kw["messages"][0] == {"role": "user", "content": "hi"}
    assert "extra_body" not in kw
    assert build_anthropic_kwargs(GenerationRequest(model="m", messages=[], temperature=0.2))[
        "extra_body"
    ] == {"temperature": 0.2}
    msg = NS(
        content=[NS(type="thinking", text=""), NS(type="text", text="ok")],
        usage=NS(input_tokens=3, output_tokens=1),
        stop_reason="end_turn",
    )
    out = parse_anthropic_message(msg)
    assert (out.text, out.usage.output_tokens) == ("ok", 1)


# ---- OpenAI-compatible ----------------------------------------------------------------------------
class FakeOpenAI:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if kwargs.get("stream"):

            async def gen() -> Any:
                for t in ("Hel", "lo"):
                    yield NS(choices=[NS(delta=NS(content=t), finish_reason=None)], usage=None)
                yield NS(choices=[NS(delta=NS(content=None), finish_reason="stop")], usage=None)
                yield NS(choices=[], usage=NS(prompt_tokens=9, completion_tokens=2))

            return gen()
        return NS(
            choices=[NS(message=NS(content="Hello"), finish_reason="stop")],
            usage=NS(prompt_tokens=9, completion_tokens=2),
        )


async def test_openai_complete_and_stream() -> None:
    fake = FakeOpenAI()
    p = OpenAIProvider(spec("openai"), client=fake)
    out = await p.complete(REQ)
    assert out.text == "Hello" and fake.kwargs["max_completion_tokens"] == 50
    assert fake.kwargs["messages"][0] == {"role": "system", "content": "be brief"}
    events = await collect(p.stream(REQ))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello"
    assert events[-1].usage.input_tokens == 9 and fake.kwargs["stream_options"] == {"include_usage": True}


async def test_ollama_uses_max_tokens_and_no_stream_usage() -> None:
    fake = FakeOpenAI()
    p = OllamaProvider(spec("ollama"), client=fake)
    await collect(p.stream(REQ))
    assert fake.kwargs["max_tokens"] == 50 and "stream_options" not in fake.kwargs


def test_openai_messages_without_system() -> None:
    assert to_openai_messages(GenerationRequest(model="m", messages=[ChatMessage("user", "x")])) == [
        {"role": "user", "content": "x"}
    ]


# ---- Gemini ----------------------------------------------------------------------------------------
class FakeGemini:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.aio = NS(models=NS(generate_content=self.generate, generate_content_stream=self.generate_stream))

    async def generate(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return NS(
            text="Hello",
            usage_metadata=NS(prompt_token_count=4, candidates_token_count=1),
            candidates=[NS(finish_reason=NS(name="STOP"))],
        )

    async def generate_stream(self, **kwargs: Any) -> Any:
        async def gen() -> Any:
            yield NS(text="Hel", usage_metadata=None, candidates=[])
            yield NS(
                text="lo",
                usage_metadata=NS(prompt_token_count=4, candidates_token_count=1),
                candidates=[NS(finish_reason=NS(name="STOP"))],
            )

        return gen()


async def test_gemini_mapping_complete_and_stream() -> None:
    assert to_gemini_contents(REQ)[1] == {"role": "model", "parts": [{"text": "hello"}]}
    assert build_gemini_config(REQ) == {"max_output_tokens": 50, "system_instruction": "be brief"}
    fake = FakeGemini()
    p = GeminiProvider(spec("gemini"), client=fake)
    out = await p.complete(REQ)
    assert (out.text, out.usage.input_tokens, out.stop_reason) == ("Hello", 4, "STOP")
    events = await collect(p.stream(REQ))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello"
    assert events[-1].usage.output_tokens == 1


# ---- shared --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status, kind",
    [
        (429, ErrorKind.RETRYABLE),
        (503, ErrorKind.RETRYABLE),
        (529, ErrorKind.RETRYABLE),
        (None, ErrorKind.RETRYABLE),
        (401, ErrorKind.UNAVAILABLE),
        (404, ErrorKind.UNAVAILABLE),
        (400, ErrorKind.INVALID_REQUEST),
    ],
)
def test_status_classification(status: int | None, kind: ErrorKind) -> None:
    assert kind_for_status(status) is kind


def test_availability_reports_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIProvider.availability(spec("openai")) == (False, "OPENAI_API_KEY is not set")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert OpenAIProvider.availability(spec("openai"))[0]
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert not GeminiProvider.availability(spec("gemini", vertexai=True))[0]


def test_bedrock_region_comes_from_arn_model_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    arn = "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc"
    assert region_for(arn, "ap-south-1") == "us-east-1"
    assert region_for("global.anthropic.claude-sonnet-5-5", "ap-south-1") == "ap-south-1"
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert region_for("us.amazon.nova-pro-v1:0", None) == "eu-west-1"


async def test_invalid_model_id_error_names_model_region_and_fix() -> None:
    err = ClientError(
        {
            "Error": {"Code": "ValidationException", "Message": "The provided model identifier is invalid."},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "Converse",
    )
    p = BedrockProvider(spec("bedrock", region="ap-south-1"), client=FakeBedrock(err))
    with pytest.raises(Exception) as exc:
        await p.complete(REQ)
    msg = str(exc.value)
    assert "model 'm-1', region ap-south-1" in msg and "list-inference-profiles --region ap-south-1" in msg
    assert exc.value.kind is ErrorKind.INVALID_REQUEST
