import asyncio

import pytest

from conversation_api.providers import ChatMessage
from conversation_api.schemas import SendMessage, UpdateConversation
from conversation_api.service import ConversationService, TurnFailed, fit_history

from .conftest import ScriptedProvider


async def test_swap_keeps_history_and_routes_next_turn(service: ConversationService) -> None:
    conv = await service.create("smart")
    await service.send(conv.id, SendMessage(content="first"))
    await service.update(conv.id, UpdateConversation(model="fast"))
    turn = await service.send(conv.id, SendMessage(content="second"))

    assert turn.model_used.name == "other"
    name, req = ScriptedProvider.calls[-1]
    assert name == "other"
    assert [m.content for m in req.messages] == [
        "first",
        "primary reply #1",
        "second",
    ]  # full history went over
    conv = await service.get(conv.id)
    assert conv.model == "fast"
    assert conv.model_history[0]["from"] == "smart" and conv.model_history[0]["after_turn"] == 1
    assert [m.model.name for m in conv.messages if m.model] == ["primary", "other"]


async def test_per_turn_override_does_not_change_conversation_model(service: ConversationService) -> None:
    conv = await service.create("smart")
    turn = await service.send(conv.id, SendMessage(content="hi", model="other"))
    assert turn.model_used.name == "other" and turn.model_requested == "other"
    assert (await service.get(conv.id)).model == "smart"


@pytest.mark.parametrize("model", ["down", "noauth"])
async def test_falls_back_on_retryable_and_unavailable(service: ConversationService, model: str) -> None:
    conv = await service.create(model)
    turn = await service.send(conv.id, SendMessage(content="hi"))
    assert turn.fell_back and turn.model_used.name == "backup"
    assert [a.model for a in turn.attempts] == [model]


async def test_invalid_request_does_not_fall_back_and_leaves_history_untouched(
    service: ConversationService,
) -> None:
    conv = await service.create("picky")
    with pytest.raises(TurnFailed) as exc:
        await service.send(conv.id, SendMessage(content="hi"))
    assert exc.value.invalid_request
    assert [c[0] for c in ScriptedProvider.calls] == ["picky"]
    assert (await service.get(conv.id)).messages == []


async def test_all_models_failing_raises_and_commits_nothing(service: ConversationService) -> None:
    conv = await service.create("dead")
    with pytest.raises(TurnFailed) as exc:
        await service.send(conv.id, SendMessage(content="hi"))
    assert not exc.value.invalid_request and len(exc.value.attempts) == 1
    assert (await service.get(conv.id)).messages == []


async def test_cost_is_estimated_from_catalog_prices(service: ConversationService) -> None:
    conv = await service.create("primary")
    turn = await service.send(conv.id, SendMessage(content="hi"))
    assert turn.usage.cost_usd == pytest.approx((10 * 3.0 + 5 * 15.0) / 1_000_000)


def test_fit_history_drops_oldest_and_starts_on_user_turn() -> None:
    msgs = [ChatMessage("user" if i % 2 == 0 else "assistant", "x" * 40) for i in range(7)]  # ~14 tok each
    fitted, trimmed = fit_history(msgs, None, budget_tokens=45)
    assert fitted[0].role == "user" and fitted[-1] is msgs[-1]
    assert trimmed == len(msgs) - len(fitted) and trimmed > 0
    # the newest message always survives, even alone over budget
    assert fit_history([ChatMessage("user", "y" * 4000)], None, 10)[0] == [ChatMessage("user", "y" * 4000)]


async def test_long_history_is_trimmed_for_small_context_models(service: ConversationService) -> None:
    conv = await service.create("tiny")
    for i in range(6):
        turn = await service.send(conv.id, SendMessage(content=f"message number {i} " * 5))
    assert turn.history_trimmed > 0
    assert (
        len((await service.get(conv.id)).messages) == 12
    )  # storage keeps everything; only the request is trimmed


async def test_stream_falls_back_before_first_token(service: ConversationService) -> None:
    conv = await service.create("down")
    events = [e async for e in service.send_stream(conv.id, SendMessage(content="hi"))]
    names = [e[0] for e in events]
    assert names[0] == "start" and "fallback" in names and names[-1] == "done"
    assert events[-1][1]["model_used"]["name"] == "backup"


async def test_stream_failure_after_first_token_is_not_committed(service: ConversationService) -> None:
    conv = await service.create("flaky-stream")
    events = [e async for e in service.send_stream(conv.id, SendMessage(content="hi"))]
    assert events[-1][0] == "error" and events[-1][1]["partial"] is True
    assert "fallback" not in [e[0] for e in events]  # tokens already reached the client; no silent restart
    assert (await service.get(conv.id)).messages == []


async def test_concurrent_turns_on_one_conversation_are_serialised(service: ConversationService) -> None:
    conv = await service.create("primary")
    await asyncio.gather(*(service.send(conv.id, SendMessage(content=f"m{i}")) for i in range(5)))
    roles = [m.role for m in (await service.get(conv.id)).messages]
    assert roles == ["user", "assistant"] * 5
