from typing import Any

from fastapi.testclient import TestClient

from conversation_api.app import create_app
from conversation_api.cli import parse_sse
from conversation_api.config import Settings
from conversation_api.registry import ModelRegistry
from conversation_api.service import ConversationService
from conversation_api.store import SqliteStore

from .conftest import TEST_PROVIDERS


def sse_events(client: TestClient, path: str, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    with client.stream("POST", path, json=body | {"stream": True}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        return list(parse_sse(r.iter_lines()))


def test_health_and_models(client: TestClient) -> None:
    assert client.get("/healthz").json()["default_model"] == "primary"
    models = {m["name"]: m for m in client.get("/v1/models").json()}
    assert models["primary"]["is_default"] and models["primary"]["aliases"] == ["smart"]
    assert models["mock"]["available"]


def test_conversation_lifecycle_with_swap(client: TestClient) -> None:
    conv = client.post("/v1/conversations", json={"model": "smart", "system": "be brief"}).json()
    t1 = client.post(f"/v1/conversations/{conv['id']}/messages", json={"content": "hello"}).json()
    assert t1["model_used"]["name"] == "primary" and t1["usage"]["input_tokens"] == 10

    swapped = client.patch(f"/v1/conversations/{conv['id']}", json={"model": "mock"}).json()
    assert swapped["model"] == "mock" and len(swapped["messages"]) == 2

    t2 = client.post(
        f"/v1/conversations/{conv['id']}/messages", json={"content": "what did I say first?"}
    ).json()
    assert t2["model_used"]["provider"] == "mock"
    assert "'hello'" in t2["message"]["content"]  # the new model saw the old model's conversation

    listed = client.get("/v1/conversations").json()
    assert listed[0]["id"] == conv["id"] and listed[0]["turns"] == 2
    assert client.delete(f"/v1/conversations/{conv['id']}").status_code == 204
    assert client.get(f"/v1/conversations/{conv['id']}").status_code == 404


def test_errors_are_structured(client: TestClient) -> None:
    r = client.post("/v1/conversations", json={"model": "nope"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_model"
    assert "smart" in r.json()["error"]["available"]

    assert client.post("/v1/conversations/conv_missing/messages", json={"content": "x"}).status_code == 404
    assert (
        client.post(
            "/v1/conversations/conv_missing/messages", json={"content": "x", "stream": True}
        ).status_code
        == 404
    )

    conv = client.post("/v1/conversations", json={"model": "dead"}).json()
    r = client.post(f"/v1/conversations/{conv['id']}/messages", json={"content": "x"})
    assert r.status_code == 502 and r.json()["error"]["attempts"][0]["model"] == "dead"

    conv = client.post("/v1/conversations", json={"model": "picky"}).json()
    r = client.post(f"/v1/conversations/{conv['id']}/messages", json={"content": "x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"

    assert client.post("/v1/chat", json={"content": "x" * 40_000}).status_code == 413
    assert client.post("/v1/chat", json={"content": ""}).status_code == 422


def test_streaming(client: TestClient) -> None:
    conv = client.post("/v1/conversations", json={"model": "down"}).json()
    events = sse_events(client, f"/v1/conversations/{conv['id']}/messages", {"content": "hi"})
    names = [e for e, _ in events]
    assert names[0] == "start" and names[1] == "fallback" and names[-1] == "done"
    text = "".join(d["text"] for e, d in events if e == "delta")
    assert text == "backup streamed" == events[-1][1]["message"]["content"]


def test_chat_endpoint_creates_then_continues_with_sticky_model(client: TestClient) -> None:
    first = client.post("/v1/chat", json={"content": "hi", "model": "fast"}).json()
    cid = first["conversation_id"]
    assert first["model_used"]["name"] == "other"
    second = client.post(
        "/v1/chat", json={"conversation_id": cid, "content": "again", "model": "mock"}
    ).json()
    assert second["model_used"]["name"] == "mock"
    conv = client.get(f"/v1/conversations/{cid}").json()
    assert conv["model"] == "mock" and len(conv["model_history"]) == 1 and len(conv["messages"]) == 4

    events = sse_events(client, "/v1/chat", {"conversation_id": cid, "content": "stream it"})
    assert events[-1][0] == "done"


def test_api_key_required_when_configured(make_client: Any) -> None:
    with make_client(api_key="s3cret") as c:
        assert c.get("/healthz").status_code == 200  # health stays open for probes
        r = c.get("/v1/models")
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
        assert c.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_sqlite_store_survives_restart(tmp_path: Any, registry: ModelRegistry) -> None:
    db = tmp_path / "c.db"
    with TestClient(
        create_app(
            Settings(_env_file=None),
            service=ConversationService(registry, SqliteStore(db), provider_classes=TEST_PROVIDERS),
        )
    ) as c:
        cid = c.post("/v1/chat", json={"content": "remember me"}).json()["conversation_id"]
    with TestClient(
        create_app(
            Settings(_env_file=None),
            service=ConversationService(registry, SqliteStore(db), provider_classes=TEST_PROVIDERS),
        )
    ) as c:
        conv = c.get(f"/v1/conversations/{cid}").json()
        assert conv["messages"][0]["content"] == "remember me"


def test_app_builds_from_settings_and_shipped_catalog(tmp_path: Any) -> None:
    settings = Settings(_env_file=None, store="sqlite", sqlite_path=tmp_path / "x.db")
    with TestClient(create_app(settings)) as c:
        assert c.get("/healthz").json() == {
            "status": "ok",
            "version": "0.1.0",
            "default_model": "mock",
            "store": "sqlite",
        }
        turn = c.post("/v1/chat", json={"content": "hi"}).json()
        assert turn["model_used"]["provider"] == "mock"
