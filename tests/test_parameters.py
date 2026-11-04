"""Parameter Store references in the catalog, against a moto-simulated SSM (no AWS account needed)."""

import time
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from conversation_api.app import create_app
from conversation_api.config import Settings, default_models_file
from conversation_api.parameters import ParameterError, SsmResolver
from conversation_api.registry import CatalogError, ModelRegistry
from conversation_api.service import ConversationService
from conversation_api.store import MemoryStore

from .conftest import TEST_PROVIDERS

PARAM = "/conversation-api/models/sonnet"


@pytest.fixture
def ssm(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        client = boto3.client("ssm", region_name="us-east-1")
        client.put_parameter(
            Name=PARAM,
            Value="arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc",
            Type="String",
        )
        yield client


def catalog(model_ref: str = f"ssm:{PARAM}") -> dict[str, Any]:
    return {
        "default": "smart",
        "models": {
            "mine": {"provider": "scripted", "model": model_ref, "fallbacks": ["backup"]},
            "backup": {"provider": "scripted", "model": "b-1"},
        },
        "aliases": {"smart": "mine"},
    }


def test_resolver_reads_string_and_securestring(ssm: Any) -> None:
    ssm.put_parameter(Name="/x/secret", Value="  s3cret-id  ", Type="SecureString")
    r = SsmResolver()
    assert r.get(PARAM).endswith("application-inference-profile/abc")
    assert r.get("/x/secret") == "s3cret-id"  # decrypted and trimmed
    ssm.put_parameter(Name="/x/quoted", Value='"us.anthropic.claude-sonnet-5-5"', Type="String")
    assert r.get("/x/quoted") == "us.anthropic.claude-sonnet-5-5"


def test_resolver_explains_missing_parameter(ssm: Any) -> None:
    with pytest.raises(ParameterError, match="not found"):
        SsmResolver().get("/does/not/exist")


def test_resolver_uses_region_from_arn() -> None:
    seen: list[str | None] = []

    class Fake:
        def get_parameter(self, **kw: Any) -> dict[str, Any]:
            return {"Parameter": {"Value": "v"}}

    r = SsmResolver(region="us-east-1", client_factory=lambda region: seen.append(region) or Fake())
    r.get("arn:aws:ssm:ap-south-1:111122223333:parameter/convo/model")
    r.get("/plain/name")
    assert seen == ["ap-south-1", "us-east-1"]


def test_catalog_value_from_parameter_store(ssm: Any) -> None:
    reg = ModelRegistry.from_dict(catalog(), env={})
    spec = reg.resolve("smart")
    assert spec.model == "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc"
    assert reg.source_of("mine") == f"ssm:{PARAM}"
    assert reg.source_of("backup") is None


def test_reference_can_come_from_an_env_var(ssm: Any) -> None:
    """The documented path: export BEDROCK_SONNET_ID=ssm:/... and the shipped catalog picks it up."""
    reg = ModelRegistry.from_file(default_models_file(), env={"BEDROCK_SONNET_ID": f"ssm:{PARAM}"})
    assert reg.resolve("smart").model.endswith("application-inference-profile/abc")
    assert reg.source_of("claude-sonnet") == f"ssm:{PARAM}"


def test_default_model_name_can_come_from_parameter_store(ssm: Any) -> None:
    ssm.put_parameter(Name="/convo/default", Value="backup", Type="String")
    reg = ModelRegistry.from_dict(catalog(), env={}, default_override="ssm:/convo/default")
    assert reg.resolve(None).name == "backup"


def test_missing_parameter_fails_the_load_with_a_clear_error(ssm: Any) -> None:
    with pytest.raises(CatalogError, match=r"Parameter Store: parameter '/nope' .* not found"):
        ModelRegistry.from_dict(catalog("ssm:/nope"), env={})


def test_catalog_without_references_never_calls_aws() -> None:
    def boom() -> Any:
        raise AssertionError("resolver should not be created")

    reg = ModelRegistry.from_dict(catalog("plain-id"), env={}, resolver_factory=boom)
    assert reg.resolve("smart").model == "plain-id" and reg.sources == {}


def _client(loader: Any, **settings: Any) -> TestClient:
    service = ConversationService(loader(), MemoryStore(), provider_classes=TEST_PROVIDERS)
    return TestClient(
        create_app(Settings(_env_file=None, **settings), service=service, registry_loader=loader)
    )


def test_reload_endpoint_picks_up_a_changed_parameter(ssm: Any) -> None:
    loader = lambda: ModelRegistry.from_dict(catalog(), env={})  # noqa: E731
    with _client(loader) as c:
        models = {m["name"]: m for m in c.get("/v1/models").json()}
        assert models["mine"]["model_source"] == f"ssm:{PARAM}"
        cid = c.post("/v1/chat", json={"content": "hi"}).json()["conversation_id"]

        ssm.put_parameter(Name=PARAM, Value="us.anthropic.new-model-v2", Type="String", Overwrite=True)
        r = c.post("/v1/admin/reload-models").json()
        assert r["changes"] == [
            {
                "model": "mine",
                "from": "scripted:arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc",
                "to": "scripted:us.anthropic.new-model-v2",
            }
        ]
        assert r["parameters"] == [f"ssm:{PARAM}"]

        # the existing conversation moves to the new model ID on its next turn
        turn = c.post(f"/v1/conversations/{cid}/messages", json={"content": "again"}).json()
        assert turn["model_used"]["model_id"] == "us.anthropic.new-model-v2"


def test_failed_reload_keeps_the_previous_catalog(ssm: Any) -> None:
    loader = lambda: ModelRegistry.from_dict(catalog(), env={})  # noqa: E731
    with _client(loader) as c:
        ssm.delete_parameter(Name=PARAM)
        r = c.post("/v1/admin/reload-models")
        assert r.status_code == 503 and r.json()["error"]["code"] == "reload_failed"
        assert "not found" in r.json()["error"]["message"]
        assert (
            c.post("/v1/chat", json={"content": "still works"})
            .json()["model_used"]["model_id"]
            .endswith("/abc")
        )


def test_periodic_refresh(ssm: Any) -> None:
    loader = lambda: ModelRegistry.from_dict(catalog(), env={})  # noqa: E731
    with _client(loader, catalog_refresh_s=1) as c:
        ssm.put_parameter(Name=PARAM, Value="refreshed-id", Type="String", Overwrite=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if {m["name"]: m for m in c.get("/v1/models").json()}["mine"]["model_id"] == "refreshed-id":
                break
            time.sleep(0.2)
        else:
            pytest.fail("catalog was not refreshed")


def test_empty_ssm_reference_fails_fast_instead_of_reaching_bedrock() -> None:
    """export BEDROCK_SONNET_ID="ssm:$PARAM" in a shell where $PARAM is empty gives just "ssm:"."""
    with pytest.raises(CatalogError, match=r"models\.claude-sonnet\.model is 'ssm:' with no parameter name"):
        ModelRegistry.from_file(default_models_file(), env={"BEDROCK_SONNET_ID": "ssm:"})
