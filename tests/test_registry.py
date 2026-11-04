from pathlib import Path

import pytest

from conversation_api.config import default_models_file
from conversation_api.registry import CatalogError, ModelRegistry, UnknownModelError

from .conftest import CATALOG


def test_alias_and_name_resolve_to_same_spec(registry: ModelRegistry) -> None:
    assert registry.resolve("smart") is registry.resolve("primary")
    assert registry.resolve(None).name == "primary"  # default


def test_adhoc_model_keeps_colons_in_model_id(registry: ModelRegistry) -> None:
    spec = registry.resolve("bedrock:us.meta.llama3-3-70b-instruct-v1:0")
    assert (spec.provider, spec.model) == ("bedrock", "us.meta.llama3-3-70b-instruct-v1:0")


def test_adhoc_can_be_disabled() -> None:
    reg = ModelRegistry.from_dict(CATALOG, env={}, allow_adhoc=False)
    with pytest.raises(UnknownModelError):
        reg.resolve("openai:gpt-x")


def test_unknown_model_lists_known_names(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownModelError) as exc:
        registry.resolve("nope")
    assert "smart" in exc.value.known and "primary" in exc.value.known
    with pytest.raises(UnknownModelError):
        registry.resolve("notaprovider:thing")


def test_env_interpolation_with_defaults() -> None:
    raw = {
        "default": "${DEF:-a}",
        "models": {"a": {"provider": "mock", "model": "${MODEL_ID:-fallback-id}", "region": "${REGION}"}},
    }
    reg = ModelRegistry.from_dict(raw, env={"MODEL_ID": "from-env"})
    assert reg.resolve("a").model == "from-env"
    assert "region" not in reg.resolve("a").options  # unset and no default -> dropped
    assert ModelRegistry.from_dict(raw, env={}).resolve("a").model == "fallback-id"


def test_unknown_keys_become_provider_options() -> None:
    reg = ModelRegistry.from_dict(
        {"models": {"a": {"provider": "bedrock", "model": "m", "region": "eu-west-1"}}}, env={}
    )
    assert reg.resolve("a").options == {"region": "eu-west-1"}


def test_fallback_chain_dedupes_and_orders(registry: ModelRegistry) -> None:
    assert [s.name for s in registry.fallback_chain(registry.resolve("smart"))] == ["primary", "backup"]


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"models": {"a": {"provider": "nope", "model": "m"}}}, "unknown provider"),
        ({"models": {"a": {"provider": "mock"}}}, "needs at least"),
        ({"models": {"a": {"provider": "mock", "model": "m"}}, "aliases": {"x": "missing"}}, "unknown model"),
        ({"models": {"a": {"provider": "mock", "model": "m", "fallbacks": ["ghost"]}}}, "unknown fallback"),
        ({"default": "ghost", "models": {"a": {"provider": "mock", "model": "m"}}}, "default model"),
        ({"models": {}}, "no models"),
    ],
)
def test_catalog_validation(bad: dict, match: str) -> None:
    with pytest.raises(CatalogError, match=match):
        ModelRegistry.from_dict(bad, env={})


def test_shipped_catalog_loads_and_defaults_to_offline_model() -> None:
    path = default_models_file()
    assert path == Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    reg = ModelRegistry.from_file(path, env={})
    assert reg.default == "mock"
    assert {"smart", "fast", "offline"} <= set(reg.aliases)
    assert {s.provider for s in reg.models.values()} >= {
        "bedrock",
        "anthropic",
        "openai",
        "azure_openai",
        "gemini",
        "ollama",
        "mock",
    }


def test_default_override_wins(registry: ModelRegistry) -> None:
    reg = ModelRegistry.from_dict(CATALOG, env={}, default_override="fast")
    assert reg.resolve(None).name == "other"
