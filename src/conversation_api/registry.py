"""Model catalog: resolves the name a client asks for to a provider + model ID.

Resolution order for a requested name:
  1. an alias            ("smart" -> "claude-sonnet")
  2. a catalog entry     ("claude-sonnet")
  3. an ad-hoc reference ("bedrock:us.meta.llama3-3-70b-instruct-v1:0"), if allowed by settings

Catalog values can come from the environment (${VAR}, ${VAR:-default}) and from AWS Systems Manager
Parameter Store (`ssm:/path/to/param`); see parameters.py.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .parameters import ParameterError, ParameterResolver, SsmResolver, resolve_references
from .providers import PROVIDERS

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SPEC_FIELDS = {
    "provider",
    "model",
    "description",
    "max_tokens",
    "context_tokens",
    "temperature",
    "fallbacks",
    "price_per_mtok",
}


class CatalogError(ValueError):
    pass


class UnknownModelError(LookupError):
    def __init__(self, name: str, known: list[str]):
        super().__init__(f"unknown model {name!r}")
        self.name = name
        self.known = known


class Price(BaseModel):
    input: float
    output: float


class ModelSpec(BaseModel):
    name: str
    provider: str
    model: str
    description: str = ""
    max_tokens: int = Field(default=1024, gt=0)
    context_tokens: int = Field(default=32_000, gt=0)
    temperature: float | None = None
    fallbacks: list[str] = Field(default_factory=list)
    price_per_mtok: Price | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_options(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            opts = dict(data.pop("options", {}) or {})
            for key in [k for k in data if k not in _SPEC_FIELDS | {"name"}]:
                opts[key] = data.pop(key)
            data["options"] = {k: v for k, v in opts.items() if v is not None}
        return data

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if not self.price_per_mtok:
            return None
        p = self.price_per_mtok
        return round((input_tokens * p.input + output_tokens * p.output) / 1_000_000, 6)


def _interpolate(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = _ENV.sub(lambda m: env.get(m.group(1)) or (m.group(2) or ""), value)
        return out if out != "" else None
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    return value


class ModelRegistry:
    def __init__(
        self,
        models: dict[str, ModelSpec],
        aliases: dict[str, str],
        default: str,
        allow_adhoc: bool = True,
        sources: dict[tuple[str, ...], str] | None = None,
    ):
        self.models = models
        self.aliases = aliases
        self.allow_adhoc = allow_adhoc
        self.sources = sources or {}  # catalog path -> "ssm:/..." it was read from
        self._validate(default)
        self.default = default

    # ---- loading ---------------------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        default_override: str | None = None,
        allow_adhoc: bool = True,
        env: dict[str, str] | None = None,
        resolver_factory: Callable[[], ParameterResolver] | None = None,
    ) -> ModelRegistry:
        try:
            raw = yaml.safe_load(Path(path).read_text()) or {}
        except FileNotFoundError as exc:
            raise CatalogError(f"model catalog not found: {path}") from exc
        return cls.from_dict(
            raw,
            default_override=default_override,
            allow_adhoc=allow_adhoc,
            env=env,
            resolver_factory=resolver_factory,
        )

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        default_override: str | None = None,
        allow_adhoc: bool = True,
        env: dict[str, str] | None = None,
        resolver_factory: Callable[[], ParameterResolver] | None = None,
    ) -> ModelRegistry:
        data = _interpolate(raw, dict(os.environ) if env is None else env)
        if default_override:
            data["default"] = default_override  # may itself be an ssm: reference
        try:
            data, sources = resolve_references(data, resolver_factory or SsmResolver)
        except ParameterError as exc:
            raise CatalogError(f"Parameter Store: {exc}") from exc
        models: dict[str, ModelSpec] = {}
        for name, body in (data.get("models") or {}).items():
            if not isinstance(body, dict) or "provider" not in body or "model" not in body:
                raise CatalogError(f"model {name!r} needs at least `provider` and `model`")
            if body["provider"] not in PROVIDERS:
                raise CatalogError(
                    f"model {name!r}: unknown provider {body['provider']!r} "
                    f"(known: {', '.join(sorted(PROVIDERS))})"
                )
            models[name] = ModelSpec(name=name, **body)
        aliases = {k: v for k, v in (data.get("aliases") or {}).items() if v}
        default = data.get("default") or next(iter(models), "")
        return cls(models, aliases, default, allow_adhoc, sources)

    def _validate(self, default: str) -> None:
        if not self.models:
            raise CatalogError("model catalog has no models")
        for alias, target in self.aliases.items():
            if alias in self.models:
                raise CatalogError(f"alias {alias!r} shadows a model of the same name")
            if target not in self.models:
                raise CatalogError(f"alias {alias!r} points at unknown model {target!r}")
        for spec in self.models.values():
            for fb in spec.fallbacks:
                if self._lookup(fb) is None:
                    raise CatalogError(f"model {spec.name!r}: unknown fallback {fb!r}")
        try:
            self.resolve(default)
        except UnknownModelError as exc:
            raise CatalogError(f"default model {default!r} is not in the catalog") from exc

    # ---- resolution -----------------------------------------------------------------------------
    def _lookup(self, name: str) -> ModelSpec | None:
        return self.models.get(self.aliases.get(name, name))

    def resolve(self, name: str | None) -> ModelSpec:
        name = (name or self.default).strip()
        if spec := self._lookup(name):
            return spec
        provider, sep, model_id = name.partition(":")
        if self.allow_adhoc and sep and provider in PROVIDERS and model_id:
            opts: dict[str, Any] = {}
            if provider == "bedrock" and os.environ.get("AWS_REGION"):
                opts["region"] = os.environ["AWS_REGION"]
            return ModelSpec(name=name, provider=provider, model=model_id, description="ad-hoc", options=opts)
        raise UnknownModelError(name, self.names())

    def fallback_chain(self, spec: ModelSpec) -> list[ModelSpec]:
        chain, seen = [spec], {spec.name}
        for fb in spec.fallbacks:
            target = self.resolve(fb)
            if target.name not in seen:
                chain.append(target)
                seen.add(target.name)
        return chain

    def names(self) -> list[str]:
        return sorted(self.aliases) + sorted(self.models)

    def source_of(self, model_name: str) -> str | None:
        """Where a catalog entry's model ID came from, if not written in the file (e.g. "ssm:/x/y")."""
        return self.sources.get(("models", model_name, "model"))

    def describe_changes(self, new: ModelRegistry) -> list[dict[str, Any]]:
        """What a reload would change: model IDs, providers, added/removed entries, alias targets, default."""
        changes: list[dict[str, Any]] = []
        for name in sorted(set(self.models) | set(new.models)):
            old_s, new_s = self.models.get(name), new.models.get(name)
            before = f"{old_s.provider}:{old_s.model}" if old_s else None
            after = f"{new_s.provider}:{new_s.model}" if new_s else None
            if before != after:
                changes.append({"model": name, "from": before, "to": after})
        for alias in sorted(set(self.aliases) | set(new.aliases)):
            if self.aliases.get(alias) != new.aliases.get(alias):
                changes.append(
                    {"alias": alias, "from": self.aliases.get(alias), "to": new.aliases.get(alias)}
                )
        if self.default != new.default:
            changes.append({"default": True, "from": self.default, "to": new.default})
        return changes

    def aliases_for(self, model_name: str) -> list[str]:
        return sorted(a for a, t in self.aliases.items() if t == model_name)
