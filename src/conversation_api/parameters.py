"""AWS Systems Manager Parameter Store references in the model catalog.

After ${ENV} interpolation, any catalog value written as

    ssm:/path/to/parameter
    ssm:arn:aws:ssm:<region>:<account>:parameter/path/to/parameter

is replaced by that parameter's value (SecureString values are decrypted). Because env interpolation runs
first, a reference can come from the environment too:

    export BEDROCK_SONNET_ID=ssm:/conversation-api/models/sonnet

IAM needed by whoever runs the server: ssm:GetParameter on the parameter, plus kms:Decrypt on the key if
it is a SecureString encrypted with a customer-managed key.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Protocol

PREFIX = "ssm:"
Path = tuple[str, ...]


class ParameterError(RuntimeError):
    pass


class ParameterResolver(Protocol):
    def get(self, name: str) -> str: ...


def is_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


class SsmResolver:
    """Reads parameters with boto3. Region: the ARN's region, else CONVO_SSM_REGION, else AWS_REGION."""

    def __init__(self, region: str | None = None, client_factory: Callable[[str | None], Any] | None = None):
        self.region = (
            region
            or os.environ.get("CONVO_SSM_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        self._factory = client_factory or self._make_client
        self._clients: dict[str | None, Any] = {}

    @staticmethod
    def _make_client(region: str | None) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ParameterError("reading Parameter Store needs boto3: pip install -e '.[bedrock]'") from exc
        return boto3.client("ssm", region_name=region)

    def _region_for(self, name: str) -> str | None:
        if name.startswith("arn:"):
            parts = name.split(":")
            if len(parts) > 3 and parts[3]:
                return parts[3]
        return self.region

    def get(self, name: str) -> str:
        region = self._region_for(name)
        if region not in self._clients:
            self._clients[region] = self._factory(region)
        try:
            resp = self._clients[region].get_parameter(Name=name, WithDecryption=True)
        except Exception as exc:
            raise ParameterError(self._explain(name, region, exc)) from exc
        value = str(resp["Parameter"]["Value"]).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()  # a quoted value ("us.anthropic...") is a common copy-paste slip
        return value

    @staticmethod
    def _explain(name: str, region: str | None, exc: Exception) -> str:
        where = f"{name!r} (region {region or 'unset'})"
        code = (
            getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        )
        if code == "ParameterNotFound":
            return (
                f"parameter {where} not found. Check the name and that AWS_REGION is the parameter's region"
            )
        if code in ("AccessDeniedException", "AccessDenied"):
            return f"not allowed to read parameter {where}: needs ssm:GetParameter (and kms:Decrypt for SecureString)"
        if type(exc).__name__ == "NoCredentialsError":
            return f"no AWS credentials to read parameter {where}"
        if type(exc).__name__ == "NoRegionError":
            return f"no AWS region set to read parameter {name!r}: export AWS_REGION"
        return f"could not read parameter {where}: {type(exc).__name__}: {exc}"


def resolve_references(
    data: Any, resolver_factory: Callable[[], ParameterResolver]
) -> tuple[Any, dict[Path, str]]:
    """Replace every `ssm:` string in `data`. Returns (resolved data, {path: reference}) for provenance.

    The resolver is only created if a reference exists, so catalogs without `ssm:` never touch AWS.
    Each parameter is fetched once per load.
    """
    refs: dict[Path, str] = {}
    cache: dict[str, str] = {}
    resolver: ParameterResolver | None = None

    def fetch(ref: str) -> str:
        nonlocal resolver
        if ref not in cache:
            if resolver is None:
                resolver = resolver_factory()
            value = resolver.get(ref[len(PREFIX) :])
            if not value:
                raise ParameterError(f"parameter {ref[len(PREFIX) :]!r} is empty")
            cache[ref] = value
        return cache[ref]

    def walk(value: Any, path: Path) -> Any:
        if is_reference(value):
            if not value[len(PREFIX) :].strip():
                where = ".".join(path) or "value"
                raise ParameterError(
                    f"{where} is 'ssm:' with no parameter name. If it came from an environment variable "
                    '(e.g. BEDROCK_SONNET_ID="ssm:$PARAM"), the variable it was built from was empty'
                )
            refs[path] = value
            return fetch(value)
        if isinstance(value, dict):
            return {k: walk(v, (*path, str(k))) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, (*path, str(i))) for i, v in enumerate(value)]
        return value

    return walk(data, ()), refs
