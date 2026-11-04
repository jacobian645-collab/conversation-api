"""HTTP API (FastAPI). Run with `convo serve` or `python -m conversation_api`."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .config import Settings, get_settings
from .registry import CatalogError, ModelRegistry, UnknownModelError
from .schemas import (
    ChatRequest,
    Conversation,
    ConversationSummary,
    CreateConversation,
    ModelInfo,
    ReloadResult,
    SendMessage,
    TurnResponse,
    UpdateConversation,
)
from .service import ConversationNotFound, ConversationService, TurnFailed
from .store import ConversationStore, MemoryStore, SqliteStore
from .training import MAX_CONTENT_CHARS, Dataset, ExportOptions, build_dataset, to_jsonl

log = logging.getLogger("conversation_api")


# ---- logging -------------------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    _std = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload |= {k: v for k, v in record.__dict__.items() if k not in self._std}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("conversation_api")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


# ---- app factory ---------------------------------------------------------------------------------
def build_store(settings: Settings) -> ConversationStore:
    if settings.store == "sqlite":
        return SqliteStore(settings.sqlite_path)
    return MemoryStore(settings.max_conversations)


def export_options(
    mode: Literal["conversation", "turns"] = Query("conversation"),
    model: list[str] = Query(default=[], description="catalog name or model ID; repeat or comma-separate"),
    min_turns: int = Query(1, ge=1),
    since: datetime | None = Query(None),
    where: list[str] = Query(default=[], description="metadata filter key=value; repeatable"),
    system: str | None = Query(None, description="system prompt for every sample"),
    validation_split: float = Query(0.1, ge=0, le=0.5),
    seed: int = Query(42),
    redact: bool = Query(True),
    fix_reserved: bool = Query(True),
    max_chars: int = Query(MAX_CONTENT_CHARS, gt=0),
) -> ExportOptions:
    filters: dict[str, str] = {}
    for item in where:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise HTTPException(
                422, {"code": "bad_filter", "message": f"where must be key=value, got {item!r}"}
            )
        filters[key] = value
    try:
        return ExportOptions(
            mode=mode,
            models=[m for item in model for m in item.split(",") if m],
            min_turns=min_turns,
            since=since,
            where=filters,
            system=system,
            validation_split=validation_split,
            seed=seed,
            redact=redact,
            fix_reserved=fix_reserved,
            max_chars=max_chars,
        )
    except ValidationError as exc:
        raise HTTPException(422, {"code": "bad_export_options", "message": str(exc)}) from exc


def load_registry(settings: Settings) -> ModelRegistry:
    """Read the catalog: file, then ${ENV} interpolation, then ssm: Parameter Store references."""
    return ModelRegistry.from_file(
        settings.models_file,
        default_override=settings.default_model,
        allow_adhoc=settings.allow_adhoc_models,
    )


def create_app(
    settings: Settings | None = None,
    service: ConversationService | None = None,
    registry_loader: Callable[[], ModelRegistry] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    loader: Callable[[], ModelRegistry] = registry_loader or (lambda: load_registry(settings))

    async def reload_catalog() -> ReloadResult:
        new = await asyncio.to_thread(loader)  # Parameter Store calls are blocking
        changes = svc().reload(new)
        return ReloadResult(
            models=len(new.models),
            default_model=new.default,
            changes=changes,
            parameters=sorted(set(new.sources.values())),
        )

    async def refresh_forever(every_s: int) -> None:
        while True:
            await asyncio.sleep(every_s)
            try:
                result = await reload_catalog()
                if result.changes:
                    log.info("catalog.refresh_changed", extra={"changes": result.changes})
            except Exception as exc:  # keep serving on the last good catalog
                log.warning("catalog.refresh_failed", extra={"error": str(exc)})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if app.state.service is None:
            registry = loader()
            app.state.service = ConversationService(
                registry, build_store(settings), timeout_s=settings.request_timeout_s
            )
        svc: ConversationService = app.state.service
        log.info(
            "server.started",
            extra={
                "version": __version__,
                "default_model": svc.registry.default,
                "store": svc.store.kind,
                "auth": bool(settings.api_key),
                "models": len(svc.registry.models),
                "parameters": sorted(set(svc.registry.sources.values())),
                "catalog_refresh_s": settings.catalog_refresh_s,
            },
        )
        if not settings.api_key and settings.host not in ("127.0.0.1", "localhost"):
            log.warning(
                "server.no_api_key",
                extra={"host": settings.host, "hint": "set CONVO_API_KEY before exposing this port"},
            )
        refresher = (
            asyncio.create_task(refresh_forever(settings.catalog_refresh_s))
            if settings.catalog_refresh_s
            else None
        )
        yield
        if refresher:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher

    app = FastAPI(
        title="Conversation API",
        version=__version__,
        description="Multi-turn conversations with hot-swappable models. "
        "Swap per conversation (PATCH), per turn (`model` on a message), or for every caller "
        "at once (re-point an alias in config/models.yaml).",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.settings = settings

    def svc() -> ConversationService:
        return app.state.service  # type: ignore[no-any-return]

    def require_key(
        x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)
    ) -> None:
        expected = settings.api_key
        if not expected:
            return
        supplied = x_api_key
        if supplied is None and authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(
                401,
                detail={"code": "unauthorized", "message": "missing or wrong API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ---- middleware + errors -------------------------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.monotonic()
        response: Response = await call_next(request)
        response.headers["x-request-id"] = rid
        if request.url.path != "/healthz":
            log.info(
                "http.request",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": int((time.monotonic() - t0) * 1000),
                },
            )
        return response

    def error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
        return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status)

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_: Request, exc: UnknownModelError) -> JSONResponse:
        return error(400, "unknown_model", str(exc), available=exc.known)

    @app.exception_handler(ConversationNotFound)
    async def _not_found(_: Request, exc: ConversationNotFound) -> JSONResponse:
        return error(404, "conversation_not_found", f"no conversation {exc}")

    @app.exception_handler(TurnFailed)
    async def _turn_failed(_: Request, exc: TurnFailed) -> JSONResponse:
        status, code = (400, "invalid_request") if exc.invalid_request else (502, "all_models_failed")
        return error(status, code, str(exc), attempts=[a.model_dump() for a in exc.attempts])

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        detail = (
            exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        )
        return JSONResponse({"error": detail}, status_code=exc.status_code, headers=exc.headers)

    def check_sizes(content: str | None = None, system: str | None = None) -> None:
        if content and len(content) > settings.max_message_chars:
            raise HTTPException(
                413, {"code": "message_too_long", "message": f"max {settings.max_message_chars} chars"}
            )
        if system and len(system) > settings.max_system_chars:
            raise HTTPException(
                413, {"code": "system_too_long", "message": f"max {settings.max_system_chars} chars"}
            )

    def sse(events: AsyncIterator[tuple[str, dict[str, Any]]]) -> StreamingResponse:
        async def body() -> AsyncIterator[str]:
            try:
                async for name, data in events:
                    yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"
            except Exception as exc:  # never leave a client hanging on an unexpected error
                log.exception("stream.failed")
                yield f"event: error\ndata: {json.dumps({'error': str(exc), 'kind': 'internal'})}\n\n"

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- routes ----------------------------------------------------------------------------------
    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, Any]:
        s = svc()
        return {
            "status": "ok",
            "version": __version__,
            "default_model": s.registry.default,
            "store": s.store.kind,
        }

    v1 = [Depends(require_key)]

    @app.get("/v1/models", response_model=list[ModelInfo], dependencies=v1, tags=["models"])
    async def list_models() -> list[ModelInfo]:
        return svc().list_models()

    async def dataset(opts: ExportOptions) -> Dataset:
        return build_dataset(await svc().store.list(limit=1_000_000), opts)

    @app.get("/v1/training/report", dependencies=v1, tags=["training"])
    async def training_report(opts: ExportOptions = Depends(export_options)) -> dict[str, Any]:
        """What an export with these options would contain, and whether Nova would accept it."""
        report = (await dataset(opts)).report
        return report.model_dump() | {"ready": report.ready}

    @app.get(
        "/v1/training/export.jsonl",
        dependencies=v1,
        tags=["training"],
        responses={200: {"content": {"application/jsonl": {}}}},
    )
    async def training_export(
        split: Literal["train", "validation", "all"] = Query("train"),
        opts: ExportOptions = Depends(export_options),
    ) -> PlainTextResponse:
        """Amazon Nova fine-tuning JSONL (bedrock-conversation-2024). The split is deterministic for a given
        seed, so fetching train and validation separately gives disjoint sets."""
        data = await dataset(opts)
        records = {"train": data.train, "validation": data.validation, "all": data.train + data.validation}[
            split
        ]
        r = data.report
        return PlainTextResponse(
            to_jsonl(records),
            media_type="application/jsonl",
            headers={
                "X-Samples": str(len(records)),
                "X-Train": str(r.train),
                "X-Validation": str(r.validation),
                "X-Ready": str(r.ready).lower(),
            },
        )

    @app.post("/v1/admin/reload-models", response_model=ReloadResult, dependencies=v1, tags=["models"])
    async def reload_models() -> Any:
        """Re-read config/models.yaml and any ssm: Parameter Store values without restarting.
        On failure nothing changes and the previous catalog keeps serving."""
        try:
            return await reload_catalog()
        except CatalogError as exc:
            return error(503, "reload_failed", str(exc))

    @app.post(
        "/v1/conversations",
        status_code=201,
        response_model=Conversation,
        dependencies=v1,
        tags=["conversations"],
    )
    async def create_conversation(body: CreateConversation) -> Conversation:
        check_sizes(system=body.system)
        return await svc().create(body.model, body.system, body.metadata)

    @app.get(
        "/v1/conversations", response_model=list[ConversationSummary], dependencies=v1, tags=["conversations"]
    )
    async def list_conversations(limit: int = Query(20, ge=1, le=200)) -> list[ConversationSummary]:
        return [c.summary() for c in await svc().store.list(limit)]

    @app.get(
        "/v1/conversations/{conv_id}", response_model=Conversation, dependencies=v1, tags=["conversations"]
    )
    async def get_conversation(conv_id: str) -> Conversation:
        return await svc().get(conv_id)

    @app.patch(
        "/v1/conversations/{conv_id}", response_model=Conversation, dependencies=v1, tags=["conversations"]
    )
    async def update_conversation(conv_id: str, body: UpdateConversation) -> Conversation:
        check_sizes(system=body.system)
        return await svc().update(conv_id, body)

    @app.delete("/v1/conversations/{conv_id}", status_code=204, dependencies=v1, tags=["conversations"])
    async def delete_conversation(conv_id: str) -> Response:
        await svc().delete(conv_id)
        return Response(status_code=204)

    @app.post(
        "/v1/conversations/{conv_id}/messages",
        response_model=TurnResponse,
        dependencies=v1,
        tags=["conversations"],
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def send_message(conv_id: str, body: SendMessage) -> Any:
        check_sizes(content=body.content)
        if body.stream:
            await svc().get(conv_id)  # 404 before the stream starts, not inside it
            if body.model:
                svc().registry.resolve(body.model)
            return sse(svc().send_stream(conv_id, body))
        return await svc().send(conv_id, body)

    @app.post(
        "/v1/chat",
        response_model=TurnResponse,
        dependencies=v1,
        tags=["chat"],
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def chat(body: ChatRequest) -> Any:
        """One endpoint for shell use: omit `conversation_id` to start, pass it back to continue.
        `model` here is sticky (it swaps the conversation's model)."""
        check_sizes(content=body.content, system=body.system)
        conv_id, msg = await svc().chat(body)
        if msg.stream:
            return sse(svc().send_stream(conv_id, msg))
        return await svc().send(conv_id, msg)

    return app


def run(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "conversation_api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level="warning",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
