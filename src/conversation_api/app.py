"""HTTP API (FastAPI). Run with `convo serve` or `python -m conversation_api`."""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import Settings, get_settings
from .registry import ModelRegistry, UnknownModelError
from .schemas import (
    ChatRequest,
    Conversation,
    ConversationSummary,
    CreateConversation,
    ModelInfo,
    SendMessage,
    TurnResponse,
    UpdateConversation,
)
from .service import ConversationNotFound, ConversationService, TurnFailed
from .store import ConversationStore, MemoryStore, SqliteStore

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


def create_app(settings: Settings | None = None, service: ConversationService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if app.state.service is None:
            registry = ModelRegistry.from_file(
                settings.models_file,
                default_override=settings.default_model,
                allow_adhoc=settings.allow_adhoc_models,
            )
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
            },
        )
        if not settings.api_key and settings.host not in ("127.0.0.1", "localhost"):
            log.warning(
                "server.no_api_key",
                extra={"host": settings.host, "hint": "set CONVO_API_KEY before exposing this port"},
            )
        yield

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
