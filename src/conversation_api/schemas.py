"""API and storage models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return "conv_" + uuid.uuid4().hex[:20]


# ---- stored conversation -------------------------------------------------------------------------
class UsageOut(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class ModelUsed(BaseModel):
    name: str  # catalog name actually used
    provider: str
    model_id: str


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    model: ModelUsed | None = None  # assistant turns record which model wrote them
    usage: UsageOut | None = None


class Conversation(BaseModel):
    id: str = Field(default_factory=new_id)
    model: str  # catalog name this conversation uses by default
    system: str | None = None
    messages: list[StoredMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_history: list[dict[str, Any]] = Field(default_factory=list)  # swaps: {at, from, to}
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def summary(self) -> ConversationSummary:
        first_user = next((m.content for m in self.messages if m.role == "user"), "")
        return ConversationSummary(
            id=self.id,
            model=self.model,
            turns=sum(1 for m in self.messages if m.role == "user"),
            preview=first_user[:80],
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class ConversationSummary(BaseModel):
    id: str
    model: str
    turns: int
    preview: str
    created_at: datetime
    updated_at: datetime


# ---- requests --------------------------------------------------------------------------------------
class CreateConversation(BaseModel):
    model: str | None = Field(
        default=None, description="Catalog name, alias, or provider:model-id. Default: server default."
    )
    system: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateConversation(BaseModel):
    model: str | None = Field(
        default=None, description="Swap the model for all later turns. History is kept."
    )
    system: str | None = None


class SendMessage(BaseModel):
    content: str = Field(min_length=1)
    model: str | None = Field(default=None, description="Use this model for this turn only.")
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0, le=64_000)
    temperature: float | None = Field(default=None, ge=0, le=2)


class ChatRequest(SendMessage):
    """Convenience: one call that creates the conversation on the first turn and continues it after."""

    conversation_id: str | None = None
    system: str | None = None


# ---- responses ------------------------------------------------------------------------------------
class Attempt(BaseModel):
    model: str
    provider: str
    error: str
    kind: str


class TurnResponse(BaseModel):
    conversation_id: str
    message: StoredMessage
    model_requested: str
    model_used: ModelUsed
    fell_back: bool
    attempts: list[Attempt] = Field(default_factory=list)  # failed attempts before the one that answered
    usage: UsageOut
    latency_ms: int
    history_trimmed: int = 0  # oldest messages left out to fit the model's context budget


class ModelInfo(BaseModel):
    name: str
    provider: str
    model_id: str
    model_source: str | None = (
        None  # e.g. "ssm:/conversation-api/models/sonnet" when read from Parameter Store
    )
    description: str
    aliases: list[str]
    is_default: bool
    available: bool
    status: str
    fallbacks: list[str]
    context_tokens: int
    max_tokens: int


class ReloadResult(BaseModel):
    models: int
    default_model: str
    changes: list[dict[str, Any]]
    parameters: list[str]  # Parameter Store references read during the reload
