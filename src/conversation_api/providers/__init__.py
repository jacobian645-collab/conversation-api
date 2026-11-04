"""Provider registry: provider name -> adapter class. Adding a backend is one class and one line here."""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import (
    ChatMessage,
    Completion,
    ErrorKind,
    GenerationRequest,
    ModelSwitch,
    Provider,
    ProviderError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)
from .bedrock import BedrockProvider
from .gemini import GeminiProvider
from .mock import MockProvider
from .openai_compat import AzureOpenAIProvider, OllamaProvider, OpenAIProvider

PROVIDERS: dict[str, type[Provider]] = {
    cls.name: cls
    for cls in (
        MockProvider,
        BedrockProvider,
        AnthropicProvider,
        OpenAIProvider,
        AzureOpenAIProvider,
        GeminiProvider,
        OllamaProvider,
    )
}

__all__ = [
    "PROVIDERS",
    "ChatMessage",
    "Completion",
    "ErrorKind",
    "GenerationRequest",
    "ModelSwitch",
    "Provider",
    "ProviderError",
    "StreamEnd",
    "StreamEvent",
    "TextDelta",
    "Usage",
]
