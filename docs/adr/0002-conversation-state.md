# ADR-0002: Conversation state and atomic turns

**Status:** accepted · **Date:** 2026-11-04

## Context

The server keeps history so shell clients only send the new message. Bedrock and Anthropic reject
histories that do not alternate user/assistant or that start with an assistant turn. Two concurrent
requests on one conversation could interleave, and a failed model call could leave a dangling user message.

## Decision

- A turn is **committed atomically**: user message + reply are appended and saved together, only after a
  model returns a complete answer. Failed, rejected and abandoned (client disconnect, mid-stream error)
  turns are not saved.
- Turns on the same conversation are **serialised** with a per-conversation `asyncio.Lock`.
- Storage sits behind a four-method `ConversationStore` protocol (`get`, `put`, `delete`, `list`).
  `memory` (LRU, default) and `sqlite` (WAL) ship. Each assistant message records the model that wrote it;
  swaps are appended to `model_history`.
- What is **sent** to a model is fitted to that model's budget (`context_tokens - max_tokens`), oldest
  first. What is **stored** is never trimmed.

## Consequences

- History is always valid for every provider, whatever happened on earlier turns.
- The lock is in-process. With several replicas, two requests on one conversation can race. The
  production path is a shared store with an optimistic version (DynamoDB conditional write, Cosmos DB
  ETag, Firestore transaction): read version *n*, write only if still *n*, otherwise retry the turn or
  return 409.
- SQLite suits a single instance with a volume (a cloud shell, a VM). It is not a multi-writer store.
