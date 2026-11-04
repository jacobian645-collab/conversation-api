# ADR-0001: Provider contract and a model catalog

**Status:** accepted · **Date:** 2026-11-04

## Context

Callers need to change models often: a cheaper model for simple turns, a vendor migration, an outage,
an A/B test. If clients hard-code a vendor SDK or model ID, each change is a client release. Models from
different vendors also disagree on message shape (system prompt placement, role names, content blocks,
token parameters).

## Decision

1. **One provider contract**: `complete(GenerationRequest) -> Completion` and
   `stream(GenerationRequest) -> TextDelta… StreamEnd`. The request is provider-neutral
   (`model`, `messages[{role, content}]`, `system`, `max_tokens`, `temperature`).
2. **Adapters per API shape**: Bedrock Converse (all Bedrock families), Anthropic Messages,
   OpenAI Chat Completions (OpenAI, Azure OpenAI, any OpenAI-compatible server), Gemini. Each owns the
   translation and the classification of its own errors.
3. **A catalog** (`config/models.yaml`) maps names clients use to `provider + model ID + limits +
   fallbacks + prices`. **Aliases** (`smart`, `fast`) add one more level of indirection, so re-pointing an
   alias moves every caller.
4. Conversations store the **name they asked for** (for example `smart`), resolved on each turn. Re-pointing
   the alias therefore also moves existing conversations.
5. SDKs are **optional extras, imported lazily**. A missing SDK or credential disables only the models that
   need it, and `/v1/models` reports why.

## Consequences

- Swapping is a config change; adding a provider is one class.
- The lowest common denominator is plain text turns. Features that differ by vendor (tools, images,
  citations, prompt caching) need typed content blocks in the contract before they can be exposed.
- Model IDs in the catalog drift as vendors release models; `${VAR:-default}` lets each environment
  override them without editing the file.
