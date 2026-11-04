# ADR-0003: Fallback driven by error class

**Status:** accepted · **Date:** 2026-11-04

## Context

Model calls fail for different reasons, and only some are fixed by trying a different model. Retrying
a malformed request on three providers triples cost and latency and still fails. Streaming adds a second
problem: once tokens have reached the client, switching models mid-answer would splice two answers.

## Decision

Each adapter maps its SDK's errors to one of three kinds:

| Kind | Examples | Action |
|---|---|---|
| `retryable` | 429, 5xx, 529 overloaded, Bedrock `ThrottlingException`, timeouts, connection errors | try the next model in `fallbacks` |
| `unavailable` | 401/403/404, missing SDK, no credentials, no model access | try the next model (another provider may be configured) |
| `invalid_request` | other 4xx, Bedrock `ValidationException` | stop, return 400 |

- SDK-level retries (botocore **adaptive** mode for Bedrock, SDK `max_retries` elsewhere) handle brief blips
  **within** a model. Bedrock clients are shared per region so adaptive mode's client-side rate limiter
  sees the whole process's traffic.
- A Bedrock `ThrottlingException` that survives those retries is caught in the adapter
  (`except BedrockThrottled`) and the request is sent once to the entry's `throttle_fallback_model`, a
  smaller model with separate quota, in the same provider and region. This is faster and cheaper than
  leaving the provider, and the turn reports the model ID that answered.
- The catalog `fallbacks` list handles a model or provider being **out**. An `invalid_request` from the
  *requested* model stops the turn; one from a *fallback* model is a configuration problem, so it is
  recorded and the next fallback is tried (the turn then ends as 502 `all_models_failed`, not 400).
- Streaming falls back **only before the first token**. After that, a failure ends the stream with
  `event: error` (`partial: true`) and the turn is not saved.
- Every response reports `model_requested`, `model_used`, `fell_back` and the failed `attempts`, and each
  attempt is logged, so fallbacks are visible rather than silent.

## Consequences

- A throttled primary degrades to a cheaper or different-vendor model with no client change.
- A fallback model may answer differently from the primary; callers that need a specific model can
  name a catalog entry that has no `fallbacks`.
