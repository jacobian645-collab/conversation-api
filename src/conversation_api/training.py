"""Fine-tuning datasets from stored conversations, in Amazon Nova's Bedrock format.

Each JSONL line (Nova Micro / Lite / Pro, text):

    {"schemaVersion": "bedrock-conversation-2024",
     "system": [{"text": "..."}],                      # optional
     "messages": [{"role": "user", "content": [{"text": "..."}]},
                  {"role": "assistant", "content": [{"text": "..."}]}, ...]}

Rules checked here (Amazon Nova user guide, "Prepare data for fine-tuning Understanding models"):
  * roles are only user/assistant; the first message is user, the last is assistant (and turns alternate)
  * 8 to 20,000 samples per dataset; at least 100 recommended
  * maximum content length 32,000 (the guide gives no unit; this checks characters, which is conservative)
  * reserved keywords "User:", "Bot:", "Assistant:", "System:", "<image>", "<video>", "[EOS]" in a user or
    system prompt make the training job fail, so they are rewritten (or the sample dropped)

The export also deduplicates, can redact obvious secrets/PII (emails, AWS access key IDs, card numbers),
and makes a deterministic train/validation split.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .schemas import Conversation, StoredMessage

SCHEMA_VERSION = "bedrock-conversation-2024"
MIN_SAMPLES, MAX_SAMPLES, RECOMMENDED_SAMPLES = 8, 20_000, 100
MAX_CONTENT_CHARS = 32_000

# Reserved keywords and the minimal rewrite that keeps the meaning but no longer matches.
RESERVED: dict[str, str] = {
    "User:": "User -",
    "Bot:": "Bot -",
    "Assistant:": "Assistant -",
    "System:": "System -",
    "<image>": "(image)",
    "<video>": "(video)",
    "[EOS]": "(EOS)",
}

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


class ExportOptions(BaseModel):
    mode: Literal["conversation", "turns"] = Field(
        default="conversation",
        description="conversation: one sample per conversation (all turns). "
        "turns: one sample per assistant reply, with the history before it.",
    )
    models: list[str] = Field(
        default_factory=list, description="only replies written by these catalog names or model IDs"
    )
    min_turns: int = Field(default=1, ge=1, description="minimum user turns per conversation")
    since: datetime | None = Field(
        default=None, description="only conversations updated at or after this time"
    )
    where: dict[str, str] = Field(
        default_factory=dict, description="conversation metadata must match, e.g. dataset=support"
    )
    system: str | None = Field(
        default=None, description="system prompt for every sample (default: the conversation's own)"
    )
    validation_split: float = Field(default=0.1, ge=0, le=0.5)
    seed: int = 42
    redact: bool = True
    fix_reserved: bool = Field(
        default=True, description="rewrite reserved keywords; false drops those samples"
    )
    max_chars: int = Field(default=MAX_CONTENT_CHARS, gt=0)
    max_samples: int = Field(default=MAX_SAMPLES, gt=0)


class ExportReport(BaseModel):
    format: str = "nova"
    schema_version: str = SCHEMA_VERSION
    conversations_scanned: int = 0
    conversations_used: int = 0
    samples: int = 0
    train: int = 0
    validation: int = 0
    dropped: dict[str, int] = Field(default_factory=dict)
    reserved_rewritten: int = 0
    redactions: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.errors


@dataclass
class Dataset:
    train: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    report: ExportReport = field(default_factory=ExportReport)


# ---- cleaning ---------------------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0


def redact(text: str) -> tuple[str, int]:
    """Mask emails, AWS access key IDs and Luhn-valid card numbers. Returns (text, replacements)."""
    count = 0

    def sub(pattern: re.Pattern[str], repl: str, s: str) -> str:
        nonlocal count
        s, n = pattern.subn(repl, s)
        count += n
        return s

    text = sub(_EMAIL, "[email]", text)
    text = sub(_AWS_KEY, "[aws-access-key]", text)

    def card(m: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            count += 1
            return "[card-number]"
        return m.group(0)

    return _CARD.sub(card, text), count


def reserved_in(text: str) -> list[str]:
    return [k for k in RESERVED if k in text]


def rewrite_reserved(text: str) -> str:
    for k, v in RESERVED.items():
        text = text.replace(k, v)
    return text


# ---- record building & validation -----------------------------------------------------------------
def nova_record(system: str | None, messages: Sequence[tuple[str, str]]) -> dict[str, Any]:
    rec: dict[str, Any] = {"schemaVersion": SCHEMA_VERSION}
    if system:
        rec["system"] = [{"text": system}]
    rec["messages"] = [{"role": role, "content": [{"text": text}]} for role, text in messages]
    return rec


def record_texts(rec: dict[str, Any]) -> list[tuple[str, str]]:
    """(role, text) for every text block, system included as role 'system'."""
    out = [("system", b.get("text", "")) for b in rec.get("system") or []]
    for m in rec.get("messages") or []:
        out += [(m.get("role", ""), b.get("text", "")) for b in m.get("content") or []]
    return out


def validate_record(rec: Any, max_chars: int = MAX_CONTENT_CHARS) -> list[str]:
    """Problems that would make Nova reject this sample (empty list = valid)."""
    if not isinstance(rec, dict):
        return ["not a JSON object"]
    errs: list[str] = []
    if not rec.get("schemaVersion"):
        errs.append("missing schemaVersion")
    system = rec.get("system", [])
    if not isinstance(system, list) or any(not isinstance(b, dict) or "text" not in b for b in system):
        errs.append("system must be a list of {text}")
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return [*errs, "messages must be a non-empty list"]
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            errs.append(f"message {i}: role must be user or assistant")
            continue
        content = m.get("content")
        if not isinstance(content, list) or not content:
            errs.append(f"message {i}: content must be a non-empty list")
        elif not all(
            isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"].strip() for b in content
        ):
            errs.append(f"message {i}: every content block needs non-empty text")
        if i > 0 and isinstance(msgs[i - 1], dict) and msgs[i - 1].get("role") == m.get("role"):
            errs.append(f"message {i}: roles must alternate user/assistant")
    if isinstance(msgs[0], dict) and msgs[0].get("role") != "user":
        errs.append("first message must be from user")
    if isinstance(msgs[-1], dict) and msgs[-1].get("role") != "assistant":
        errs.append("last message must be from assistant")
    texts = record_texts(rec) if not errs else []
    for role, text in texts:
        if role in ("user", "system") and (found := reserved_in(text)):
            errs.append(f"{role} text contains reserved keyword(s) {', '.join(found)}")
    if sum(len(t) for _, t in texts) > max_chars:
        errs.append(f"content longer than {max_chars} characters")
    return errs


# ---- export -------------------------------------------------------------------------------------------
def _reply_matches(msg: StoredMessage, models: set[str]) -> bool:
    return msg.model is not None and (msg.model.name in models or msg.model.model_id in models)


def _conversation_ok(conv: Conversation, opts: ExportOptions) -> str | None:
    """None if the conversation is eligible, else the reason it isn't."""
    if conv.metadata.get("training") in (False, "false", "no", "exclude"):
        return "opted out (metadata training=false)"
    for k, v in opts.where.items():
        if str(conv.metadata.get(k)) != v:
            return f"metadata {k} != {v}"
    if opts.since:
        since = opts.since if opts.since.tzinfo else opts.since.replace(tzinfo=timezone.utc)
        if conv.updated_at < since:
            return "older than --since"
    if sum(1 for m in conv.messages if m.role == "user") < opts.min_turns:
        return "fewer than min_turns"
    return None


def _samples(conv: Conversation, opts: ExportOptions) -> list[list[StoredMessage]]:
    msgs = conv.messages
    if opts.mode == "conversation":
        return [msgs] if msgs else []
    return [msgs[: i + 1] for i, m in enumerate(msgs) if m.role == "assistant"]


def build_dataset(conversations: list[Conversation], opts: ExportOptions) -> Dataset:
    report = ExportReport(conversations_scanned=len(conversations))
    dropped: Counter[str] = Counter()
    models = set(opts.models)
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    used: set[str] = set()

    for conv in sorted(conversations, key=lambda c: c.created_at):
        if reason := _conversation_ok(conv, opts):
            dropped[reason] += 1
            continue
        for window in _samples(conv, opts):
            if models and not all(_reply_matches(m, models) for m in window if m.role == "assistant"):
                dropped["reply from a model not in --model"] += 1
                continue
            system = opts.system if opts.system is not None else conv.system
            pairs: list[tuple[str, str]] = [(m.role, m.content) for m in window]
            if opts.redact:
                cleaned = []
                for role, text in pairs:
                    text, n = redact(text)
                    report.redactions += n
                    cleaned.append((role, text))
                pairs = cleaned
                if system:
                    system, n = redact(system)
                    report.redactions += n
            prompt_texts = [t for r, t in pairs if r == "user"] + ([system] if system else [])
            if any(reserved_in(t) for t in prompt_texts):
                if not opts.fix_reserved:
                    dropped["reserved keyword in prompt"] += 1
                    continue
                report.reserved_rewritten += 1
                pairs = [(r, rewrite_reserved(t) if r == "user" else t) for r, t in pairs]
                system = rewrite_reserved(system) if system else system
            rec = nova_record(system, pairs)
            if problems := validate_record(rec, opts.max_chars):
                dropped["invalid: " + problems[0].split(":")[0]] += 1
                continue
            key = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            samples.append(rec)
            used.add(conv.id)

    if len(samples) > opts.max_samples:
        report.warnings.append(f"{len(samples)} samples; kept the newest {opts.max_samples} (Nova maximum)")
        dropped["over the 20,000-sample limit"] += len(samples) - opts.max_samples
        samples = samples[-opts.max_samples :]

    # deterministic split: shuffle by seed, hold out a share for validation (only if train stays >= minimum)
    order = list(range(len(samples)))
    random.Random(opts.seed).shuffle(order)
    n_val = round(len(samples) * opts.validation_split)
    if len(samples) - n_val < MIN_SAMPLES:
        n_val = max(0, len(samples) - MIN_SAMPLES)
    val_idx = set(order[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    validation = [s for i, s in enumerate(samples) if i in val_idx]

    report.conversations_used = len(used)
    report.samples, report.train, report.validation = len(samples), len(train), len(validation)
    report.dropped = dict(dropped)
    if len(train) < MIN_SAMPLES:
        report.errors.append(
            f"{len(train)} training samples; Nova fine-tuning needs at least {MIN_SAMPLES}. "
            "Collect more conversations (convo collect) or relax the filters."
        )
    elif len(train) < RECOMMENDED_SAMPLES:
        report.warnings.append(
            f"{len(train)} training samples; AWS recommends at least {RECOMMENDED_SAMPLES}"
        )
    if opts.validation_split and not validation:
        report.warnings.append("too few samples for a validation split; exporting training data only")
    return Dataset(train=train, validation=validation, report=report)


def to_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)


def validate_jsonl(
    text: str, max_chars: int = MAX_CONTENT_CHARS, check_counts: bool = True
) -> tuple[int, list[str]]:
    """Validate a whole JSONL file's text. Returns (valid sample count, problems as 'line N: ...').
    The 8..20,000 sample limits apply to training data; pass check_counts=False for a validation file."""
    problems: list[str] = []
    valid = 0
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {n}: not valid JSON ({exc.msg})")
            continue
        errs = validate_record(rec, max_chars)
        problems += [f"line {n}: {e}" for e in errs]
        valid += not errs
    if not check_counts:
        return valid, problems
    if valid < MIN_SAMPLES:
        problems.append(f"{valid} valid samples; Nova fine-tuning needs at least {MIN_SAMPLES}")
    elif valid > MAX_SAMPLES:
        problems.append(f"{valid} samples; Nova accepts at most {MAX_SAMPLES}")
    return valid, problems
