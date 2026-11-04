"""Amazon Nova fine-tuning JSONL: record shape, Nova's rules, filters, cleaning, split, API and CLI."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conversation_api import cli
from conversation_api.schemas import Conversation, ModelUsed, StoredMessage
from conversation_api.training import (
    SCHEMA_VERSION,
    ExportOptions,
    build_dataset,
    nova_record,
    redact,
    validate_jsonl,
    validate_record,
)


def conv(
    *turns: str,
    model: str = "claude-sonnet",
    model_id: str = "us.anthropic.claude-sonnet-4-6",
    system: str | None = None,
    days_ago: int = 0,
    **metadata: Any,
) -> Conversation:
    msgs = []
    for i, text in enumerate(turns):
        if i % 2 == 0:
            msgs.append(StoredMessage(role="user", content=text))
        else:
            msgs.append(
                StoredMessage(
                    role="assistant",
                    content=text,
                    model=ModelUsed(name=model, provider="bedrock", model_id=model_id),
                )
            )
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return Conversation(
        model=model, system=system, messages=msgs, metadata=metadata, created_at=ts, updated_at=ts
    )


def many(n: int, **kw: Any) -> list[Conversation]:
    return [conv(f"question {i}", f"answer {i}", **kw) for i in range(n)]


# ---- record shape and rules ----------------------------------------------------------------------
def test_record_matches_nova_schema() -> None:
    data = build_dataset(
        [conv("hi", "hello", "how are you?", "fine", system="be brief")], ExportOptions(validation_split=0)
    )
    rec = data.train[0]
    assert rec == {
        "schemaVersion": SCHEMA_VERSION,
        "system": [{"text": "be brief"}],
        "messages": [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]},
            {"role": "user", "content": [{"text": "how are you?"}]},
            {"role": "assistant", "content": [{"text": "fine"}]},
        ],
    }
    assert "system" not in nova_record(None, [("user", "a"), ("assistant", "b")])


@pytest.mark.parametrize(
    "rec, problem",
    [
        (
            {
                "messages": [
                    {"role": "user", "content": [{"text": "a"}]},
                    {"role": "assistant", "content": [{"text": "b"}]},
                ]
            },
            "missing schemaVersion",
        ),
        (
            nova_record(None, [("assistant", "a"), ("user", "b"), ("assistant", "c")]),
            "first message must be from user",
        ),
        (nova_record(None, [("user", "a")]), "last message must be from assistant"),
        (nova_record(None, [("user", "a"), ("user", "b"), ("assistant", "c")]), "roles must alternate"),
        (nova_record(None, [("user", " "), ("assistant", "b")]), "non-empty text"),
        (nova_record("System: rules", [("user", "a"), ("assistant", "b")]), "reserved keyword"),
        (nova_record(None, [("user", "see <image>"), ("assistant", "b")]), "reserved keyword"),
        (nova_record(None, [("user", "x" * 40_000), ("assistant", "b")]), "longer than 32000"),
    ],
)
def test_validation_catches_what_nova_rejects(rec: dict[str, Any], problem: str) -> None:
    assert any(problem in e for e in validate_record(rec))


def test_reserved_keywords_in_assistant_text_are_allowed() -> None:
    assert validate_record(nova_record(None, [("user", "hi"), ("assistant", "User: is a label")])) == []


# ---- export behaviour ----------------------------------------------------------------------------
def test_turns_mode_makes_one_sample_per_reply() -> None:
    data = build_dataset(
        [conv("q1", "a1", "q2", "a2", "q3", "a3")], ExportOptions(mode="turns", validation_split=0)
    )
    assert [len(r["messages"]) for r in data.train] == [2, 4, 6]


def test_filters_model_metadata_since_min_turns_and_opt_out() -> None:
    convs = [
        conv("a", "b", dataset="support"),
        conv("c", "d", dataset="sales"),
        conv("e", "f", dataset="support", model="mock", model_id="echo-v1"),
        conv("g", "h", dataset="support", days_ago=30),
        conv("i", "j", "k", "l", dataset="support"),
        conv("m", "n", dataset="support", training=False),
    ]
    opts = ExportOptions(
        where={"dataset": "support"},
        models=["us.anthropic.claude-sonnet-4-6"],
        since=datetime.now(timezone.utc) - timedelta(days=7),
        validation_split=0,
    )
    data = build_dataset(convs, opts)
    assert [r["messages"][0]["content"][0]["text"] for r in data.train] == ["a", "i"]
    assert data.report.dropped["metadata dataset != support"] == 1
    assert data.report.dropped["reply from a model not in --model"] == 1
    assert data.report.dropped["older than --since"] == 1
    assert data.report.dropped["opted out (metadata training=false)"] == 1
    assert len(build_dataset(convs, opts.model_copy(update={"min_turns": 2})).train) == 1


def test_reserved_keywords_rewritten_or_dropped() -> None:
    c = [conv("User: where is my order?", "On its way."), *many(8)]
    fixed = build_dataset(c, ExportOptions(validation_split=0))
    texts = [r["messages"][0]["content"][0]["text"] for r in fixed.train]
    assert "User - where is my order?" in texts and fixed.report.reserved_rewritten == 1
    dropped = build_dataset(c, ExportOptions(validation_split=0, fix_reserved=False))
    assert dropped.report.dropped == {"reserved keyword in prompt": 1} and len(dropped.train) == 8


def test_redaction() -> None:
    text, n = redact(
        "mail me at a.b@example.com, key AKIAABCDEFGHIJKLMNOP, card 4111 1111 1111 1111, order 1234567890123"
    )
    assert "[email]" in text and "[aws-access-key]" in text and "[card-number]" in text
    assert "1234567890123" in text and n == 3  # not Luhn-valid: left alone
    data = build_dataset(
        [conv("my email is x@y.io", "noted x@y.io"), *many(8)], ExportOptions(validation_split=0)
    )
    assert data.report.redactions == 2 and "x@y.io" not in json.dumps(data.train)
    raw = build_dataset(
        [conv("my email is x@y.io", "ok"), *many(8)], ExportOptions(validation_split=0, redact=False)
    )
    assert "x@y.io" in json.dumps(raw.train)


def test_duplicates_are_removed() -> None:
    data = build_dataset(
        [conv("same", "same"), conv("same", "same"), *many(8)], ExportOptions(validation_split=0)
    )
    assert data.report.dropped == {"duplicate": 1} and data.report.samples == 9


def test_sample_limits_and_split() -> None:
    small = build_dataset(many(5), ExportOptions())
    assert not small.report.ready and "at least 8" in small.report.errors[0]
    assert small.validation == []  # never splits below the training minimum

    data = build_dataset(many(50), ExportOptions(validation_split=0.2, seed=7))
    assert (data.report.train, data.report.validation) == (40, 10) and data.report.ready
    assert "recommends at least 100" in data.report.warnings[0]
    train_keys = {json.dumps(r) for r in data.train}
    assert not train_keys & {json.dumps(r) for r in data.validation}
    again = build_dataset(many(50), ExportOptions(validation_split=0.2, seed=7))
    assert again.validation == data.validation  # deterministic for a seed

    capped = build_dataset(many(30), ExportOptions(validation_split=0, max_samples=20))
    assert capped.report.train == 20 and capped.report.dropped["over the 20,000-sample limit"] == 10


def test_validate_jsonl_file_text() -> None:
    good = "".join(
        json.dumps(nova_record(None, [("user", f"q{i}"), ("assistant", "a")])) + "\n" for i in range(8)
    )
    assert validate_jsonl(good) == (8, [])
    valid, problems = validate_jsonl(good + "{bad\n")
    assert valid == 8 and problems[0].startswith("line 9: not valid JSON")
    assert validate_jsonl(good.splitlines()[0])[1] == ["1 valid samples; Nova fine-tuning needs at least 8"]
    assert validate_jsonl(good.splitlines()[0], check_counts=False) == (1, [])


# ---- API + CLI ---------------------------------------------------------------------------------------
@pytest.fixture
def seeded(client: TestClient) -> TestClient:
    for i in range(12):
        c = client.post("/v1/conversations", json={"model": "mock", "metadata": {"dataset": "faq"}}).json()
        client.post(f"/v1/conversations/{c['id']}/messages", json={"content": f"question number {i}"})
    client.post("/v1/chat", json={"content": "not in the dataset", "model": "mock"})
    return client


def test_api_report_and_jsonl_export(seeded: TestClient) -> None:
    report = seeded.get(
        "/v1/training/report", params={"where": "dataset=faq", "validation_split": 0.25}
    ).json()
    assert report["ready"] and (report["train"], report["validation"]) == (9, 3)
    r = seeded.get("/v1/training/export.jsonl", params={"where": "dataset=faq", "validation_split": 0.25})
    assert r.headers["content-type"].startswith("application/jsonl") and r.headers["x-ready"] == "true"
    train = [json.loads(line) for line in r.text.splitlines()]
    val = seeded.get(
        "/v1/training/export.jsonl",
        params={"where": "dataset=faq", "validation_split": 0.25, "split": "validation"},
    ).text
    assert len(train) == 9 and len(val.splitlines()) == 3
    assert all(validate_record(rec) == [] for rec in train)
    assert seeded.get("/v1/training/report", params={"where": "oops"}).status_code == 422


def _cli(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    real_init = cli.Client.__init__

    def init(self: cli.Client, *a: Any, **kw: Any) -> None:
        real_init(self, *a, **kw)
        self.http = client  # TestClient is an httpx.Client: the CLI talks to the in-process app

    monkeypatch.setattr(cli.Client, "__init__", init)


def test_cli_collect_then_export_then_validate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli(client, monkeypatch)
    prompts = tmp_path / "support.txt"
    lines = [
        "# comment",
        *[f"How do I do thing {i}?" for i in range(9)],
        json.dumps({"prompt": "Refund please", "followups": ["Order 42"], "system": "Be concise."}),
    ]
    prompts.write_text("\n".join(lines) + "\n")
    cli.main(["collect", str(prompts), "-m", "mock"])
    assert "collected 10/10 conversations into dataset 'support'" in capsys.readouterr().out

    out = tmp_path / "out"
    cli.main(["export", "--where", "dataset=support", "--out", str(out), "--validation-split", "0.2"])
    printed = capsys.readouterr().out
    assert "samples: 10 (train 8, validation 2)" in printed
    refund = next(
        json.loads(x)
        for x in (out / "train.jsonl").read_text().splitlines()
        + (out / "validation.jsonl").read_text().splitlines()
        if "Refund" in x
    )
    assert refund["system"] == [{"text": "Be concise."}] and len(refund["messages"]) == 4

    cli.main(["validate-jsonl", str(out / "train.jsonl"), "--validation", str(out / "validation.jsonl")])
    assert "8 valid samples · ok" in capsys.readouterr().out


def test_cli_export_refuses_too_small_dataset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _cli(client, monkeypatch)
    client.post("/v1/chat", json={"content": "only one", "model": "mock"})
    with pytest.raises(SystemExit):
        cli.main(["export", "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x" / "train.jsonl").exists()


# ---- reinforcement fine-tuning conversion -----------------------------------------------------------
def test_sft_to_rft_makes_prompt_only_records_with_reference_answers() -> None:
    from conversation_api.training import sft_to_rft, validate_rft_record

    rec = nova_record("Be brief.", [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")])
    out = sft_to_rft(rec, "s-1")
    assert [r["id"] for r in out] == ["s-1-t1", "s-1-t2"]
    assert out[0] == {
        "id": "s-1-t1",
        "reference_answer": "a1",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q1"}],
    }
    assert [m["role"] for m in out[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert out[1]["reference_answer"] == "a2"
    assert all(validate_rft_record(r) == [] for r in out)
    single = sft_to_rft(nova_record(None, [("user", "q"), ("assistant", "a")]), "x")
    assert single == [{"id": "x", "messages": [{"role": "user", "content": "q"}], "reference_answer": "a"}]


def test_validate_rft_record_rejects_sft_shapes() -> None:
    from conversation_api.training import validate_rft_record

    sft = nova_record(None, [("user", "q"), ("assistant", "a")])
    assert any("content must be a non-empty string" in e for e in validate_rft_record(sft))
    assert "last message must be the user prompt (no assistant answer)" in validate_rft_record(
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    )


def test_cli_to_rft_and_shipped_rft_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from conversation_api.training import RFT_MIN_SAMPLES, validate_rft_record

    src = Path(__file__).resolve().parents[1] / "examples" / "datasets" / "ai-ml-tutor"
    cli.main(["to-rft", str(src / "train.jsonl"), str(tmp_path / "rft.jsonl"), "--id-prefix", "tutor-train"])
    assert "120 RFT records" in capsys.readouterr().out
    assert (tmp_path / "rft.jsonl").read_text() == (src / "rft" / "train.jsonl").read_text()
    shipped = [json.loads(x) for x in (src / "rft" / "train.jsonl").read_text().splitlines()]
    assert len(shipped) >= RFT_MIN_SAMPLES and all(validate_rft_record(r) == [] for r in shipped)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"messages": [{"role": "assistant", "content": [{"text": "a"}]}]}) + "\n")
    with pytest.raises(SystemExit):
        cli.main(["to-rft", str(bad), str(tmp_path / "out.jsonl")])
