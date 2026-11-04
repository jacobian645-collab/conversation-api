"""`convo`: run the server, or talk to it from a terminal (AWS CloudShell, Azure Cloud Shell, Google Cloud Shell).

  convo serve                         start the API on 127.0.0.1:8080
  convo models                        what can I call, and is it configured?
  convo ask "question" -m smart       one-shot (prints the conversation id so you can continue)
  convo say <id> "follow-up"          continue a conversation
  convo swap <id> gpt                 change the model for later turns (history is kept)
  convo reload                        re-read models.yaml + Parameter Store values without a restart
  convo chat -m fast                  interactive; type /help inside

Environment: CONVO_URL (default http://127.0.0.1:8080), CONVO_API_KEY, and CONVO_BEARER_TOKEN for a
platform token in front of the app (for example `gcloud auth print-identity-token` for a private Cloud Run service).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterator
from typing import Any

import httpx

DIM, BOLD, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[31m", "\033[0m"


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else text


class ApiError(Exception):
    pass


class Client:
    def __init__(self, url: str, api_key: str | None, bearer: str | None, timeout: float = 300.0):
        headers = {"User-Agent": "convo-cli"}
        if api_key:
            headers["X-API-Key"] = api_key
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        self.http = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout)

    def _check(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                err = r.json().get("error", {})
                msg = f"{r.status_code} {err.get('code', '')}: {err.get('message', r.text)}"
                if err.get("available"):
                    msg += f"\n  available: {', '.join(err['available'])}"
                for a in err.get("attempts", []):
                    msg += f"\n  tried {a['model']} ({a['provider']}): {a['error']}"
            except (ValueError, AttributeError):
                msg = f"{r.status_code}: {r.text[:300]}"
            raise ApiError(msg)
        return r.json() if r.content else None

    def get(self, path: str, **kw: Any) -> Any:
        return self._check(self.http.get(path, **kw))

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._check(self.http.post(path, json=body))

    def patch(self, path: str, body: dict[str, Any]) -> Any:
        return self._check(self.http.patch(path, json=body))

    def delete(self, path: str) -> Any:
        return self._check(self.http.delete(path))

    def stream(self, path: str, body: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        with self.http.stream("POST", path, json=body | {"stream": True}) as r:
            if r.status_code >= 400:
                r.read()
                self._check(r)
            yield from parse_sse(r.iter_lines())


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    event, data = "message", list[str]()
    for line in lines:
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, json.loads("\n".join(data))


def footer(turn: dict[str, Any]) -> str:
    used, usage = turn["model_used"], turn["usage"]
    bits = [
        f"{used['name']} ({used['provider']}:{used['model_id']})",
        f"{usage['input_tokens']}→{usage['output_tokens']} tok",
        f"{turn['latency_ms'] / 1000:.1f}s",
    ]
    if usage.get("cost_usd") is not None:
        bits.append(f"${usage['cost_usd']:.5f}")
    if turn.get("history_trimmed"):
        bits.append(f"{turn['history_trimmed']} old msgs trimmed")
    line = _c(DIM, "— " + " · ".join(bits))
    if turn.get("fell_back"):
        tried = ", ".join(f"{a['model']} ({a['kind']})" for a in turn["attempts"])
        line += "\n" + _c(YELLOW, f"↪ fell back after: {tried}")
    return line


def run_turn(client: Client, path: str, body: dict[str, Any], stream: bool) -> dict[str, Any] | None:
    if not stream:
        turn = client.post(path, body)
        print(turn["message"]["content"])
        print(footer(turn), file=sys.stderr)
        return turn  # type: ignore[no-any-return]
    turn = None
    for event, data in client.stream(path, body):
        if event == "delta":
            print(data["text"], end="", flush=True)
        elif event == "fallback":
            print(
                _c(YELLOW, f"↪ {data['from']} failed ({data['error']}), trying {data['to']}"), file=sys.stderr
            )
        elif event == "done":
            turn = data
            print()
            print(footer(turn), file=sys.stderr)
        elif event == "error":
            if data.get("partial"):
                print()
            raise ApiError(
                f"{data.get('kind', 'error')}: {data.get('error')}"
                + (" (partial reply discarded)" if data.get("partial") else "")
            )
    return turn


# ---- commands ------------------------------------------------------------------------------------
def cmd_models(client: Client, args: argparse.Namespace) -> None:
    models = client.get("/v1/models")
    if args.json:
        print(json.dumps(models, indent=2))
        return
    w = max(len(m["name"]) for m in models)
    for m in models:
        mark = _c(BOLD, "*") if m["is_default"] else " "
        state = "ready" if m["available"] else _c(DIM, "not ready")
        alias = f" [{', '.join(m['aliases'])}]" if m["aliases"] else ""
        print(f"{mark} {m['name']:<{w}}  {m['provider']:<12} {state:<10} {m['model_id']}{_c(DIM, alias)}")
        if m.get("model_source"):
            print(_c(DIM, f"  {'':<{w}}  └ model ID from {m['model_source']}"))
        if not m["available"]:
            print(_c(DIM, f"  {'':<{w}}  └ {m['status']}"))
    print(_c(DIM, "\n* default. Ad-hoc models also work: -m bedrock:<model-id>, -m openai:<model>, ..."))


def cmd_ask(client: Client, args: argparse.Namespace) -> None:
    body = {
        "content": " ".join(args.text),
        "model": args.model,
        "system": args.system,
        "max_tokens": args.max_tokens,
    }
    turn = run_turn(client, "/v1/chat", {k: v for k, v in body.items() if v is not None}, not args.no_stream)
    if turn:
        print(
            _c(
                DIM,
                f'conversation: {turn["conversation_id"]}   (continue: convo say {turn["conversation_id"]} "...")',
            ),
            file=sys.stderr,
        )


def cmd_say(client: Client, args: argparse.Namespace) -> None:
    body = {"content": " ".join(args.text), "model": args.model, "max_tokens": args.max_tokens}
    run_turn(
        client,
        f"/v1/conversations/{args.id}/messages",
        {k: v for k, v in body.items() if v is not None},
        not args.no_stream,
    )


def cmd_reload(client: Client, args: argparse.Namespace) -> None:
    r = client.post("/v1/admin/reload-models", {})
    print(f"catalog reloaded: {r['models']} models, default {r['default_model']}")
    for ref in r["parameters"]:
        print(_c(DIM, f"  read {ref}"))
    if not r["changes"]:
        print("  no changes")
    for c in r["changes"]:
        what = c.get("model") or (f"alias {c['alias']}" if "alias" in c else "default")
        print(f"  {what}: {c['from']} → {c['to']}")


def cmd_swap(client: Client, args: argparse.Namespace) -> None:
    conv = client.patch(f"/v1/conversations/{args.id}", {"model": args.model})
    print(f"{conv['id']} now uses {conv['model']} (history kept: {len(conv['messages'])} messages)")


def print_history(conv: dict[str, Any]) -> None:
    print(_c(DIM, f"{conv['id']} · model {conv['model']} · {len(conv['messages'])} messages"))
    if conv.get("system"):
        print(_c(DIM, f"system: {conv['system']}"))
    for m in conv["messages"]:
        who = "you" if m["role"] == "user" else (m.get("model") or {}).get("name", "assistant")
        print(f"{_c(BOLD, who + ':')} {m['content']}")


def cmd_history(client: Client, args: argparse.Namespace) -> None:
    conv = client.get(f"/v1/conversations/{args.id}")
    if args.json:
        print(json.dumps(conv, indent=2))
    else:
        print_history(conv)


def cmd_list(client: Client, args: argparse.Namespace) -> None:
    for c in client.get("/v1/conversations", params={"limit": args.limit}):
        print(f"{c['id']}  {c['model']:<14} {c['turns']:>3} turns  {c['preview']!r}")


def cmd_rm(client: Client, args: argparse.Namespace) -> None:
    client.delete(f"/v1/conversations/{args.id}")
    print(f"deleted {args.id}")


REPL_HELP = """commands:
  /model NAME     swap model for the rest of the conversation (history is kept)
  /once NAME      use NAME for the next message only
  /models         list models
  /system TEXT    set the system prompt
  /history        show the conversation
  /new            start a new conversation
  /id             print the conversation id
  /quit           exit"""


def cmd_chat(client: Client, args: argparse.Namespace) -> None:
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401  (line editing + history)
    conv_id = args.id
    if conv_id is None:
        conv_id = client.post("/v1/conversations", {"model": args.model, "system": args.system})["id"]
    conv = client.get(f"/v1/conversations/{conv_id}")
    print(_c(DIM, f"conversation {conv_id} · model {conv['model']} · /help for commands"))
    once: str | None = None
    while True:
        try:
            text = input(_c(BOLD, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        try:
            if text.startswith("/"):
                cmd, _, arg = text.partition(" ")
                arg = arg.strip()
                if cmd in ("/quit", "/exit", "/q"):
                    return
                elif cmd == "/help":
                    print(REPL_HELP)
                elif cmd == "/model" and arg:
                    c = client.patch(f"/v1/conversations/{conv_id}", {"model": arg})
                    print(_c(DIM, f"model → {c['model']} (history kept)"))
                elif cmd == "/once" and arg:
                    once = arg
                    print(_c(DIM, f"next message uses {arg}"))
                elif cmd == "/models":
                    cmd_models(client, argparse.Namespace(json=False))
                elif cmd == "/system":
                    client.patch(f"/v1/conversations/{conv_id}", {"system": arg})
                    print(_c(DIM, "system prompt " + ("set" if arg else "cleared")))
                elif cmd == "/history":
                    print_history(client.get(f"/v1/conversations/{conv_id}"))
                elif cmd == "/new":
                    model = client.get(f"/v1/conversations/{conv_id}")["model"]
                    conv_id = client.post("/v1/conversations", {"model": model})["id"]
                    print(_c(DIM, f"new conversation {conv_id} · model {model}"))
                elif cmd == "/id":
                    print(conv_id)
                else:
                    print(REPL_HELP)
                continue
            body: dict[str, Any] = {"content": text}
            if once:
                body["model"], once = once, None
            print(_c(BOLD, "ai> "), end="", flush=True)
            run_turn(client, f"/v1/conversations/{conv_id}/messages", body, stream=True)
        except ApiError as exc:
            print(_c(RED, f"error: {exc}"), file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    from .app import run

    run(host=args.host, port=args.port, reload=args.reload)


# ---- entry point -----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=os.environ.get("CONVO_URL", "http://127.0.0.1:8080"))
    p.add_argument("--api-key", default=os.environ.get("CONVO_API_KEY"))
    p.add_argument(
        "--bearer",
        default=os.environ.get("CONVO_BEARER_TOKEN"),
        help="platform token sent as Authorization: Bearer (e.g. Cloud Run IAM)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--reload", action="store_true")

    s = sub.add_parser("models", help="list models and whether each is configured")
    s.add_argument("--json", action="store_true")

    for name, help_ in (("ask", "start a conversation with one message"), ("say", "continue a conversation")):
        s = sub.add_parser(name, help=help_)
        if name == "say":
            s.add_argument("id")
        s.add_argument("text", nargs="+")
        s.add_argument("-m", "--model")
        if name == "ask":
            s.add_argument("-s", "--system")
        s.add_argument("--max-tokens", type=int)
        s.add_argument("--no-stream", action="store_true")

    sub.add_parser("reload", help="re-read the catalog and Parameter Store values (no restart)")

    s = sub.add_parser("swap", help="change a conversation's model (history is kept)")
    s.add_argument("id")
    s.add_argument("model")

    s = sub.add_parser("history", help="show a conversation")
    s.add_argument("id")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("list", help="recent conversations")
    s.add_argument("--limit", type=int, default=20)

    s = sub.add_parser("rm", help="delete a conversation")
    s.add_argument("id")

    s = sub.add_parser("chat", help="interactive chat")
    s.add_argument("-m", "--model")
    s.add_argument("-s", "--system")
    s.add_argument("--id", help="resume an existing conversation")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        cmd_serve(args)
        return
    client = Client(args.url, args.api_key, args.bearer)
    handlers = {
        "models": cmd_models,
        "ask": cmd_ask,
        "say": cmd_say,
        "swap": cmd_swap,
        "reload": cmd_reload,
        "history": cmd_history,
        "list": cmd_list,
        "rm": cmd_rm,
        "chat": cmd_chat,
    }
    try:
        handlers[args.command](client, args)
    except ApiError as exc:
        print(_c(RED, f"error: {exc}"), file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(
            _c(
                RED, f"error: cannot reach {args.url}. Is the server running? (convo serve, or set CONVO_URL)"
            ),
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
