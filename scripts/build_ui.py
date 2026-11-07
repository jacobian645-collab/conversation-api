#!/usr/bin/env python3
"""Build the chat UI as a static site (AWS Amplify Hosting, S3 + CloudFront, any CDN).

    CONVO_API_URL=https://conversation-api.ecs.us-east-1.on.aws python3 scripts/build_ui.py
    python3 scripts/build_ui.py --api-url https://... --out dist/web

Writes <out>/index.html: the same page the API serves at /ui, plus the API's address and a
Content-Security-Policy that lets the page call only that API. Without an API URL the page still builds
and says it isn't connected yet, so a first Amplify build before the API exists succeeds.

Standard library only (Python 3.9+): it runs on Amplify's build image without installing anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_api import ui  # noqa: E402  (stdlib-only module; no FastAPI needed)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-url", default=os.environ.get("CONVO_API_URL", ""), help="default: $CONVO_API_URL")
    p.add_argument("--out", default="dist/web", help="output directory (default: dist/web)")
    args = p.parse_args(argv)

    api = args.api_url.strip()
    if api:
        try:
            ui.origin_of(api)
        except ValueError as exc:
            print(f"build_ui: CONVO_API_URL {exc}", file=sys.stderr)
            return 2
        if api.startswith("http://") and not api.startswith(("http://127.0.0.1", "http://localhost")):
            print(
                "build_ui: warning: the API URL is not HTTPS; browsers block it from an HTTPS page",
                file=sys.stderr,
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(ui.render_static(ui.load_page(), api or None), encoding="utf-8")
    print(f"build_ui: wrote {out / 'index.html'} (API: {api or 'not set'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
