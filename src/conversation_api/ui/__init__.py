"""The browser chat UI: one static page, served two ways.

* By the API itself at /ui (same origin): the page calls relative /v1 paths.
* As a static site (AWS Amplify Hosting or any CDN): `scripts/build_ui.py` writes a copy that carries the
  API's address in a <meta name="convo-api"> tag and its Content-Security-Policy in a <meta> tag, because a
  static host can't compute the script hash or know the API origin.

Standard library only, so the static build runs on a plain build image without installing the package.
"""

from __future__ import annotations

import base64
import hashlib
import re
from importlib import resources
from urllib.parse import urlsplit

PAGE = "index.html"
FONT_CSS = "https://fonts.googleapis.com"
FONT_FILES = "https://fonts.gstatic.com"


def load_page() -> str:
    return resources.files(__name__).joinpath(PAGE).read_text(encoding="utf-8")


def script_hashes(html: str) -> list[str]:
    """CSP source expressions for every inline <script> in the page."""
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(s.encode()).digest()).decode() + "'" for s in scripts
    ]


def origin_of(url: str) -> str:
    """https://host[:port] of an API URL; raises ValueError for anything that isn't an absolute http(s) URL."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"not an absolute http(s) URL: {url!r}")
    if parts.username or parts.password:
        raise ValueError("the API URL must not contain credentials")
    return f"{parts.scheme}://{parts.netloc}".lower()


def content_security_policy(html: str, connect_src: str = "'self'", *, header: bool = True) -> str:
    """The page may run only its own inline script and may call only `connect_src`.

    frame-ancestors is ignored inside a <meta> CSP, so a static host sends it as a header instead
    (see customHttp.yml)."""
    directives = [
        "default-src 'none'",
        "script-src " + " ".join(script_hashes(html)),
        f"style-src 'unsafe-inline' {FONT_CSS}",
        f"font-src {FONT_FILES}",
        "img-src data:",
        f"connect-src {connect_src}",
        "base-uri 'none'",
        "form-action 'none'",
    ]
    if header:
        directives.append("frame-ancestors 'none'")
    return "; ".join(directives)


def security_headers(html: str) -> dict[str, str]:
    return {
        "Content-Security-Policy": content_security_policy(html),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-cache",
    }


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def render_static(html: str, api_url: str | None) -> str:
    """The page for a static host. With no api_url the page says it hasn't been pointed at an API yet."""
    api = api_url.strip().rstrip("/") if api_url else ""
    connect = origin_of(api) if api else "'none'"
    tags = (
        f'<meta http-equiv="Content-Security-Policy" content="{_attr(content_security_policy(html, connect, header=False))}">\n'
        f'<meta name="convo-api" content="{_attr(api)}">\n'
    )
    marker = '<meta charset="utf-8">\n'
    if marker not in html:
        raise ValueError("page has no <meta charset> to anchor the injected tags")
    # The CSP meta must come before the stylesheet and script it governs.
    return html.replace(marker, marker + tags, 1)
