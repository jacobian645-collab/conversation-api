from __future__ import annotations

import re
from typing import Any

from fastapi.testclient import TestClient

from conversation_api import ui
from conversation_api.app import ui_page


def test_root_redirects_to_ui_and_page_is_served(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/ui"

    assert client.head("/ui").status_code == 200
    r = client.get("/ui")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>Conversation API</title>" in r.text
    # the page talks to the same API the CLI uses
    for path in ("/v1/models", "/v1/conversations", "/messages", "/v1/admin/reload-models"):
        assert path in r.text


def test_ui_csp_pins_the_inline_script_and_same_origin_calls(client: TestClient) -> None:
    r = client.get("/ui")
    csp = r.headers["content-security-policy"]
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";", 1)[0]
    scripts = re.findall(r"<script>(.*?)</script>", r.text, flags=re.S)
    assert len(scripts) == 1
    assert csp.count("'sha256-") == 1
    assert r.headers["x-content-type-options"] == "nosniff"


def test_ui_needs_no_key_but_the_api_still_does(make_client: Any) -> None:
    with make_client(api_key="s3cret") as c:
        assert c.get("/ui").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_ui_can_be_turned_off(make_client: Any) -> None:
    with make_client(ui=False) as c:
        assert c.get("/ui").status_code == 404
        assert c.get("/").status_code == 404
        assert c.get("/healthz").status_code == 200


def test_ui_is_not_in_the_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/ui" not in paths and "/" not in paths


def test_ui_page_ships_inside_the_package() -> None:
    html, headers = ui_page()
    assert html.startswith("<!doctype html>")
    assert "Content-Security-Policy" in headers


# ---- hosted on another origin (AWS Amplify) -----------------------------------------------------------
AMPLIFY = "https://main.d1abc2def3.amplifyapp.com"


def test_cors_lets_the_hosted_page_call_the_api(make_client: Any) -> None:
    with make_client(api_key="s3cret", cors_origins=f"{AMPLIFY}/, https://chat.example.com") as c:
        pre = c.options(
            "/v1/conversations/conv_x/messages",
            headers={
                "Origin": AMPLIFY,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-api-key",
            },
        )
        assert pre.status_code == 200  # answered before auth: preflights carry no key
        assert pre.headers["access-control-allow-origin"] == AMPLIFY
        assert "x-api-key" in pre.headers["access-control-allow-headers"].lower()
        assert "access-control-allow-credentials" not in pre.headers

        # errors carry CORS headers too, so the page can read the 401 and ask for a key
        r = c.get("/v1/models", headers={"Origin": AMPLIFY})
        assert r.status_code == 401 and r.headers["access-control-allow-origin"] == AMPLIFY
        r = c.get("/v1/models", headers={"Origin": "https://chat.example.com", "X-API-Key": "s3cret"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "https://chat.example.com"
        assert "x-request-id" in r.headers["access-control-expose-headers"].lower()


def test_cors_is_off_unless_configured(client: TestClient, make_client: Any) -> None:
    r = client.get("/healthz", headers={"Origin": AMPLIFY})
    assert "access-control-allow-origin" not in r.headers
    with make_client(cors_origins=AMPLIFY) as c:
        r = c.get("/healthz", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers


def test_static_build_points_the_page_at_the_api(tmp_path: Any) -> None:
    from scripts.build_ui import main  # type: ignore[import-not-found]

    api = "https://conversation-api.ecs.us-east-1.on.aws/"
    assert main(["--api-url", api, "--out", str(tmp_path)]) == 0
    html = (tmp_path / "index.html").read_text()
    assert '<meta name="convo-api" content="https://conversation-api.ecs.us-east-1.on.aws">' in html
    csp = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', html)
    assert csp
    policy = csp.group(1)
    assert "connect-src https://conversation-api.ecs.us-east-1.on.aws;" in policy
    assert "frame-ancestors" not in policy  # ignored in <meta>; customHttp.yml sends it as a header
    # the hash still matches the script after the tags are injected
    assert len(re.findall(r"<script>(.*?)</script>", html, flags=re.S)) == 1
    assert ui.script_hashes(html)[0] in policy
    # CSP meta must precede the stylesheet and script it governs
    assert (
        html.index("Content-Security-Policy") < html.index('<link rel="stylesheet"') < html.index("<script>")
    )


def test_static_build_without_api_url_still_builds_and_calls_nothing(tmp_path: Any) -> None:
    from scripts.build_ui import main  # type: ignore[import-not-found]

    assert main(["--api-url", "", "--out", str(tmp_path)]) == 0
    html = (tmp_path / "index.html").read_text()
    assert '<meta name="convo-api" content="">' in html
    assert "connect-src 'none'" in html


def test_static_build_rejects_a_bad_api_url(tmp_path: Any) -> None:
    from scripts.build_ui import main  # type: ignore[import-not-found]

    assert main(["--api-url", "conversation-api.example.com", "--out", str(tmp_path)]) == 2
    assert main(["--api-url", "https://user:pw@api.example.com", "--out", str(tmp_path)]) == 2
    assert not (tmp_path / "index.html").exists()


def test_attribute_escaping_in_injected_tags() -> None:
    html = ui.render_static(ui.load_page(), 'https://api.example.com/"><script>alert(1)</script>')
    assert "<script>alert(1)</script>" not in html
