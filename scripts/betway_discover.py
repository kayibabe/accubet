"""Dev tool: drive betway.mw in a headless browser and capture its odds-feed network
calls, so we can learn the exact working endpoints (and JSON shape) for the connector.

Run: .\.venv\Scripts\python.exe scripts\betway_discover.py
"""

from __future__ import annotations

import json
import re

from playwright.sync_api import sync_playwright

FEED_RE = re.compile(r"_apis|feeds|sport|event|market|highlight", re.I)
START_URLS = [
    "https://www.betway.mw/",
    "https://www.betway.mw/sport/soccer",
    "https://www.betway.mw/sport",
]


def main() -> None:
    captured: list[tuple[str, int, str]] = []
    json_hits: list[tuple[str, object]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            )
        )
        page = ctx.new_page()

        def on_response(resp):
            url = resp.url
            if not FEED_RE.search(url):
                return
            captured.append((resp.request.method, resp.status, url))
            ctype = resp.headers.get("content-type", "")
            if "json" in ctype and resp.status == 200 and len(json_hits) < 12:
                try:
                    json_hits.append((url, resp.json()))
                except Exception:
                    pass

        page.on("response", on_response)

        for url in START_URLS:
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(6000)
            except Exception as exc:
                print(f"[goto] {url} -> {type(exc).__name__}: {exc}")

        browser.close()

    print(f"\n==== captured {len(captured)} feed-ish responses ====")
    seen = set()
    for method, status, url in captured:
        key = re.sub(r"\d{3,}", "N", url.split("?")[0])  # collapse ids
        if key in seen:
            continue
        seen.add(key)
        print(f"{status} {method:4} {url[:140]}")

    print(f"\n==== {len(json_hits)} JSON payloads (top-level shape) ====")
    for url, payload in json_hits:
        if isinstance(payload, dict):
            shape = list(payload.keys())[:12]
        elif isinstance(payload, list):
            shape = f"list[{len(payload)}] item0_keys=" + str(
                list(payload[0].keys())[:12] if payload and isinstance(payload[0], dict) else type(payload[0]).__name__ if payload else "empty"
            )
        else:
            shape = type(payload).__name__
        print(f"\n--- {url[:130]}")
        print(f"    shape: {shape}")
        snippet = json.dumps(payload)[:500]
        print(f"    snippet: {snippet}")


if __name__ == "__main__":
    main()
