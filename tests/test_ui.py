"""Playwright tests against the published web build in site/ (run after pipeline.run)."""
import functools
import http.server
import threading
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent / "site"
BOARDS = ["QB", "RB", "WR", "TE", "OT", "IOL", "EDGE", "IDL", "LB", "CB", "S", "K", "P", "RET"]

pytestmark = pytest.mark.skipif(not (SITE / "latest.json").exists(), reason="no site/ snapshot; run pipeline.run first")


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


@pytest.fixture(scope="module")
def page(server):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context()
        ctx.route("**/fonts.googleapis.com/**", lambda r: r.abort())
        pg = ctx.new_page()
        pg.set_default_timeout(15000)
        yield pg
        b.close()


def test_every_board_renders(page, server):
    for board in BOARDS:
        page.goto(f"{server}#/board/{board}")
        page.wait_for_selector("h1")
        rows = page.locator(".rank-table tbody tr")
        empty = page.locator(".empty")
        page.wait_for_function("document.querySelector('.rank-table tbody tr') || document.querySelector('.empty')")
        assert rows.count() > 0 or empty.count() > 0, board


def test_search_finds_player_by_partial_name(page, server):
    page.goto(f"{server}#/board/QB")
    page.wait_for_selector(".rank-table tbody tr")
    first = page.locator(".rank-table tbody tr .pname").first.inner_text()
    partial = first.split()[-1][:4]
    page.keyboard.press("Meta+k")
    if not page.locator("#palette.open").count():
        page.keyboard.press("Control+k")
    page.fill("#pq", partial)
    page.wait_for_selector("#plist li[data-id]")
    assert first in page.locator("#plist").inner_text()


def test_player_page_shows_every_component(page, server):
    page.goto(f"{server}#/board/WR")
    page.wait_for_selector(".rank-table tbody tr")
    page.locator(".rank-table tbody tr").first.click()
    page.wait_for_selector(".comprow")
    assert page.locator(".comprow").count() == 6           # WR has six components
    assert page.locator(".mt tr.group").count() == 6
    assert page.locator("text=Estimated").count() >= 1      # run blocking is marked estimated


def test_falls_back_to_cache_when_offline(page, server):
    page.goto(f"{server}#/board/QB")
    page.wait_for_selector(".rank-table tbody tr")
    page.wait_for_timeout(1500)                               # let the prefetch cache rankings
    # the app ships its UI, so "offline" means the data host is unreachable
    block = lambda route: route.abort("internetdisconnected")
    page.route("**/latest.json", block)
    page.route("**/20[0-9][0-9]/**", block)
    try:
        page.reload()
        page.wait_for_selector(".rank-table tbody tr")
        assert "offline" in page.locator("#pill").inner_text()
    finally:
        page.unroute("**/latest.json")
        page.unroute("**/20[0-9][0-9]/**")
