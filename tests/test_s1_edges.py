"""Regression tests for examples/s1_ai_report*/s1_edges.py post parsing.

These tests run against a real saved stage1st thread page
(tests/fixtures/s1_thread.html) so they are offline / deterministic.

They lock in two bugs that were fixed (and previously lost) twice:

1. The post selector ``div[id^="post_"]`` also matched the empty rating
   placeholder ``id="post_rate_div_<pid>"``, producing bogus empty "posts".
2. The timestamp is rendered as ``发表于 2026-8-29 13:10`` (Chinese prefix,
   no ``span[title]``), which the old ``re.match``/``strptime`` logic failed to
   parse -> ``dt=None`` -> every post was dropped by the 24h cutoff ->
   ``Result: 0 replies`` even for busy threads.
"""
import importlib.util
import os
import sys

import pytest
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "s1_thread.html")

EDGE_PATHS = [
    os.path.join(ROOT, "examples", "s1_ai_report_map", "s1_edges.py"),
    os.path.join(ROOT, "examples", "s1_ai_report", "s1_edges.py"),
]


def _load_s1_edges(path):
    sys.path.insert(0, ROOT)  # framework package
    name = "s1_edges_" + os.path.basename(os.path.dirname(path)).replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def soup():
    with open(FIXTURE, encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "html.parser")


@pytest.mark.parametrize("path", EDGE_PATHS)
def test_parse_posts_skips_rate_divs(path, soup):
    mod = _load_s1_edges(path)
    posts = mod._parse_posts_from_soup(soup)
    # Real thread page has many posts; rate_div placeholders must be excluded.
    assert len(posts) >= 5
    for idx, user, _ts, _dt, content in posts:
        assert user, f"post {idx} has empty user"
        assert content and content != "(empty)", f"post {idx} has empty content"


@pytest.mark.parametrize("path", EDGE_PATHS)
def test_chinese_timestamp_parsed(path, soup):
    mod = _load_s1_edges(path)
    posts = mod._parse_posts_from_soup(soup)
    # The "发表于 2026-8-29 13:10" style must be parsed to a real datetime
    # (not None), otherwise the 24h cutoff silently drops every post.
    parsed = [dt for _u, _t, _ts, dt, _c in posts if dt is not None]
    assert parsed, "no post had its timestamp parsed"
    assert len(parsed) >= 3
    # sanity: the parsed year should be a plausible recent year
    assert all(dt.year >= 2020 for dt in parsed)


def test_time_regex_handles_prefix():
    from datetime import datetime, timedelta, timezone

    raw = "发表于 2026-8-29 13:10"
    stripped = __import__("re").sub(r"^(发表于|Post on|Posted at)[\s:：]*", "", raw).strip()
    m = __import__("re").search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2}):(\d{2})", stripped)
    assert m, "timestamp regex did not match 2026-8-29 13:10"
    y, mo, d, h, mi = map(int, m.groups())
    dt = datetime(y, mo, d, h, mi, tzinfo=timezone(timedelta(hours=8)))
    assert dt.year == 2026 and dt.month == 8 and dt.day == 29
