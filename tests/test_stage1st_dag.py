"""Automated unit test for Stage1st 157 AI Crawler and Digest DAG Pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient
from examples.ex23_stage1st_ai_crawler_and_digest_dag import (
    fetch_thread_24h_posts,
    run_stage1st_pipeline,
    scrape_s1_ai_catalog,
)


class TestStage1stPipelineDAG:
    """Test suite validating S1 157 forum scraping, Python interleaving, and multi-stage DAG synthesis."""

    def test_01_catalog_scraper_filters_ai_topics(self):
        threads = scrape_s1_ai_catalog()
        assert len(threads) > 0
        for t in threads:
            assert "tid" in t
            assert "title" in t
            assert "url" in t
            assert t["board_id"] == "157"

    def test_02_thread_24h_posts_fetcher(self):
        posts = fetch_thread_24h_posts("2275806", "DeepSeek", "https://stage1st.com/2b/thread-2275806-1-1.html")
        assert len(posts) > 0
        for p in posts:
            assert "time" in p
            assert "text" in p
            assert len(p["text"]) > 0

    def test_03_end_to_end_pipeline_with_mock_llm(self):
        """Verify the full interleaving pipeline: DAG Step 1 -> Python fetch -> DAG Resume -> Python readout."""
        res = run_stage1st_pipeline(use_live_llm=False)
        assert res["dag"] is not None
        assert len(res["ai_threads"]) > 0
        assert "# S1 归墟板块 (157)" in res["markdown_digest"]
        assert res["summary"] is not None
        assert res["timeline"] is not None
        assert res["final_report"] is not None

        # Verify DAG state
        dag = res["dag"]
        assert dag.has_tag("ai_news_links")
        assert dag.has_tag("recent_ai_discussions_md")
        assert dag.has_tag("ai_discussion_summary")
        assert dag.has_tag("ai_timeline")
        assert dag.has_tag("final_ai_report")
