"""Tests for the chatto「去 stage1st 收集帖子」task edge script (examples/s1profile_collect).

背景:把 chatto-bot 的 s1profile 每日任务(plugins/s1parse.py + plugins/s1profile.py)用
vertex-edge graph 实现:forum → e_collect(版块最新 N 帖按 uid 归并)→ posts_pool →
e_enrich(每人 do=thread 补抓 + 合并进 data/profiles/<uid>.json)→ profiles。

测试用罐头 HTML(离线),验证:版块/个人主页清单解析、楼层解析、档案合并去重(与 s1profile
的 (uid,page,content[:40]) 键一致)、graph 装配(两条链式边 + script 挂载)。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "s1profile_collect")
sys.path.insert(0, SCRIPT_DIR)
import s1profile_collect as sc  # noqa: E402

# ── 罐头 HTML ─────────────────────────────────────────────────────
INDEX_HTML = """
<div class="bm_c"><table>
  <tr><td><a href="thread-2206666-1-1.html" class="s xst">S1义父好人赞助楼。</a></td>
      <td class="by">2026-8-14 16:55</td></tr>
  <tr><td><a href="thread-334540-1-1.html" class="s xst">外野版规 - 禁止无期限投票</a></td>
      <td class="by">2020-9-11 18:29</td></tr>
  <tr><td><a href="thread-2288668-1-1.html" class="s xst">孙哥居然被坑了</a></td>
      <td class="by">2026-8-28 12:44</td></tr>
</table></div>
"""

THREAD_HTML = """
<html><body>
<div id="post_528683"><div class="authi">奥尔加·伊兹卡</div>
  <div class="pcb"><div class="t_fsz">
    <a href="home.php?mod=space&amp;uid=528683">奥尔加·伊兹卡</a>
    楼主发表于 2026-08-27 19:58
  </div><td class="t_f" id="postmessage_528683">五千万美元……</td></div>
</div>
<div id="post_111"><div class="authi">路人甲</div>
  <div class="pcb"><div class="t_fsz">
    <a href="home.php?mod=space&amp;uid=111">路人甲</a>
    发表于 刚刚
  </div><td class="t_f" id="postmessage_111">刚回帖测试</td></div>
</div>
</body></html>
"""


class TestParseIndex:
    def test_parse_index_extracts_threads(self):
        out = sc.parse_index(INDEX_HTML)
        assert len(out) == 3
        assert out[0]["tid"] == "2206666"
        assert out[0]["title"] == "S1义父好人赞助楼。"
        assert out[0]["url"].startswith("http")
        assert out[2]["last_reply"] == "2026-8-28 12:44"

    def test_parse_index_skips_non_first_page_links(self):
        html = INDEX_HTML.replace("thread-2206666-1-1.html", "thread-2206666-3-1.html")
        out = sc.parse_index(html)
        assert all(t["tid"] != "2206666" for t in out)


class TestParseThread:
    def test_parse_thread_floors(self):
        posts, total = sc.parse_thread_html(THREAD_HTML, page=1)
        assert len(posts) == 2
        first = posts[0]
        assert first["uid"] == "528683"
        assert first["username"] == "奥尔加·伊兹卡"
        assert first["time"] == "2026-08-27 19:58"
        assert first["content"] == "五千万美元……"
        assert first["page"] == 1

    def test_relative_time_normalized(self):
        posts, _ = sc.parse_thread_html(THREAD_HTML, page=1)
        second = posts[1]
        assert second["uid"] == "111"
        assert second["time"] and "-" in second["time"]  # 相对时间被规范成时间戳


class TestMergeProfiles:
    """镜像 s1profile._merge_posts:键 (uid, page, content[:40]),输出格式一致。"""

    def _tmpdir(self, tmp_path):
        return str(tmp_path / "profiles")

    def test_merge_adds_and_dedups(self, tmp_path):
        d = self._tmpdir(tmp_path)
        by_uid = {
            "528683": [
                {"uid": "528683", "username": "奥尔加·伊兹卡", "time": "2026-08-27 19:58",
                 "page": 1, "content": "五千万美元……"},
                {"uid": "528683", "username": "奥尔加·伊兹卡", "time": "2026-08-27 20:00",
                 "page": 1, "content": "五千万美元……"},  # 同 uid/page/正文 → 应去重
            ]
        }
        added = sc.merge_into_profiles(by_uid, d)
        assert added == {"528683": 1}
        prof = json.load(open(os.path.join(d, "528683.json"), encoding="utf-8"))
        assert prof["uid"] == "528683"
        assert prof["username"] == "奥尔加·伊兹卡"
        assert len(prof["posts"]) == 1
        assert prof["posts"][0]["content"] == "五千万美元……"
        assert prof["posts"][0]["time"] == "2026-08-27 19:58"
        assert "revisits" in prof

    def test_merge_persists_and_no_duplicate_on_rerun(self, tmp_path):
        d = self._tmpdir(tmp_path)
        by_uid = {"111": [{"uid": "111", "username": "路人甲", "time": "2026-08-28 10:00",
                           "page": 1, "content": "第一条"}]}
        sc.merge_into_profiles(by_uid, d)
        # 第二次跑同样的数据:新增 0,档案条数不变(幂等,对应 s1profile 每日重复收集)
        added = sc.merge_into_profiles(by_uid, d)
        assert added == {"111": 0}
        prof = json.load(open(os.path.join(d, "111.json"), encoding="utf-8"))
        assert len(prof["posts"]) == 1


class TestGraph:
    def test_config_loads_chain_with_edge_scripts(self):
        cfg = os.path.join(SCRIPT_DIR, "config.json")
        graph = Graph.from_json(cfg)
        assert list(graph.vertices) == ["forum", "posts_pool", "profiles"]
        assert set(graph.edges) == {"e_collect", "e_enrich"}
        e1 = graph.edges["e_collect"]
        e2 = graph.edges["e_enrich"]
        assert e1.source_id == "forum" and e1.destination_id == "posts_pool"
        assert e2.source_id == "posts_pool" and e2.destination_id == "profiles"
        assert e1.settings["stage"] == "collect"
        assert e2.settings["stage"] == "enrich"
        for e in (e1, e2):
            assert e._pipeline_module is not None
            assert hasattr(e._pipeline_module, "pre_process")
            assert hasattr(e._pipeline_module, "post_process")

    def test_config_json_valid(self):
        with open(os.path.join(SCRIPT_DIR, "config.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["metadata"]["name"] == "S1 Profile Collect"
