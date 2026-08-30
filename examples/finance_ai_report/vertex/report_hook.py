"""Finance Report Aggregator Vertex subclass.

Loaded as the body of v_report by the graph (config: ``"script": "vertex/report_hook.py"``).
Accumulates AI summaries from summarize edges via on_receive and writes report.md to disk.

Output path:
- Default: the example root (``<example>/report.md``), same as s1/hn examples.
- Override: set env ``FINANCE_REPORT_OUT`` to a target file path. The chatto-bot
  plugin uses this so the bot keeps its own report copy and never collides with
  manual demo runs (which write the example root).
"""
import os
import re

from framework.vertex import Vertex


class ReportVertex(Vertex):
    """Accumulates AI summaries across threads and outputs report.md."""

    def on_receive(self, data, channel, settings):
        # Accumulate reports across multiple on_receive calls.
        if not hasattr(self, "reports"):
            self.reports = []

        self.reports.append(data)

        lines = ["# 财经讨论报告\n"]
        for index, report in enumerate(self.reports, start=1):
            if isinstance(report, dict):
                # Structured summary: title/url come from the fetched thread
                # data (not restated by the LLM), summary is the LLM body.
                title = report.get("title", "Unknown")
                url = report.get("url", "")
                summary = report.get("summary", "")
                # 把 LLM 正文里的 ## 子标题(【事件】/【观点】/【影响】)降级成 ###,
                # 从属于帖子的 ## N. 序号标题,层级清晰(否则同级 ## 在分段发送时
                # 会被 _split_for_post 当成新帖切开)。
                summary = re.sub(r"(?m)^## ", "### ", summary)
                lines += [f"## {index}. [{title}]({url})", "", summary, "\n---\n"]
            else:
                lines += [str(report), "\n---\n"]
        content = "\n".join(lines)

        out = os.environ.get("FINANCE_REPORT_OUT")
        if out:
            # bot 侧指定输出文件;目录若不存在则自动创建
            out_dir = os.path.dirname(os.path.abspath(out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # example root
            out = os.path.join(out_dir, "report.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)

        return self.reports