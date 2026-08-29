"""HN Report Aggregator Vertex subclass.

Loaded as the body of v_report by the graph (config: ``"script": "vertex/report_hook.py"``).
Accumulates AI summaries from summarize edges via on_receive and writes report.md to disk.
"""
import os

from framework.vertex import Vertex


class ReportVertex(Vertex):
    """Accumulates AI summaries across stories and outputs report.md."""

    def on_receive(self, data, channel, settings):
        # Accumulate reports across multiple on_receive calls.
        if not hasattr(self, "reports"):
            self.reports = []

        self.reports.append(data)

        lines = ["# Hacker News AI Report\n"]
        for report in self.reports:
            if isinstance(report, dict):
                # Structured summary: title/url come from the fetched story
                # data (not restated by the LLM), summary is the LLM body.
                title = report.get("title", "Unknown")
                url = report.get("url", "")
                summary = report.get("summary", "")
                lines += [f"# [{title}]({url})", "", summary, "\n---\n"]
            else:
                lines += [str(report), "\n---\n"]
        content = "\n".join(lines)

        out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # example root
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(content)

        return self.reports