"""S1 Report Aggregator Vertex subclass.

Loaded as the body of v_report by the graph (config: ``"script": "vertex/report_hook.py"``).
Accumulates AI summaries from summarize edges via on_receive and writes report.md to disk.
"""
import os

from framework.vertex import Vertex


class ReportVertex(Vertex):
    """Accumulates AI summaries across threads and outputs report.md."""

    def on_receive(self, data, channel, settings):
        # Accumulate reports across multiple on_receive calls.
        if not hasattr(self, "reports"):
            self.reports = []

        self.reports.append(data)

        lines = ["# S1 AI Discussion Report\n"]
        for i, report in enumerate(self.reports):
            lines += [f"## Thread {i + 1}", report, "\n---\n"]
        content = "\n".join(lines)

        out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # example root
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(content)

        return self.reports
