import os
import asyncio
from framework.vertex import Vertex

class ReportVertex(Vertex):
    """Accumulates summaries of HN AI stories and outputs to report.md."""

    def on_receive(self, data, channel, settings):
        if not hasattr(self, "reports"):
            self.reports = []
        
        # When using MapEdge, data might come as a list of results
        if isinstance(data, list):
            self.reports.extend(data)
        else:
            self.reports.append(data)
            
        return self.reports

    def on_ready(self, all_data, settings):
        reports = getattr(self, "reports", [])
        lines = ["# Hacker News AI Report\n"]
        for i, report in enumerate(reports):
            lines += [f"## Story {i + 1}", str(report), "\n---\n"]
        content = "\n".join(lines)

        out_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(out_dir, "..", "report.md"), "w", encoding="utf-8") as f:
            f.write(content)
            
        return {"report_md": content}
