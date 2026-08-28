"""S1 报告汇聚 Vertex 子类。

作为 v_report 的本体被 graph 加载（config: ``"script": "vertex/report_hook.py"``，
框架用 load_class_from_script 自动发现本文件中的 Vertex 子类并实例化）。
每条 summarize 边的产出经 on_receive 累积，并实时生成 report.md。

发现背景：原为模块级函数 on_receive，靠 Vertex.set_pipeline_module 挂载——
但 Vertex 从未实现 module 委托，该路径加载即 AttributeError（死代码）。
随框架统一转纯子类覆盖，改为 ReportVertex(Vertex)。
"""
import os

from framework.vertex import Vertex


class ReportVertex(Vertex):
    """累积各 thread 的 AI 摘要，落盘 report.md。"""

    def on_receive(self, data, channel, settings):
        # reports 跨多次 on_receive 累积；用实例属性替代原模块级全局
        # （on_receive.reports）。懒初始化以避免重写 __init__。
        if not hasattr(self, "reports"):
            self.reports = []

        self.reports.append(data)

        lines = ["# S1 AI Discussion Report\n"]
        for i, report in enumerate(self.reports):
            lines += [f"## Thread {i + 1}", report, "\n---\n"]
        content = "\n".join(lines)

        out_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(content)

        return self.reports
