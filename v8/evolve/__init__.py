"""evolve —— 受约束的 DAG 演化 + 人工评价候选。

最小闭环：

    Genome（DAG 配置 + 运行参数）
      → mutate（prompt / thinking / contract / critique 节点 / workers）
      → run（dag8 执行，落独立目录）
      → 自动检查（结构/长度/关键词）
      → 人工评价包（review.md + ratings.json）
      → select（human / auto / combined）

设计取向：

- 评测器与候选隔离：候选只改 DAG 配置，不碰 `evolve` 本体；
- 人工评价是一等公民：每个候选都有独立输出文件、评分位与备注；
- 没有可信自动指标时，`--by human` 仍然可以选出候选；
- 所有结果可复现：genome、run 目录、sqlite、review.json 都落盘。
"""
from evolve.genome import Genome
from evolve.leakage import audit, audit_optional
from evolve.tasks import Task, check_output
from evolve.review import build_review, rate_candidate, select_candidate

__all__ = ["Genome", "Task", "check_output", "build_review", "rate_candidate",
           "select_candidate", "audit", "audit_optional"]
