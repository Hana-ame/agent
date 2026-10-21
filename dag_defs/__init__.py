"""任务 DAG 定义目录（非库）。

每个模块用 `@register(...)` 注册一条任务 DAG；`dags.registry.load_definitions()`
会在 `import dags` 时自动发现并加载本目录下的所有模块。
"""
