# Custom Classes — 原生子类（script 加载）

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「自定义 Vertex/Edge 用原生子类而不是顶层 hook 函数」。

---

## 问题

早期框架支持「外部脚本导出顶层 hook 函数」（`on_receive`/`pre_process` 等模块级函数），
但那套机制已废弃：`load_class_from_script` 只找子类，找不到就静默降级成基类并打 warning。
按旧写法写的脚本，自定义行为根本不会执行。

## 方案

自定义行为 = 在外部 `.py` 里定义 **`Vertex`/`Edge` 子类**，config 用 `script` 引用，框架
动态加载并实例化。行为写在子类 override 的方法里（`on_receive`/`on_ready`/`pre_process`/
`post_process`）。

## 修改

- `examples/custom_classes/my_nodes.py`：
  - `SafeFilterVertex(Vertex)`：`on_receive` 校验非空 + 去 HTML 实体；
  - `PrefixEdge(Edge)`：`pre_process` 加 `[PRE]`、`post_process` 加 `[POST]`。
- `examples/custom_classes/config.json`：
  - `filter_node`：`"script": "my_nodes.py"`（自动发现唯一 `Vertex` 子类）；
  - `e_custom`：`"script": "my_nodes.py"`（自动发现唯一 `Edge` 子类）。

## 测试

**测试方案**：script 指向的 `.py` 里子类被正确加载并实例化，自定义方法生效。
**测试方法**：
```bash
python examples/run.py examples/custom_classes/config.json
```
**测试结果**：
- `SafeFilterVertex` 的 `on_receive` 生效：空数据被拒绝，非空数据去实体后进入节点；
- `PrefixEdge` 的 `pre_process`/`post_process` 生效：数据带 `[PRE]`…`[POST]`；
- 若 script 里没有子类，会打 `[ScriptLoader] ... 没有 X 子类，已降级用 X——自定义行为不会执行` warning
  （该示例不含这种情况，正常加载）。

> 多子类文件用 `script: 文件.py:类名` 显式指定；见 `s1_ai_report_map` / `hn_ai_report`。