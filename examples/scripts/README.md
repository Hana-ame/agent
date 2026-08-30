# Scripts — 公共子类脚本

> 被 `examples/complex`、`examples/custom_classes` 等通过 `script` 引用。全部是**子类**，
> 不是顶层 hook 函数（旧写法已失效）。按「问题/方案/修改/测试」记录。

## 问题
多个示例需要相同的「数据变换」逻辑（转大写、前后缀、校验），应复用而不是各自复制。

## 方案
放公共目录，config 用相对 `script` 引用（相对 config 文件目录解析）。

## 修改
- `uppercase_handler.py`：`UpperVertex(Vertex)` — `on_receive` 转大写、`on_ready` 汇总。
- `prefix_handler.py`：`PrefixEdge(Edge)` — `pre_process` 加 `[PRE]`、`post_process` 加 `[POST]`（可配 `settings.prefix/suffix`）。
- `validator.py`：`ValidatorVertex(Vertex)` — `on_receive` 校验并拒绝非法数据。

## 测试
**测试方案**：三个脚本被子类加载、行为生效。**测试方法**：
`python examples/run.py examples/complex/config.json`（引用 uppercase+prefix）。
**测试结果**：`transform` 大写、`e3` 前后缀生效；`tests/test_script_loader.py` 覆盖加载。