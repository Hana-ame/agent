# v7：DAG 本体

从原 `dag/` 抽出的稳定版本，保留模块级 API：

```python
import v7

v7.load("graph.json")      # 或 dict
v7.step()                  # 推进
print(v7.contents)         # tag -> 内容
v7.set("tag", "外部结果")   # 把外面处理的结果写回
v7.step()
print(v7.get_result("result"))
```

- `dag/` 仍保留，作为旧 import 路径；新代码优先 `v7`
- v8 的 graph 节点会优先 import `v7`，失败再回退 `dag`
- v7 负责：标签匹配、边执行、DefaultProcessor、Meta-DAG
- v8 负责：多张 v7 图的并行组织、合并、缓存、契约、落库
