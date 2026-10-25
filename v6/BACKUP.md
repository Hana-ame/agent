# v6 备份（Vertex-Edge Agent V6）

- 来源分支：`origin/vea-v6-bootstrap-line`
- 来源提交：`4e924e3`
- 状态：**历史设计备份，不参与当前 v7/v8/v9 运行**
- 内容：纯标签信息流、preset edge 合成、元图自举

## 为什么留

- preset edge：缺 tag 时从相近内容合成临时输入（当前 v7 DefaultProcessor 有类似能力，v8 没要）
- 外部终止循环：`while no result: dag.run()`
- 元图自举：DAG 生成 DAG

## 注意

`models.json.example` 是历史模型清单（含已不可用的模型），只作参考，不要直接用于当前运行。
