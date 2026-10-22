# v8 · JSON 配置 + pi agent + sqlite3

v8 把框架缩到三件事：

| 层 | 做什么 | 代码 |
|---|---|---|
| 配置 | 纯 JSON 描述图与 prompt | `dag8/config.py` |
| 执行 | 每条 edge 交给 `pi`（coding agent）去跑 | `dag8/pi.py` |
| 存储 | 内容、执行记录全落 sqlite3 | `dag8/store.py` |

引擎只有 140 行（`dag8/engine.py`）：按 `layer` 顺序跑边，tag 精确匹配取输入，缺输入就跳过，出错就记 error 停下。

## 与 v7 的区别

| | v7 | v8 |
|---|---|---|
| edge 执行者 | Python 函数 / HTTP LLM 客户端 | **`pi` agent（子进程）** |
| 内容存放 | 内存 + 输出 Markdown | **sqlite3**（可查询、可回溯） |
| prompt 定义 | 写死在 Python 里 | **配置 JSON 的一个字段** |
| 图定义 | Python 代码 | **JSON** |
| 依赖 | `requests` / `httpx` | 标准库 + `pi` 二进制 |

v7 的 `scripts/` 保留不动；v8 是独立目录，互不影响。

## 快速开始

```bash
cd v8
python -m dag8 run configs/demo.json            # 三边流水线：起草 → 挑错 → 定稿
python -m dag8 run configs/demo.json --dry-run  # 不调 pi，只验配置与占位符渲染
python -m dag8 runs                             # 列出历史运行
python -m dag8 show <run_id>                    # 看某次运行的全部中间内容
```

实测（`laptop/deepseek-v4-flash`）：

```
run 77c97fbb431a · demo_pipeline · ok
  result: final = 320 字符
  ✓ [0] draft      5.8s   238 字符
  ✓ [1] critique  12.0s    74 字符
  ✓ [2] final     26.1s   320 字符
```

## 配置格式

```json
{
  "name": "demo_pipeline",
  "seed_tag": "task",
  "seed_content": "用三句话解释什么是标签驱动的 DAG",
  "result_tag": "final",
  "pi": {
    "provider": "laptop",
    "model": "deepseek-v4-flash",
    "tools": false,
    "timeout": 180,
    "retries": 3,
    "ssh_host": "lumin@127.0.0.1",
    "ssh_key": "~/.ssh/id_rsa"
  },
  "edges": [
    {
      "id": "draft",
      "layer": 0,
      "input_tags": ["task"],
      "output_tags": ["draft"],
      "prompt": "任务：{contents}"
    },
    {
      "id": "final",
      "layer": 1,
      "input_tags": ["draft", "critique"],
      "output_tags": ["final"],
      "prompt": "【草稿】{tag:draft}\n\n【批评】{tag:critique}"
    }
  ]
}
```

### prompt 占位符

| 占位符 | 替换成 |
|---|---|
| `{contents}` | 该边所有输入 tag 的内容拼接（`---` 分隔） |
| `{content}` | 只取最新一条 |
| `{tag:名称}` | 指定 tag 的内容 |
| `{a,b}` | 多 tag 简写 |

**未识别的占位符原样保留** —— prompt 里给模型看的 `{"ai": [1,3]}` 这类 JSON 示例不会被吃掉。

### pi 的两种执行后端

- `ssh_host` 为空：直接跑本机 `pi`
- `ssh_host` 有值：`ssh host 'export PATH=<nvm>; pi -p ...'`，prompt 走 stdin

WSL 有两个可用地址：`172.29.89.192`（eth0 实际 IP，示例配置用它）和 `127.0.0.1`
（Windows 侧 `wslrelay` 转发）。**前者在 WSL 重启后会变**，变了用
`DAG8_SSH_HOST=lumin@<新IP> python -m dag8 run ...` 覆盖，或把配置改回 127。

prompt 用 stdin 传而不是命令行参数，避免多层 shell 的引号和换行问题（实测命令行方式在多行中文 prompt 上会被截断）。

## 存储

`v8/data/dag8.sqlite3`，三张表：

```sql
runs     (run_id, dag_name, config_path, seed_tag, seed_content,
          started_at, finished_at, status, error, result_tag, result_content_id)
contents (id, run_id, tags, content, gen, source_edge, created_at)
edges    (id, run_id, edge_id, layer, input_tags, output_tags, model,
          prompt_chars, status, error, elapsed, content_id, created_at)
```

一次运行 = 一个 `run_id`，所有中间内容都挂在它下面。同一个配置跑多次会在库里留下多组 run，可以直接对比输出差异。

```sql
-- 找出某次运行里所有 edge 的耗时
SELECT edge_id, model, elapsed FROM edges WHERE run_id='...' ORDER BY id;

-- 对比同一配置两次运行的最终产出
SELECT run_id, content FROM contents WHERE tags LIKE '%final%' AND run_id IN (...);
```

## 测试

```bash
cd v8 && python -m pytest tests/ -q                  # 17 条，含真实调用，约 2 分钟
cd v8 && python -m pytest tests/ -q -m "not slow"    # 跳过整图那条，约 50 秒
```

| 文件 | 条数 | 是否碰 pi |
|---|---|---|
| `tests/test_dag8.py` | 13 | 否 —— 配置加载、占位符渲染、sqlite 读写、调度（注入 fake agent） |
| `tests/test_live_pi.py` | 4 | **是** —— 真实调用 `laptop/deepseek-v4-flash` |

live 的四条：

1. 单次直调：返回「收到」，耗时 < 120s
2. provider 写错时抛 `PiError`，而不是静默返回空串
3. 边级：engine 渲染 prompt → pi 执行 → 结果落 sqlite（校验 contents 条数、edge 耗时 > 0、run 状态 `ok`）
4. 整图三边（标 `slow`）：起草 → 挑错 → 定稿，校验多输入边的 `prompt_chars`

gating：`tests/_probe.py` 只跑 `ssh ... command -v pi`（**不调模型**）。探不到时**直接 fail**，
不是 skip —— skip 会让「环境没准备好」和「全部通过」长得一模一样，那是假绿。
环境变量可覆盖：`DAG8_SSH_HOST` / `DAG8_SSH_KEY` / `DAG8_LOCAL_PI=1`（本机有 pi 时不走 ssh）。

实测：全量 `17 passed in 123.78s`；跳过 slow `16 passed, 1 deselected in 49.51s`。

## 已知限制

1. **没有相似度匹配**：tag 必须精确相等。v7 那套 N-gram/向量匹配没搬过来。
2. **单趟执行**：不满足 `result_tag` 就标 `incomplete`，不会自动重跑。
3. **pi 的 RPM 限制**：网关会返回 `429 inference exceeds tpm/rpm limit`，适配器做了 2/4/8 秒退避重试，但高并发跑多条边仍可能撞上。
