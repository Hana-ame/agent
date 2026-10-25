# v8 · JSON 配置 + pi agent 节点 + sqlite3

> 当前能力/证据/阻塞与恢复后的待办见 [STATUS.md](STATUS.md)。

v8 只做一件事：**按顺序调用节点**。节点默认是一次 `pi` agent 调用；调度器不认识
prompt/HTTP，只负责按 `layer` 取输入、调用节点、落库。

| 层 | 做什么 | 代码 |
|---|---|---|
| 配置 | 纯 JSON 描述图、节点与 prompt | `dag8/config.py` |
| 节点 | `PiNode`（pi agent）/ `GraphNode`（一张 v7 图，实验） | `dag8/nodes.py` |
| 调度 | 按 layer 取输入 → 调节点 → 落 sqlite | `dag8/scheduler.py` |
| 存储 | 内容、执行记录全落 sqlite3 | `dag8/store.py` |

调度器约 150 行（`dag8/scheduler.py`）：tag 精确匹配取输入，缺输入就跳过，出错就记 error 停下。

## 与 v7 的区别

| | v7 | v8 |
|---|---|---|
| 节点执行者 | Python 函数 / HTTP LLM 客户端 | **`pi` agent（子进程）**；可选把整张 v7 图或一段 shell 当节点 |
| 节点类型 | fn / LLM 边 | **script + pi + graph + command + http + db** |
| 内容存放 | 内存 + 输出 Markdown | **sqlite3**（可查询、可回溯） |
| prompt 定义 | 写死在 Python 里 | **配置 JSON 的一个字段** |
| 图定义 | Python 代码 | **JSON** |
| 依赖 | `requests` / `httpx` | 标准库 + `pi` 二进制 |

v7 的 `scripts/` 保留不动；v8 是独立目录，互不影响（只有 graph 节点会按需 import v7；command 节点只跑 shell）。

## 快速开始

```bash
cd v8
python -m dag8 run configs/demo.json --check-model  # 先验模型能不能调通（3 秒）
python -m dag8 run configs/demo.json            # 三边流水线：起草 → 挑错 → 定稿（走 ssh 到 WSL 的 pi）
python -m dag8 run configs/demo.local.json      # 同上，但代码和 pi 在同一台机器（本机直调，不 ssh）
python -m dag8 run configs/nested_demo.json     # pi 节点 → v7 图节点（graph 节点示例）
python -m dag8 run configs/nested_demo.json --workers 2  # 波次并行：互不依赖的节点同时跑
python -m dag8 run configs/nested_demo.json --workers 2 --rpm 3  # 免费网关建议 ≤2 并发 + RPM 限速
python -m dag8 run configs/demo.json --dry-run  # 不调 pi，只验配置与占位符渲染
python -m dag8 demo                             # 离线 demo：command 节点多波次并行（不调模型）
python -m dag8 lint configs/demo.json           # 静态检查配置（环/不可达输入/子图路径）
python -m dag8 run configs/demo.json --lint --dry-run  # 跑前先 lint，有 error 直接中止
python -m dag8 run configs/demo.json --json           # 机器可读输出（CI/脚本；预检信息走 stderr）
python -m dag8 convert v7.json --out /tmp/v8.json  # v7 图 → v8 配置（可波次并行）
python -m dag8 doctor                           # 健康检查：pi / provider / 网关
python -m dag8 cost                             # 按运行记录估算 tokens / 费用
python -m dag8 stats                            # 按模型统计成功率/时延/缓存/错误 Top
python -m dag8 models                           # 模型记分卡：价格 + 实测成功率/时延
python -m dag8 route                            # 按历史 stats+价格推荐模型
python -m dag8 diff RUN_A RUN_B                 # 对比两次运行（状态/时延/模型）
python -m dag8 explain RUN_ID                   # 解释失败/卡住，给可执行建议
python -m dag8 new --list                       # 看任务模板（summarize / extract_json / …）
python -m dag8 new summarize --out /tmp/summarize.json  # 从模板生成配置
python -m dag8 runs                             # 列出历史运行
python -m dag8 show <run_id>                    # 看某次运行的全部中间内容
```

**实际可用的是 `laptop` 这一家的 5 个**（`deepseek-v4.1-flash` 已删除）。
`sensenova` provider 已按 `laptop` 复制登记，但**未配 key**，调用会在 1 秒内报
`No API key found for sensenova.`。完整清单与实测数据见 [`MODELS.md`](MODELS.md)。

`--check-model` 存在的理由：`pi` 遇到未注册的模型名只打一行 Warning 就按 custom model id
透传，**退出码仍是 0**，模型名拼错时 v8 看不出异常。预检会在开跑前用一次真实调用把问题暴露出来。

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
    "thinking": "off",
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

### 节点类型

每条 edge 默认是 **pi 节点**（一次 `pi -p` 调用），只需要 `prompt`：

```json
{"id": "draft", "input_tags": ["task"], "output_tags": ["draft"], "prompt": "任务：{contents}"}
```

也可以把一张 **v7 图** 整体当节点（`kind: "graph"`，实验特性）：

```json
{
  "id": "tighten",
  "kind": "graph",
  "input_tags": ["points"],
  "output_tags": ["final"],
  "graph": {
    "path": "graphs/v7_tighten.json",
    "seed_tag": "draft",
    "result_tag": "tightened",
    "max_steps": 10
  }
}
```

- 节点输入（按 `input_tags` 收集）拼成种子 → 注入 v7 图的 `seed_tag`
- 跑 v7 DAG 直到 `result_tag` → 该内容就是节点输出
- v7 图的模型：图 JSON 内联 `models`，或用 `graph.models` 指向 models.json（后者优先）
- `path` 相对当前 v8 配置文件所在目录

也可以把一段 **确定性 shell 命令** 当节点（`kind: "command"`）。它不调模型，适合取 URL、抓字幕、落盘这类不该交给 LLM 的活：

```json
{
  "id": "fetch",
  "kind": "command",
  "input_tags": ["url"],
  "output_tags": ["source"],
  "command": "python3 fetch_bili.py",
  "stdin": true,
  "timeout": 300
}
```

- 命令里可以用 prompt 占位符：`{url}` / `{tag:source}` / `{contents}` / `{a,b}`。
- `stdin: true` 时把 `{contents}` 渲染后的文本从 stdin 喂给命令，适合长 Markdown；否则只从命令行参数/占位符取。
- 工作目录默认是配置文件所在目录，所以 `python3 fetch_bili.py` 这种相对路径可直接写。
- 非零退出码即节点失败，stderr 会进 sqlite 的 `edges.error`。
- dry-run 不执行命令，只渲染命令文本。

示例：[configs/troll_tea_party/dag.json](configs/troll_tea_party/dag.json)（B 站 URL → 元信息 + 字幕 → 6.8 初稿/定稿 → 落盘）。

也可以把一次 **确定性 HTTP 请求** 当节点（`kind: "http"`），不经过模型：

```json
{
  "id": "fetch",
  "kind": "http",
  "input_tags": ["task"],
  "output_tags": ["out"],
  "url": "https://api.example.com/items/{tag:task}",
  "method": "GET",
  "headers": {"Authorization": "Bearer {tag:token}"},
  "body": "{\"q\": \"{tag:task}\"}",
  "max_chars": 20000
}
```

- 支持 GET/POST/PUT/PATCH/DELETE/HEAD；url/body/headers 都支持占位符
- 只允许 http(s)；4xx/5xx 直接记节点失败，响应截断到 `max_chars`
- 实机：`GET https://example.com` → 577 字符，status ok（外网 HTTPS 可用）

`kind` 可省略：写了 `graph` 就是 graph 节点，写了 `command` 就是 command 节点，写了 `url` 就是 http 节点，否则是 pi 节点。未知 `kind` 直接报错。

### 可选输入

`optional_inputs` 里的 tag 缺失时，边照常运行（占位符渲染成空）；只有必需 tag
缺失才会跳过。适合「有批评就参考、没有就按草稿走」这类分支：

```json
{
  "input_tags": ["draft", "critique"],
  "optional_inputs": ["critique"],
  "prompt": "【草稿】{tag:draft}\n【批评】{tag:critique}"
}
```

### 模型 / provider 回退

`pi.fallbacks` 按顺序写备用模型；主模型遇到可重试错误（429 / 超时）时自动切换：

```json
"pi": {
  "provider": "laptop",
  "model": "deepseek-flash",
  "fallbacks": ["glm-5.2", "sensenova/deepseek-v4-flash"]
}
```

- 不带 provider 前缀 = 同一 provider；`sensenova/xxx` 会切到别的 provider（需配 key）
- provider 前缀路径有单元测试（fake agent 记录实际 provider）
- 主模型自己的 2/4/8s 重试先跑完，再切备用
- 回退成功后，边记录里的 `model` 是实际使用的模型
- **回退结果不写缓存**：缓存键按主模型算，写进去会污染下次主模型的命中

也可以做**只读 SQLite 查询**（`kind: "db"`）：

```json
{
  "id": "lookup",
  "kind": "db",
  "input_tags": ["task"],
  "output_tags": ["rows"],
  "db_path": "data.sqlite3",
  "query": "SELECT name FROM items WHERE id > ?",
  "params": ["{tag:task}"],
  "max_rows": 1000,
  "read_only": true
}
```

- 参数用 `?` 绑定（`params` 里可写占位符），不做字符串拼接
- 默认只读（`mode=ro`），写操作直接报错
- 输出 JSON：`{"columns": [...], "rows": [[...]], "truncated": false}`
- 超过 `max_rows` 截断并标 `truncated: true`

`kind` 可省略：写了 `graph` / `command` / `script` / `url` / `db_path` 就按对应类型；写了 `prompt` 才是 pi；**什么都没有时默认 script**。未知 `kind` 直接报错（审批不是节点，见下）。

### script 节点（默认）

```json
{
  "id": "fetch",
  "input_tags": ["url"],
  "output_tags": ["html"],
  "script": "fetch.py"
}
```

- 默认 `kind=script`；`.py` 用当前解释器，其他文件直接执行
- 输入：`DAG8_INPUTS`（JSON，tag → 文本）+ stdin 同一份 JSON
- 输出：stdout；非零退出码即失败，stderr 进 `edges.error`
- `command` 仍可用于一段 shell；二选一

### 人工闸门（edge 级 `approval`）

审批**不是节点**：它不产出内容，只决定这条边能不能执行。

```json
{
  "id": "publish",
  "input_tags": ["draft"],
  "output_tags": ["published"],
  "approval": {"marker": "APPROVED.flag"},
  "prompt": "发布：{contents}"
}
```

- 输入就绪后、节点执行**前**检查 marker（相对配置文件）
- marker 不存在 → 该边 `error`，提示“创建标记文件后 --resume 续跑”
- marker 内容 `{"approved": false, "reason": "..."}` → 显式拒绝
- marker 存在 → 正常执行这条边的 prompt（内容仍由节点产出）
- 配合 `--resume-last`：先跑到闸门，人工确认后只补这条边

### 条件边（when）

输入就绪后，按某个 input tag 的内容决定这条边是否执行：

```json
{
  "id": "review",
  "input_tags": ["diff"],
  "output_tags": ["review"],
  "when": {"tag": "diff", "pattern": "\\S"},
  "prompt": "评审：{contents}"
}
```

支持 `present` / `absent`（布尔）、`pattern`（正则）、`equals`（去空白后精确匹配）；
`when.tag` 必须在 `input_tags` 里，条件在输入就绪后求值。不满足时边记为
`skipped / when 条件不满足`，不影响其它分支。

### 质量闸门（judge）

输出通过契约后，可以再调一次评审模型打分；低于阈值判该边失败：

```json
{
  "id": "answer",
  "input_tags": ["draft"],
  "output_tags": ["final"],
  "prompt": "定稿：{contents}",
  "judge": {
    "min_score": 8,
    "prompt": "按准确性/完整性给下面内容打分 0-10，只输出 JSON：{\"score\":0,\"reason\":\"...\"}\n\n{output}",
    "model": "laptop/glm-5.2"
  }
}
```

- 评审默认 prompt 内置；`{output}` 可自定义插入点
- 分数 JSON 解析失败 / 越界 / 低于 `min_score` → 该边 `error`，reason 写进 error
- 评审调用走 pi（可用 `judge.model` 单独指定），会计入耗时但不计入 `--max-cost` 估算
- `--dry-run` 不执行评审
- 实机（2026-10-24）：`min_score=0` → ok（17.6s，含评审调用）；
  `min_score=10` → error `judge 分数 6 < 10：…（完整 reason）`

### 递归 v8 子图（engine="v8"）

graph 节点默认跑 v7 顺序子图；把 `engine` 设为 `v8`，它就会指向另一张 **v8 配置**：

```json
{
  "id": "review", "kind": "graph",
  "input_tags": ["changed"], "output_tags": ["review"],
  "graph": {"path": "graphs/review.json", "engine": "v8",
            "workers": 3, "timeout": 120}
}
```

- 子图内部按波次并行，共享父级 `--rpm` 限速器
- 子图用自己的临时 db，`seed_tag` / `result_tag` 由子图配置决定
- 适合把大图拆成多个可独立修改/并行的子图（非 DAG 的嵌套组合）
- `graph.seed_tag` / `graph.result_tag` 在 v8 模式下可省略
- 实机（2026-10-24）：内层 2 并行 pi + 1 合并，父节点 **15.6s** 完成，结果 80 字；
  嵌套调度共享父级 `--rpm` 限速器

### 输出契约

边可以声明输出必须满足的格式；校验失败会把原因回灌给 pi 节点重跑：

```json
{
  "id": "extract",
  "expect": {"type": "json", "required_keys": ["topics", "count"]},
  "repair": 1
}
```

| 字段 | 含义 |
|---|---|
| `type` | `json` / `text` |
| `required_keys` | JSON 顶层必须包含的键 |
| `pattern` | 正则（`re.search`） |
| `min_chars` / `max_chars` | 正文长度 |
| `min_sections` | Markdown 标题数下限 |
| `repair` | 失败后重试次数（原因回灌给 pi 节点） |

- 只做确定性检查，不调模型，适合当便宜的质量闸门
- `graph` / `command` 节点不支持 repair（失败直接记 error）
- dry-run 不校验（预览不是真实输出）
- 校验通过才写缓存，坏输出不会污染缓存；JSON 契约通过后会去掉 ``` 围栏并缩进规范化再落库

示例：[configs/nested_demo.json](configs/nested_demo.json)。实测：

```
run 35f215a75e30 · nested_demo · ok
  result: final = 65 字符
  ✓ [0] points   13.6s   161 字符
  ✓ [1] tighten   3.6s    65 字符
```

### pi 的两种执行后端

- `ssh_host` 为空：直接跑本机 `pi`
- `ssh_host` 有值：`ssh host 'export PATH=<nvm>; pi -p ...'`，prompt 走 stdin

WSL 有两个可用地址，都能到同一个系统：

- `172.29.89.192` —— eth0 地址（固定），示例配置用它，直连 WSL 内的 sshd
- `127.0.0.1` —— Windows 侧 `wslrelay` 转发，多一层转发

**两边环境各存一份配置**，跑之前按所在环境选：

| 文件 | 在哪跑 | pi 怎么到 |
|---|---|---|
| `configs/demo.json`、`configs/s1_digest.json` | 代码在 Windows | `ssh` 到 WSL 的 eth0 地址（`172.29.89.192`） |
| `configs/demo.local.json`、`configs/s1_digest.local.json` | 代码和 pi 在同一台机器（比如直接在 WSL 里跑） | 本机直调，`ssh_host` 省略 |

两份只差 `pi.ssh_host` / `pi.ssh_key` 两个字段，prompt 与边定义完全一致。

⚠️ CLI **不读** `DAG8_SSH_HOST` / `DAG8_LOCAL_PI`——它们只在 `tests/` 里生效（`_probe.py`、`test_live_pi.py`）。换环境是换配置文件，不是换环境变量。

prompt 用 stdin 传而不是命令行参数，避免多层 shell 的引号和换行问题（实测命令行方式在多行中文 prompt 上会被截断）。

## 调度形式：波次并行

调度器按**波次**推进：每一轮把所有「输入 tag 已就绪」的节点挑出来，按 `--workers`
并发执行，结果写回 sqlite 后再进入下一轮。

- `--workers 1`（默认）= 严格按 layer 串行，行为与旧版一致
- `--workers N>1` = 互不依赖的分支真正同时跑；依赖关系仍由 tag 决定，不会被破坏

```bash
python -m dag8 run configs/nested_demo.json --workers 2
```

节点内部不碰 sqlite：输入在主线程取好，线程里只做调用，输出由主线程统一落库。
所以并发不会写坏库，也不会改变结果。

`--max-seconds N` 给整次 run 一个墙钟上限：到点停止调度，剩余边标 skipped，
run 状态 `timeout`（配合节点级 timeout，避免一次 run 无限拖）。

`--max-cost N` 给整次 run 一个**估算成本**上限（按 providers.json 的 token 价格）：
超了停止调度，剩余边标 skipped，run 状态 `over_budget`；`run --json` 里有 `cost_estimate`。

`--auto-model`：开跑前按同一 db 的历史 stats + 价格自动选模型（覆盖 `pi.model`）；
score = ok_rate − p50/120 − price_weight×(price_in+price_out)。

`--keep-going`：某条边失败时不再立刻停掉整张图，继续跑其它不受影响的分支；
最终状态 `partial`（已有结果）或 `error`（无结果）。默认仍是失败即停。
实测：bad 契约失败 + good 成功 → `partial`，result=good 的输出，退出码 1。

pi / graph 的重试退避带 **0.5–1.5× 抖动**：随机 429 下避免多个重试同步撞车再次被限流。
限速 mock 基准（并发 4、4 次重试、3 轮）：fixed p50 **2.69s / 7×429**，
jitter p50 **1.65s / 6×429**（约 1.6× 更快）。

`--adaptive`：自适应并发（AIMD）。某波次出现 429 **或批量超时** → 下一波 worker 减半；
成功 → +1，夹在 `[1, workers]`。每次实际使用的并发记录在 `run --json` 的
`workers_used` 里（如 `[4, 2]`）。mock HTTP 集成测试覆盖了“2 个 429 + 2 个成功 → 下一波 2”的真实链路。

## 速度策略（实测，2026-10-24）

| 场景 | 建议 |
|---|---|
| 免费网关（本仓库 laptop-8000） | `--workers 1~2 --rpm 3~4`；宽并行会 429/超时 |
| 纯 v7 图迁移、要速度 | `convert --mode graph`（v7 直连客户端，比 pi 快） |
| 需要 agent 能力（工具/多步） | pi 节点；接受单次 10~60s 的延迟 |
| provider 能力未知 | `--adaptive --preflight`，让 AIMD 自动降并发 |
| 追求吞吐 | **优先换高 RPM provider / 加额度**，而不是加 workers |

实测依据：2 支路慢窗口 5 对配对中位 1.17×（0.95–2.62）；4 支路 graph 0.73×（2 对）；4 路 pi 5 对 v8 0 失败、v7 2/5 失败；6 支路直接 429；
单次调用延迟方差 7–67s 是主要噪声源。

## 节点缓存

节点结果按 `sha256(节点指纹 + 输入 + 模型)` 缓存，命中直接复用输出，不调 pi、不跑 v7：

```bash
python -m dag8 run configs/demo.json --cache             # 首次真跑，之后直接命中
python -m dag8 run configs/demo.json --cache --refresh   # 忽略旧条目，重跑并覆盖
```

- 默认关闭：生成任务通常希望每次有新结果
- key 含 provider/model：**换模型不丢旧结果**，换回来直接命中
- 缓存目录 `v8/data/cache/`（文件式，可手工删）；`--cache-dir` 可改
- 命中时边记录 `cached=1`，summary 显示 `(cache)`，时延记 0

## RPM 限速

`--rpm N` 让调度器在窗口（默认 60s）内最多发起 N 次 API 调用——只限 pi / graph 节点，
command 节点不计。波次并行时先排队再发，直接减少 429：

```bash
python -m dag8 run configs/nested_demo.json --workers 4 --rpm 3
```

实现是窗口内计数 + 阻塞等待（`dag8/ratelimit.py`），线程安全；`--rate-window` 可改窗口。

- **进程级共享**：同一 `rpm/window` 的所有 run、GA 评审共用一份配额（否则每次 run 新建限速器等于没有）
- **按实际 API 调用计数**：graph 节点内部的每次 v7 `complete` 都会先过限速器，不再按“节点”算
- 离线实测：`rpm=1, window=0.2s` 时两个并行节点会被错开；graph 节点的两次 v7 调用也会被错开

## 演化 + 人工评价

`evolve/` 是受约束的 DAG 演化闭环：`Genome → mutate → dag8.run → 自动检查 →
review.md/ratings.json → human/auto/combined 选择`。人工评价是一等公民：每个候选
有独立输出、评分位与备注；没有可信自动指标时用 `--by human` 选出候选。

```bash
cd v8
python3 -m evolve run --task evolve/tasks/troll_notes.json --population 3
python3 -m evolve rate --run data/human-eval/<dir> --candidate <id> --score 4
python3 -m evolve select --run data/human-eval/<dir> --by human
python3 -m evolve from-runs --db data/dag8.sqlite3 --last 12   # 不调模型
```

泄漏惩罚：给 `run` / `from-runs` 加 `--source` 与 `--answer` 后，候选 prompt/output
命中“仅答案拥有、源材料没有”的 ≥8 字片段时，自动分直接 ×0.1，并在 `review.md`
列出命中片段。详见 [`evolve/README.md`](evolve/README.md)。

## 调度搜索

`scripts/search_schedule.py` 用预算内 successive halving 搜索更优调度：

| 维度 | 取值 |
|---|---|
| 节点粒度/拓扑 | `single` / `chain2` / `chain3` / `fanout`（并行+合并）/ `v7_only`（v7 子图）/ `hybrid`（pi + v7） |
| 并发 | `--workers 1,3` 网格 |
| 模型 | `--models deepseek-v4-flash,glm-5.2,...`；端点/价格读 `configs/providers.json` |
| 质量 | 固定评审 prompt：覆盖 / 准确 / 简洁，各 1~5 分 |
| 时延 | 整次 `dag8.run` 的 wall time（评审不计入） |
| 预算 | `--max-runs`；每个组合默认跑 `--repeats 2` 次取中位数，top-2 补到 `--target 3` 次 |
| 去噪 | 单节点形式的 `workers>1` 自动跳过（并行无意义，避免把噪声当收益） |

```bash
cd v8 && python scripts/search_schedule.py --workers 1,3 --max-runs 18 --repeats 2 --target 3
python scripts/search_schedule.py --forms single,fanout --dry-run   # 只自检配置
```

产物：`v8/data/search/<时间戳>/report.md`（排名表 + Pareto 前沿）、`best/`（最佳配置）。

### 实测（2026-10-23，6 形式 × workers 1/3，16 次）

| 调度形式 | workers | 质量(3~15) | 时延 p50 | 效率 |
|---|---:|---:|---:|---:|
| v7_only | 1 | 12.67 | 8.42s | 1.50 |
| single | 1 | 15.00 | 13.61s | 1.10 |
| chain2 | 1 | 14.00 | 18.26s | 0.77 |
| hybrid | 1 | 14.00 | 22.04s | 0.64 |
| fanout | 3 | 14.50 | 44.40s | 0.33 |
| chain3 | 1 | 15.00 | 50.66s | 0.30 |
| fanout | 1 | 15.00 | 62.23s | 0.24 |

Pareto 前沿：`v7_only`（快）、`single`（质量高）。

结论（仅这次任务）：

- **质量优先选 `single`**：15 分 / 13.6s；`chain3` 同为 15 分但慢 3.7×——多两次 pi 调用没换来质量
- **效率优先选 `v7_only`**：8.4s，但评审指出字数常不足 150，质量 12.7
- **graph 节点（v7 子图）在第一轮里最快**（8.4s vs single 13.6s）；但第二轮同模型下
  `v7_only` 9.6s vs `single` 8.3s，优势没有复现——跨轮比较必须同模型同轮，别把噪声当结论
- **并行有用但救不了形式本身**：`fanout` 从 62.2s → 44.4s（workers 3），仍慢于单节点
- 波次并行会被网关 RPM 拖累，并发 >2 时 429 退避会吃掉收益

### 实测：模型维度（2026-10-23，single / v7_only，14 次）

| 形式 | 模型 | 质量 | 时延 p50 | tokens/次 | 效率 |
|---|---|---:|---:|---:|---:|
| single | **deepseek-flash** | 15.00 | **6.20s** | 184 | 2.42 |
| single | deepseek-v4-flash | 15.00 | 8.27s | 158 | 1.82 |
| v7_only | deepseek-v4-flash | 13.50 | 9.63s | 125 | 1.40 |
| v7_only | deepseek-flash | 13.50 | 11.01s | 131 | 1.23 |
| v7_only | glm-5.2 | 14.00 | 30.18s | 143 | 0.46 |
| single | glm-5.2 | 15.00 | 34.18s | 169 | 0.44 |

Pareto 唯一：`single + deepseek-flash`（15 分 / 6.2s）。

结论：

- **同质量下 `deepseek-flash` 比 `deepseek-v4-flash` 快 ~25%**——“v4-flash 最快”的旧假设在简单任务上不成立
- `glm-5.2` 质量没更高、慢 4~5×，不适合短文本节点
- `v7_only` 的分数被字数合规拖累（13.5），换模型也补不回来——问题在子图的 prompt，不在调度
- 本轮价格全为 0（免 key 网关）；`providers.json` 的 `price_in/price_out` 已接好，
  换成付费 provider 后报告里的“成本/次”会直接生效

## 遗传算法 + 基因账本

`scripts/evolve_schedule.py` 把调度形式编码成基因，做选择 / 交叉 / 变异，并把每次评估写进**跨轮累积**的账本：

| 基因 | 等位 |
|---|---|
| `topology` | single / chain2 / fanout / v7_only |
| `model` | deepseek-flash / deepseek-v4-flash |
| `prompt` | terse / strict（strict 追加 150~250 字硬约束） |
| `workers` | 1 / 3（只在 `fanout` 上合法；串行拓扑固定 1） |
| `contract` | off / on（on 时给 final 边加 150~400 字契约 + repair 1） |

适应度 = `quality - 0.1 × latency`。账本输出四类结论：

- **生效 / 失效 / 负向**：单基因边际
- **结合才生效**：2×2 因子中各自单独无效、组合增益 > 阈值
- **结合会失效**：各自单独有效、但 interaction 显著为负
- **相互放大**：各自单独有效、interaction 为正

```bash
cd v8
python scripts/evolve_schedule.py --population 6 --generations 3 --max-evals 18
python scripts/evolve_schedule.py --ledger-only   # 只重算账本，不跑模型
```

产物：`data/evolution/<时间戳>/report.md`、`best.json`、`data/evolution/ledger.json`。

> 教训：第一版账本把「单节点 + workers=3」的噪声算成“结合才生效”。现在 `workers`
> 只在 `fanout` 样本内统计，GA 也不允许串行拓扑选 3——结构性无效的基因不参与交互分析。

### 实测（2026-10-23，77 次评估 / 有效 65 / 失败 12）

最佳基因：`single/deepseek-flash/terse/w1` — 15 分 / **5.5s** / 183 tokens。

账本（`Δadj` = 控制其他基因后的 OLS 校正效应；交互用主效应残差）：

| 类别 | 条目 |
|---|---|
| 生效 | v7_only +1.18 / single +1.00 / strict +0.94 / v4-flash +0.91 / contract=on +0.88（n=20） |
| 失效 | contract=off −0.16 / terse −0.46 / deepseek-flash −0.35 |
| 负向 | fanout −3.14 / chain2 −0.98 |
| 结合才生效 | chain2+contract +3.19（n=12）/ chain2+strict +3.45 / v4+terse +3.38 / v7_only+contract-off +3.33 |
| 结合会失效 | v4+strict −3.38 / flash+terse −3.38 |
| 次可加 | v7_only+contract −3.33 / strict+contract −3.14 / v7_only+strict −2.97 |
| 相互放大 | v7_only+v4 +1.11 / single+strict +0.67 |

结论（扩样后修正）：

- `contract`：基线 9.46（off，n=17）→ 10.94（on，n=20），校正 **+0.88**
- `chain2`：基线 8.19（off，n=7）→ 10.81（on，n=12），交互 **+3.19**，组合绝对分 10.81，仍略低于总体基线 11.0（无 chain2 均值）：
  **缓解作用明显，但尚未反转为净收益**；+3.51 → +3.24 → +3.19，已收敛
- `single/v7_only + contract` 都是负交互：契约为本来格式稳定的节点增加了重试/开销
- `flash+terse` 仍是最佳组合，但交互为负（增益冗余）

数据质量修正（这轮踩到的）：

- 失败重试一次；仍失败标 `failed`，不参与账本（第一版把一次 error 当 0 分）
- 时延 >120s 按 120s 计入适应度（一次 265s 的限流重试会污染排名）
- 交互分析改用主效应残差（否则 `single` 和 `chain2` 混在同一格，误判组合）
- 加 OLS 校正效应（`Δadj`），原始边际 `Δraw` 并列显示

### 本轮增量（round 1）

- **可选输入**：`optional_inputs` 缺失时不阻塞（治「等不该等的 task」）
- **契约 A/B**：`contract=on/off` 跑 single/v7_only 各 2 次；single 质量 15/15 平手，
  v7_only 因 429 失败——契约本身不改变 single 质量
- **429 重试**：GraphNode 在限流时重建子图重试（10s/30s）；实测一次 v7_only 成功（47s）
- 账本失败样本从 2 增到 6（全是 v7 的 429，重试修复后应下降）

### 本轮增量（round 3）

- **RPM 限速**：`--rpm N` 在调度层按窗口节流 API 节点（pi/graph；command 不计）
- **pi 超时改为总预算**：`timeout` 现在含重试，不再是最坏 `timeout × retries`；
  实测旧实现出现过一次 **637.5s** 的 chain2 节点，新实现总预算 240s 封顶
- GA 用 `--rpm 2` 跑 7 次：本轮无新增失败（样本小，只能算弱证据）
- 账本扩到 53 次（47 有效）

### 本轮增量（round 4）

- **限速器改为进程级共享**：同一 rpm/window 的所有 run + judge 共用配额；
  graph 节点按内部 v7 调用逐次计数（不再按节点）
- **契约 × 拓扑扩样**：v7_only 3+3 次、chain2 2+2 次；v7 因 429 大量失败，
  修复后补跑一次成功
- 账本 64 次（53 有效 / 11 失败）；contract 校正效应 +0.66（n=10）

### 本轮增量（round 5）

- **运行级总预算** `--max-seconds`：到点停止调度，剩余边标 skipped，run 状态 `timeout`
- 网关短暂恢复（doctor 4.3s 通过），补了 `chain2+contract` 2 次（15/42s、15/64s；
  另一次 240s 超时被 pi 总预算按预期截断）
- 账本 70 次（58 有效 / 12 失败）；contract +0.56（n=13）；chain2+contract +3.01（n=5，缓解而非反转）

## 离线 demo

```bash
python -m dag8 demo
```

```
run ... · offline_demo · ok
  ✓ [0] left  0.0s  1 字符
  ✓ [0] right 0.0s  1 字符
  ✓ [1] merge 0.0s  3 字符
result: A|B
```

三个 command 节点：`left` / `right` 同波次并行 → `merge` 汇总，带长度契约。
**不调用任何模型**，适合 onboarding、网关故障时回归、CI 冒烟。

## 从项目结构生成图

`python -m dag8 scaffold ROOT --out config.json` 扫描模块（含 `__init__.py` 的目录；
一个都没有就退回顶层目录），每个模块生成一个 `command` 节点，最后是一个 pi 汇总节点：

```bash
cd v8
python -m dag8 scaffold . --mode dir --out /tmp/arch_v8.json
python -m dag8 scaffold . --changed --out /tmp/arch_changed.json   # 只建本轮 git 变更的模块
python -m dag8 scaffold . --changed --review --out /tmp/review.json  # 变更模块 → diff → 汇总评审
python -m dag8 scaffold . --compose --out /tmp/project.json  # 每模块 v8 子图 + 父图并行组合
python -m dag8 scaffold . --compose --review --out /tmp/rev.json  # 每模块评审子图 + 汇总
python -m dag8 run /tmp/arch_v8.json --workers 2 --rpm 3
```

- 模块节点跑 `scripts/inspect_module.py`：文件数 / LOC / 定义数 / imports / 内容 hash
- 每个模块独立重跑；`cache: false`（扫描当前文件是易变的，不能吃缓存）
- 结构直接跟随目录树：改哪个模块只跑哪个节点
- `--changed`：只给 `git diff HEAD` + 未跟踪文件涉及的顶层模块建图（改哪跑哪）
- `--review`：每个模块一个 `git diff` command 节点（截断 20k 字符），最后一个 pi 节点汇总评审
- `--compose`：每个模块生成一张 v8 子图（写到 `graphs/`），父图用 `graph(engine="v8")` 并行组合
- `--compose --review`：每个模块生成「diff + 评审」子图，父图汇总成整体评审
- `expect` 保证模块输出是 JSON；汇总节点把各模块指纹写成架构报告

实测（本仓库）：4 个模块并行各 0.1~0.2s，汇总节点 50.1s，输出 2843 字报告。

## v7 图 → v8 配置

```bash
python -m dag8 convert v7_graph.json --out /tmp/v8.json
python -m dag8 convert v7_graph.json --mode graph --out /tmp/v8g.json  # 每条边一张单边 v7 子图（快）
python -m dag8 lint /tmp/v8.json && python -m dag8 run /tmp/v8.json --workers 2 --rpm 3
```

- 保留 `input_tags` / `output_tags` / `layer` / `prompt`
- v7 的 `processor.model` 若在 models.json 里，映射成 `provider/model` 覆盖
- v7 的 `contents` 第一条变成 v8 seed（tag + content）
- 转换后 v8 的波次调度可并行无依赖的边；v7 引擎是单步串行
- 结果可直接 `lint` / `run`，也能继续加契约、缓存、可选输入
- `--mode graph`：每条边变成一张**单边 v7 子图**，用 v7 直连客户端 + v8 波次并行；
  适合 v7 图迁移后要速度的场景（见下）

实测（2026-10-24，网关恢复但抖动；同一 fan-out，交错 A/B）：

| 场景 | v7 串行 | v8 mode=graph | 结论 |
|---|---:|---:|---|
| 2 支路（5 对，慢窗口） | 141.5s（p50） | 116.6s（p50） | 配对中位 **1.17×**，波动 0.95–2.62 |
| 4 支路（5 对，mode pi） | 203.9s（p50），**2/5 失败** | 131.4s（p50），0 失败 | 有效配对 3 组中位 2.82×；v7 失败让比值偏乐观 |
| 2 支路（早期） | 21.0s | 22.6s | 0.95×，同样打平 |
| 2 支路（健康窗口） | 71.7s | 22.9/25.9s | 曾见 ~3×，不可复现 |
| 4 支路（2 对） | 55.8s | 77.7s | 0.73×；单支 67.1s 慢调用主导 |
| 6 支路 workers=6 | 78.2s | 51.9s（有边 429） | 并行触发限流，结果无效 |
| 6 支路 rpm=4/5 | 78.2s | 120–188s（多条 429） | 加限速更慢 |

结论（重要修正）：

- **瓶颈是 provider 的 RPM/并发上限 + 单次调用方差**（实测 7–67s），不是调度器
- 宽并行仍会 429；graph 模式 4 支路常被单支慢调用拖死；pi 模式 4 路 v8 0 失败，但 p50 131s 不低
- 并行收益只有在“分支多 + 单次延迟稳定 + provider 允许高并发”时才成立；
  当前免费网关三个条件都不满足
- `--mode graph` 的价值是**让 v8 用 v7 直连客户端**（比 pi 快），不是无条件加速
- 免费网关的并发探针：1/2/3/4/6 并发在冷却期全部 429；等 90s 后 n=1 仍 429、n=2 仅 1/2 成功
  —— **没有稳定阈值，429 近似随机**；`--rpm` 防主动打爆，但救不了随机 429
- 推荐：免费网关 `workers≤2 + rpm 3–4` 或纯 v7 串行；高 RPM provider 才值得宽并行；
  随机 429 靠退避重试 + `pi.fallbacks` / 换 provider

pi 开销基准（`scripts/bench_pi.py`，预热后 3 次）：
- pi 进程启动（`--version`）：p50 **0.28s**，可忽略
- pi 端到端 trivial 调用：p50 **11.4s**；v7 直连：p50 **9.2s**（各有 1 次失败）
- pi 相对 v7 的固定开销约 **+2.2s**；之前 final 的 87.8s 是瞬时抖动/重试，不是进程启动

## 任务模板

反复出现的任务不必每次重写配置：

```bash
python -m dag8 new --list
python -m dag8 new summarize --out /tmp/summarize.json --seed "长文……"
python -m dag8 new extract_json --out /tmp/extract.json
```

| 模板 | 输入 → 输出 |
|---|---|
| `summarize` | 长文 → 3~5 条要点 |
| `bug_triage` | 缺陷描述 → JSON(severity/component/summary/next_step) |
| `test_plan` | 需求/代码 → 测试计划（Markdown，≥3 节） |
| `refactor_plan` | 代码 → 重构计划（Markdown，≥3 节） |
| `extract_json` | 文本 → JSON（entities/count） |
| `classify` | 文本 → JSON(label/confidence) |
| `translate` | 文本 → 英文译文 |
| `review_diff` | 仓库路径 → `git diff` → 代码评审（command + pi） |

模板只是起点：生成的 JSON 可以继续改 prompt、加边、换成 graph 节点；
自带输出契约与失败重试。

实机（2026-10-24）：`summarize` 产出 89 字要点；`bug_triage` 输出严格 JSON，
`severity=high`，四个必需键齐全（schema 契约通过）。

## 失败诊断

```bash
python -m dag8 explain RUN_ID
python -m dag8 --db data/dag8.sqlite3 explain RUN_ID --json
```

```
run 8f3a… · demo · error
result: out (缺失)
✓ e1                      ok         5.0s
✗ e2                      error     10.0s  429 Too Many Requests

建议：
  - e2: 限流：加 --rpm；配置 pi.fallbacks；graph 子图会放大调用数，先降并发
  - 可续跑：python -m dag8 --db <db> run <config> --resume 8f3a…
```

把 429 / 超时 / 契约失败 / 缺输入翻译成下一步动作，不调网关。

## 运行对比

```bash
python -m dag8 diff RUN_A RUN_B
python -m dag8 --db data/dag8.sqlite3 diff RUN_A RUN_B --json
```

```
A 8f3a…  ok  15.3s  result 320 chars
B 91c2…  ok  10.9s  result 305 chars

edge                 A          A(s)   B          B(s)   Δ(s)
------------------------------------------------------------------
e1                   ok          5.2   ok          4.9   -0.3
e2                   error      10.1   ok          6.0   -4.1
```

对比每条边的状态、时延、Δ、模型，以及 run 状态/总时延/结果长度——适合
换模型、v7 vs v8、调并发/缓存前后的 A/B。

## 模型记分卡

```bash
python -m dag8 models
python -m dag8 models --providers configs/providers.json --json
```

```
模型                                  in/1M    out/1M    n    ok%      p50      p90
----------------------------------------------------------------------------------
laptop/deepseek-flash                   0.00      0.00   17    100%    10.9s    26.1s
laptop/deepseek-v4-flash                0.00      0.00   17    100%    10.9s    26.1s
laptop/glm-5.2                          0.00      0.00    0      -        -        -
```

- 价格来自 `configs/providers.json`（每百万 token）
- 成功率/时延来自历史运行（sqlite）
- 没调用过的模型也会列出（n=0），方便判断“贵但稳/便宜但慢”
- 换 paid provider 时填价格即可直接比较

## 数据库维护

```bash
python -m dag8 prune --keep 20          # 预览：将删多少个旧 run
python -m dag8 prune --keep 20 --apply  # 真正删除（含 contents/edges）
```

- 按 `started_at` 保留最近 N 个 run，其余连同 contents/edges 一起删
- 默认 dry-run，避免误删
- 与 `dag8 cache` 一起控制磁盘占用

## 自动模型路由

```bash
python -m dag8 route                 # 看推荐
python -m dag8 run cfg.json --auto-model
```

- 只用**有调用记录**的模型；`route --min-calls` 可过滤小样本
- `--auto-model` 要求 **≥3 次**未缓存调用，避免 n=1 的偶然快样本压过稳定模型
- 打分：`ok_rate − p50/120 − price_weight × (price_in + price_out)`
- 实机：主库历史下自动选到 `laptop/deepseek-v4-flash`（100%，p50 12s），而非 n=1 的 glm
- `route --json` 输出候选与分数；`--auto-model` 在 run 开始时覆盖 `pi.model`
- 边界：edge 级显式 `model` 仍然优先；跨 provider 需要对应 key

## 库式调用：脚本自己 step

`dag8.run()` 一次跑完整张图；如果你想让脚本接管“卡住”的步骤，用 `Dag`：

```python
from dag8 import Dag

d = Dag("configs/demo.json", seed_content="用三句话解释标签驱动 DAG")

while True:
    r = d.step()
    if r.status == "blocked":
        print("卡住：", r.blocked)        # [{"edge": "...", "missing": ["tag"]}]
        print("内容池：", d.contents)      # {tag: [content, ...]}
        d.set("external_result", "脚本在外面算好的结果")
    elif r.status in ("done", "error", "incomplete"):
        break

print(d.result)
d.close()
```

- `step()` 每次只执行**一条**已就绪的边；返回 `ran / blocked / done / error / incomplete`
- `contents` 是当前内容池（tag → 内容列表），卡住时从这里看缺什么
- `blocked` 给出缺输入的边和 tag；人工审批未通过时给 `reason`
- `set(tag, content)` 把外部处理结果写回内容池，再 `step()` 继续
- 需要缓存/自适应并发时仍用 `dag8.run()`；会话模式一条一条走，不复用缓存
- 节点侧就是 **script**：`kind: "command"` 跑脚本，`pi` 只是内置的 LLM 节点

## 断点续跑

```bash
python -m dag8 --db data/dag8.sqlite3 run cfg.json --resume RUN_ID
```

- 跳过已经成功的边，只补未完成/失败的边（按 `edge id` 识别）
- 需要同一个 `--db`；地址不同就找不到 run
- 失败边在旧尝试里的记录会清掉，避免混进新一轮
- 已全部完成时直接返回结果，不再调用 pi
- `--resume-last`：自动找**同一配置**最近一个未成功的 run 续跑（崩溃/中断后不用记 run id）
- `dag8 abort RUN_ID`：把卡在 `running` 的 run 标为 `interrupted`（Ctrl-C/崩溃后），之后可 `--resume-last`
- 实测：链式图失败一次后 resume，只重跑失败的那条边（1 次调用）
- 实测（2026-10-24）：修掉 bad 边契约后 `--resume-last` → 同一 run_id，只重跑 bad（3.3s），
  good 保留（2.9s），status ok

## 运行统计

```bash
python -m dag8 stats                 # 最近 50 次运行
python -m dag8 --db data/dag8.sqlite3 stats --json
```

```
运行：12 次  {'ok': 9, 'error': 3}  时延 p50 12.3s

模型                      n   ok%   p50    p90   缓存
------------------------------------------------------------
laptop/deepseek-flash    20   85%   8.2s  45.1s    3

错误 Top：
  5× 429 Too Many Requests
```

按模型看成功率、p50/p90 时延、缓存命中与错误分布——用来决定模型/并发策略，
不调网关。时延只统计**未缓存且 >0.05s** 的调用（缓存命中与 dry-run 的微秒不计入），
避免 0s 把 p50 拉平。

## 配置检查

```bash
python -m dag8 lint configs/demo.json      # 静态检查，不跑图
python -m dag8 lint cfg.json --strict      # 警告也算失败
```

- 重复 edge id、`result_tag` 无生产者、输入 tag 永远不可达（`optional_inputs` 除外）
- graph 节点引用的子图文件不存在
- 多生产者、producer layer 不低于 consumer（警告）
- 边依赖成环（会互相等待）
- 退出码：error → 1；`--strict` 时 warning 也 → 1
- `run --lint`：跑之前先检查，有 error 就不烧额度

## 离线自检

```bash
bash scripts/smoke_offline.sh
```

不依赖网关，依次验证：

- 离线 pytest（跳过 live）
- `doctor --skip-gateway`（pi / provider）
- 8 个模板逐个生成 + `--dry-run`
- `scaffold --compose` + `lint` + `--dry-run`
- `convert` v7→v8 + `lint`
- README 测试计数与 pytest 收集一致性（`scripts/check_docs.py`）
- 账本数字 / 测试总数 / 相对链接声明一致性（`scripts/check_claims.py`）

网关 530 挂掉时，用它确认“框架本身没坏”，而不是把环境故障当成代码问题。

## 健康检查

```bash
python -m dag8 doctor                  # pi 二进制 + provider 注册表 + 网关真实调用
python -m dag8 doctor --skip-gateway   # 只查本地
python -m dag8 doctor --json           # 机器可读
```

- 原始 HTTP 探测（`/v1/models`，不经 pi）：530/代理故障会直接标成“源站问题”
- pi 调用探测：限流/超时 → `--rpm` + `pi.fallbacks`；缺 pi → `pi.ssh_host`
- 返回码：全绿 0，否则 1，适合定时轮次/CI 先跑
- 实测（网关故障时）：`✓ pi`、`✓ provider`、`✗ 网关 … 超时` → 退出码 1

## 成本估算

```bash
python -m dag8 cost                 # 最近 20 次运行
python -m dag8 --db data/dag8.sqlite3 cost --json
```

- 按运行记录统计 tokens（字符 / 2 估算；pi 不返回 usage，结果会标注）
- 价格来自 `configs/providers.json` 的 `price_in` / `price_out`（每百万 token；0 = 免费/未配）
- graph / command 节点的 model 不是 `provider/model` 时会标 `unknown_models`，不计价
- 换 paid provider 时只需填价格，报告直接给出每次 run 的费用与合计

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
cd v8 && python -m pytest tests/ -q                  # 279 条，含 4 条真实调用，约 3–5 分钟
cd v8 && python -m pytest tests/ -q -m "not slow"    # 跳过整图那条，本机实测约 2–3 分钟
cd v8 && python -m pytest tests/ -q -m "not live"    # 完全离线，约 15 秒
```

| 文件 | 条数 | 是否碰 pi |
|---|---|---|
| `tests/test_approval.py` | 5 | 否 —— 无 marker 阻断 / 放行 / 拒绝 / resume / 不是节点 |
| `tests/test_db_node.py` | 4 | 否 —— 本地 SQLite：查询/截断/只读/配置 |
| `tests/test_http_node.py` | 5 | 否 —— 本地 server：GET/POST/错误码/URL 校验 |
| `tests/test_review_fixes.py` | 3 | 否 —— 动态 PATH/排序稳定性/tag SQL 过滤 |
| `tests/test_report_check.py` | 3 | 否 —— 汇报指标/配置 lint/dry-run |
| `tests/test_router.py` | 4 | 否 —— 成功率/时延/价格权重、CLI |
| `tests/test_cli_help.py` | 1 | 否 —— 所有子命令 --help 可退出 |
| `tests/test_adaptive_http.py` | 1 | 否 —— mock 429 下 AIMD 降并发 |
| `tests/test_dag8.py` | 77 | 否 —— 配置加载、占位符渲染、sqlite 读写、调度（注入 fake agent）、graph 节点、command 节点 |
| `tests/test_explain.py` | 6 | 否 —— 429/缺输入/续跑建议、CLI |
| `tests/test_diff.py` | 4 | 否 —— 边状态/时延/Δ、缺失边、CLI |
| `tests/test_http_retry.py` | 1 | 否 —— 本地 mock HTTP server 验证 429 重试链路 |
| `tests/test_models.py` | 4 | 否 —— 价格×实测合并、unknown 模型、CLI |
| `tests/test_resume.py` | 6 | 否 —— 只补失败边/全完成不调用/未知 run |
| `tests/test_stats.py` | 5 | 否 —— 模型成功率/时延/错误 Top、CLI |
| `tests/test_cache.py` | 8 | 否 —— TTL 过期/老条目/调度器命中 |
| `tests/test_contracts.py` | 18 | 否 —— JSON/正则/长度/小节契约与非法配置 |
| `tests/test_pi_timeout.py` | 2 | 否 —— pi 总预算超时、429 快速重试 |
| `tests/test_ratelimit.py` | 6 | 否 —— 限速器窗口、禁用、调度器节流 |
| `tests/test_convert.py` | 6 | 否 —— v7→v8 映射、seed/result、CLI |
| `tests/test_cost.py` | 6 | 否 —— tokens/价格估算、unknown 模型、CLI |
| `tests/test_demo.py` | 2 | 否 —— 离线 command 多波次 |
| `tests/test_abort.py` | 3 | 否 —— running→interrupted、未知 run、explain 提示 |
| `tests/test_prune.py` | 3 | 否 —— dry-run/apply/CLI |
| `tests/test_show.py` | 3 | 否 —— show/runs 的 JSON 输出 |
| `tests/test_lint.py` | 21 | 否 —— 环/不可达输入/结果缺失/子图路径/CLI 退出码 |
| `tests/test_doctor.py` | 10 | 否 —— 探活报告、建议、CLI 退出码 |
| `tests/test_templates.py` | 7 | 否 —— 模板构建、契约、dry-run、CLI |
| `tests/test_scaffold.py` | 15 | 否 —— 模块扫描、配置生成、dry-run、指纹 hash |
| `tests/test_search.py` | 4 | 否 —— 搜索候选生成、并发宽度判定、Pareto/报告渲染 |
| `tests/test_backoff.py` | 3 | 否 —— 抖动边界/上限/零基数 |
| `tests/test_evolve_leakage.py` | 5 | 否 —— 评测泄漏检查（离线） |
| `tests/test_evolve_human.py` | 6 | 否 —— 人工评价闭环（离线） |
| `tests/test_script_node.py` | 4 | 否 —— 默认 script/JSON 输入/非零退出/缺文件 |
| `tests/test_session.py` | 3 | 否 —— step/blocked/set 注入 |
| `tests/test_evolve.py` | 11 | 否 —— GA 基因/解码、2×2 交互分桶、结构性无效过滤 |
| `tests/test_live_pi.py` | 4 | **是** —— 真实调用 `laptop/deepseek-v4-flash` |

live 的四条：

1. 单次直调：返回「收到」，耗时 < 120s
2. provider 写错时抛 `PiError`，而不是静默返回空串
3. 边级：调度器取输入 → pi 执行 → 结果落 sqlite（校验 contents 条数、edge 耗时 > 0、run 状态 `ok`）
4. 整图三边（标 `slow`）：起草 → 挑错 → 定稿，校验多输入边的 `prompt_chars`

gating：`tests/_probe.py` 先查本机 `command -v pi`（**不调模型**），没有再探测
`ssh $DAG8_SSH_HOST command -v pi`。探不到时**直接 fail**，不是 skip ——
skip 会让「环境没准备好」和「全部通过」长得一模一样，那是假绿。

- 本机有 pi：live 用例自动走本地，不需要设环境变量
- 只有 WSL/远端有 pi：设 `DAG8_SSH_HOST` / `DAG8_SSH_KEY`
- 强制：`DAG8_LOCAL_PI=1`（本地）/ `DAG8_LOCAL_PI=0`（走 ssh）

实测（2026-10-24，本机 pi + laptop 网关）：

| 套件 | 结果 | 耗时 |
|---|---|---|
| 离线（`-m "not live"` / `--ignore=tests/test_live_pi.py`） | **275 passed** | 约 15s |
| live 非 slow（3 条） | **3 passed** | 167s |
| live slow（整图三边） | **1 passed** | 113s |
| 全量 | **279 passed**（含 4 live） | 约 5 分钟 |

注意：live 用例会真实碰网关，网关 530/429/变慢时**会红**——这是有意的，不是 flaky skip。
不想碰网关就用 `-m "not live"`。

### 在 WSL 本地跑（不走 ssh）

pi 装在 WSL 时，把代码放进 WSL 直接跑最快——省掉每条边一次 ssh 往返：

```bash
# 在 WSL 里
cd /mnt/c/Users/lumin/Downloads/agent/v8
python3 -m pytest tests/ -q          # 本机有 pi，会自动走本地
```

本机有 pi 时已无需 `DAG8_LOCAL_PI=1`；只有想强制走 ssh 时才设 `DAG8_LOCAL_PI=0`。
历史实测（旧网关、旧版本）全量约 95–124s；2026-10-24 网关变慢时 live 约 280s。

`dag8.pi.find_pi()` 会先查 PATH，再翻 `~/.nvm/versions/node/*/bin/pi` —— ssh 非交互 shell
不加载 `.bashrc`，不翻 nvm 目录的话 `which pi` 会落空。

## 已知限制

1. **没有相似度匹配**：tag 必须精确相等。v7 那套 N-gram/向量匹配没搬过来。
2. **单趟执行**：不满足 `result_tag` 就标 `incomplete`，不会自动重跑。
3. **429 限流**：pi 适配器做 2/4/8 秒退避（`timeout` 是总预算，含重试）；graph 节点重建子图重试（10s/30s）；调度器还有 `--rpm` 提前节流。
4. **graph 节点是实验特性**：v7 子图内部步骤不落 v8 的 sqlite（只存节点最终输出）；且需要 v7 的 `dag` 包在 `sys.path`。子图有总预算 `graph.timeout`（默认 300s），重试也计入。
5. **波次并行受网关 RPM 限制**：并发 >2 时容易触发 429（PiAgent 有 2/4/8s 退避），实际收益取决于限流强度。
