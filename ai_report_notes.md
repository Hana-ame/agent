# AI 日报示例（map 架构）改动记录

> 本文件记录 `vertex-edge-agent` 中 `s1_ai_report_map` / `hn_ai_report` 两个示例的
> 完整改动历史、架构约定、运行/消耗对比与回归测试，供后续维护参考。

---

## 1. 架构：MapEdge 自持 agent 的 pipeline

两个示例都采用 **MapEdge fan-in/fan-out** 模式：一个 LLM 步骤先筛选候选，再对每个候选用
一条 `pipeline`（fetch → summarize）并行处理。

### 1.1 数据流

```
s1:  v_start → e_fetch_threads(FetchThreadsEdge)
              → e_filter(FilterEdge, hy3-free)        选 AI 相关帖子
              → v_router → ProcessThreadsMap(pipeline: FetchEdge + SummarizeEdge, max_concurrency 5)
              → v_report(ReportVertex, vertex/report_hook.py)

hn:  v_start → e_fetch_stories(FetchTopStoriesEdge, top 30)
              → e_filter(FilterEdge, hy3-free)        选 AI 相关帖子
              → v_router → ProcessStoriesMap(pipeline: FetchCommentsEdge + SummarizeEdge, max_concurrency 5)
              → v_report(ReportVertex, vertex/report_hook.py)
```

### 1.2 MapEdge pipeline 步骤的 config 结构

```json
{
  "id": "e_process_stories",
  "source": "v_router",
  "destination": "v_report",
  "script": "hn_edges.py:ProcessStoriesMap",
  "settings": {
    "pipeline": [
      {
        "type": "fetch",
        "script": "hn_edges.py:FetchCommentsEdge",
        "settings": { "timeout": 30 }
      },
      {
        "type": "llm",
        "script": "hn_edges.py:SummarizeEdge",
        "settings": {
          "prompt": "…(中文,见下)…",
          "model": "hy3-free",
          "base_url": "https://opencode.ai/zen/v1/chat/completions",
          "proxy": "http://127.0.2.6:7890"
        }
      }
    ],
    "max_concurrency": 5
  }
}
```

### 1.3 硬性约定（踩坑后锁定的规则）

| 规则 | 说明 |
|---|---|
| **不要 `agent` 字段** | 框架已移除，edge 脚本在 `__init__` 自持 agent |
| **prompt/model/base_url/proxy 必须嵌在 `settings` 里** | 不允许顶层级 step 配置 |
| **base_url 必须是完整 URL**（含 `/chat/completions`） | 框架 `_endpoint_url()` 原样使用，绝不自动拼接 |
| **proxy 可 per-settings 指定** | `_client_for(settings)` 按 proxy 值缓存 `httpx.AsyncClient` |
| **fetch 可声明 timeout** | fetch 函数接受 `timeout` 参数，从 settings 读取，默认 30 |
| **filter 不限条数** | prompt 写「有多少选多少，不要限定数量」，由模型自主判断 |

---

## 2. 报告格式（s1 与 hn 一致）

### 2.1 结构化标题，非 LLM 复述

`SummarizeEdge` 在 `post_process` 返回：

```python
{"title": "...", "url": "...", "summary": "LLM 正文..."}
```

`title` / `url` 来自**抓取数据**（不消耗 LLM 输出 token 去复述标题）；
`summary` 是 LLM 生成的中文正文。`ReportVertex.on_receive` 渲染为：

```markdown
# [帖子标题](原帖链接)
【…小节…】
摘要正文

---
```

- **不再**使用 `## Thread N` / `## Story N` 这类序号标题。

### 2.2 中文 prompt（chatto-bot 风格）

两个示例的 summarize prompt 结构一致（s1 多一个「按时间排列」小节）：

```
你是 {S1/HN} 上的 {帖子总结员/AI 话题观察员}，把单帖讨论提炼成一份精炼的中文小结。
## 工作约定
1. 只输出正文：一段 markdown，按【小节】组织。
2. 短而有信息量：宁可压缩/精简，也不要堆砌长句和注水。
3. 忠于原文：只写帖子里实际出现的内容(模型/工具/链接、具名用户、有效论点)；不确定的不写,不编造。
4. 语言：用中文，技术名词可保留英文。
5. 不要重复标题和链接，不要额外说明。
```

s1 额外约定：
- 【AI/LLM 趋势】按时间顺序，每条以「8月29日上午9点」格式开头（来自回帖时间戳换算，不写年份/分钟，不编造时间）。
- 【用户观点】列出具名用户观点（数量不限定）。

**禁止**：机械限制（「最多 2 个用户」「最多 120 字」等）——用户明确反对注水但也反对死板截断。

---

## 3. 框架改动（本次需求依赖）

| 文件 | 改动 |
|---|---|
| `framework/graph.py` | MapEdge pipeline step 的 `script` 相对 **config 目录**解析（此前相对 CWD → "Script not found"） |
| `framework/utils/script_loader.py` | `load_class_from_script` 先 `getattr(module, cls_name)` 精确解析；不再靠字母序自动发现（此前 `s1_edges.py:SummarizeEdge` 会加载到 `FetchEdge`） |
| `framework/agents/_http_base.py` | `_endpoint_url(settings)` 原样使用 settings 里的完整 URL，不自动补 `/chat/completions`；`_client_for(settings)` 按 proxy 缓存客户端；`_post`/`stream_process` 线程化 settings |
| `framework/agents/_http_base.py` | 真实 token 捕获：`usage_log` + `get_usage_summary()`（reasoning/visible 拆分） |

### 3.1 真实 token 上报格式

```python
agent.get_usage_summary()
# {'calls': 6, 'prompt_tokens': 9163, 'completion_tokens': 16786,
#  'reasoning_tokens': 15034, 'visible_tokens': 1752, 'total_tokens': 25949}
```

`reasoning_tokens` 是模型思考 token；`visible_tokens = completion - reasoning` 是实际输出。

---

## 4. 运行方式

### 4.1 map（框架 pipeline）

```bash
env -u HTTPS_PROXY -u HTTP_PROXY python examples/hn_ai_report/demo.py
env -u HTTPS_PROXY -u HTTP_PROXY python examples/s1_ai_report_map/demo.py
```

- proxy 已内嵌在 config 的 settings 里，**无需**再设环境变量。
- `demo.py` 从 config 读取 base_url 构造 `HttpLLMAgent`；缺失则报错（无默认值/无自动填充）。
- 输出 `report.md` 到示例根目录。

### 4.2 直出（opencode agent 循环）

```bash
HTTPS_PROXY=http://127.0.1.6:7890 \
  opencode run --model opencode/hy3-free "$(cat /tmp/prompt.txt)" > examples/.../opencode_direct.md
```

- opencode CLI **只用 opencode 自家免费模型** `opencode/hy3-free`；
  sensenova 只能走 pi/框架，不能走 opencode。
- prompt 可预置 URL（同源对比）或让 agent 自主抓取 top 列表再选（从头跑）。

---

## 5. 性能对比（实测）

### 5.1 耗时

| 路径 | 模式 | 耗时 | LLM 调用 | 帖子数 |
|---|---|---|---|---|
| s1 | map（不限条数） | 321.5s* | 3 | 3 帖 |
| s1 | map（早期 4 条） | 83.5s | 4 | 3-4 帖 |
| s1 | 直出（同源 4 帖） | 418.2s | 1 agent 循环 | 4 帖 |
| hn | map（不限条数） | 101.6s | 6 | 5 帖 |
| hn | map（旧 5 条上限） | 93.8s | 6 | 5 帖 |
| hn | 直出（从头自主选） | 304.0s | 1 agent 循环 | 5 帖 |

\* s1 map 该次偏慢为免费池当次推理波动（日志无重试/代理错误）。

### 5.2 消耗量（map 最新 / 直出对应）

#### S1

| 指标 | map（不限条数） | 直出（opencode） |
|---|---|---|
| 总 tokens | **19,804** | **73,574** |
| input / prompt | 6,108 | 67,882 |
| output / completion | 13,696 | 5,692 |
| ├ reasoning（思考） | 12,633 | 6,354 |
| └ visible（实际输出） | 1,063 | 5,692 |

#### HN

| 指标 | map（不限条数） | 直出（opencode） |
|---|---|---|
| 总 tokens | **25,949** | **80,235** |
| input / prompt | 9,163 | 78,035 |
| output / completion | 16,786 | 2,200 |
| ├ reasoning（思考） | 15,034 | 2,545 |
| └ visible（实际输出） | 1,752 | 2,200 |

（直出数据读自 opencode SQLite `~/.local/share/opencode/opencode.db` 的 `session` 表。）

### 5.3 结论

- **直出 input 爆炸**：agent 循环 WebFetch 整页（HTML→文本，每帖数千 token），4-5 帖累计 67-78k。
- **map input 极省**：只送 24h 回帖 / 前 15 条评论，6-9k。
- **map reasoning 偏高**：每帖独立调用一次 LLM，每次思考 token 累积（s1 12.6k / hn 15k）；直出只思考一次。
- **总消耗 map 省 ~70%**：s1 map 是直出的 27%，hn map 是 32%。
- 若在意成本选 map；若在意「一次到位」的灵活性选直出。

---

## 6. 已知坑与修复（回归测试锁定）

| 坑 | 修复 | 测试 |
|---|---|---|
| `load_class_from_script` 按字母序返回第一个 Edge 子类（`FetchEdge`），导致 pipeline 步骤加载错类、`SummarizeEdge.post_process` 不执行 | 按显式类名 `getattr` 精确解析 | `tests/test_script_loader.py` |
| stage1st 解析：`div[id^="post_"]` 误匹配空 `post_rate_div_<pid>`；时间戳是中文「发表于 …」无 `span[title]` | selector 收紧为 `^post_\d+$`；strip 前缀 + `re.search` 解析 `YYYY-M-D H:M` | `tests/test_s1_edges.py`（含离线 fixture `tests/fixtures/s1_thread.html`） |
| 多页回帖顺序乱（`insert(0)` 反向遍历） | 收集 `(dt, …)` 排序升序 | `tests/test_s1_edges.py` |
| MapEdge pipeline script 相对 CWD 解析 → Script not found | 相对 config 目录解析 | — |
| 标题由 LLM 复述浪费 token | 结构化标题来自抓取数据 | `tests/test_s1_edges.py` |

- `pytest.importorskip("bs4")` 保护无 bs4 环境的运行。
- 当前 **335 tests passed**。

---

## 7. 相关提交（分支 vertex-edge-agent）

```
13b47a3 hn_ai_report: sync s1 improvements (Chinese prompt, structured title, report_hook # [title](url)); hn edges proxy/timeout; both filters unlimited count; new map+direct reports
91b73a1 feat: settings-level explicit proxy for LLM calls; hn_ai_report real run
6683edf refactor: explicit endpoint + fetch timeout in settings; no fallback fill
065ea5a drop ## Thread N + chronological timeline
90ea271 chatto-bot style prompt
9aab7c8 token capture + demo print
75df373 bs4 dep
f4f7e17 script_loader fix + title
6a4d4e1 parse bug + tests
4cfedec clone + MapEdge fix
```
