# AI 日报示例（map 架构）改动记录

> 本文档按「问题 / 方案 / 修改 / 测试」记录 `s1_ai_report_map` / `hn_ai_report` 两个
> 示例的历次问题与处置。所有 token/耗时数据均为实机运行结果（测试结果）。

---

## 问题 1：MapEdge pipeline 步骤的 `script` 相对路径解析错误

### 问题
MapEdge 的 `settings.pipeline[].script`（如 `hn_edges.py:FetchCommentsEdge`）按 **CWD**
解析，从项目根运行时 `Script not found`。

### 方案
pipeline step 的 `script` 统一按 **config 文件所在目录** 做 `base_dir` 归一化。

### 修改
- `framework/graph.py`（`from_dict`）：对 `settings.pipeline` 逐 step 执行
  `os.path.join(base_dir, step_script)`（仅相对路径）。

### 测试
**测试方案**：任意 CWD 下 pipeline step 都能加载。**测试方法**：
`cd / && python /…/examples/hn_ai_report/demo.py`（config 内嵌 proxy）。**测试结果**：
报告成功生成（`report.md` ~99 行），无 `Script not found`。

---

## 问题 2：`load_class_from_script` 自动发现按字母序选错子类

### 问题
`load_class_from_script("s1_edges.py:SummarizeEdge", ...)` 曾靠字母序自动发现，
`FetchEdge` 排在 `SummarizeEdge` 前 → pipeline 步骤加载了错误类，`post_process` 不执行，
报告缺少总结。

### 方案
有显式类名时优先 `getattr(module, cls_name)` 精确解析；找不到才降级自动发现并打 warning。

### 修改
- `framework/utils/script_loader.py`：显式类名路径（`load_class_from_script` 的
  `default_class` 为 str 时精确取类）。

### 测试
**测试方案**：多子类文件按显式类名加载正确类。**测试方法**：
`load_class_from_script("s1_edges.py:SummarizeEdge", Edge, "SummarizeEdge")`。**测试结果**：
返回 `SummarizeEdge` 而非 `FetchEdge`（`tests/test_script_loader.py` 回归锁定）。

---

## 问题 3：stage1st 解析 bug（中文时间戳、空占位 div）

### 问题
- `div[id^="post_"]` 误匹配空 `post_rate_div_<pid>`；
- 时间戳是中文「发表于 …」，无 `span[title]` → `dt=None` → 帖子被 24h 过滤 → 全部
  `0 replies`，报告空洞。

### 方案
- selector 收紧为 `^post_\d+$`；
- 时间戳 strip 前缀后用 `re.search` 解析 `YYYY-M-D H:M`；
- 多页回帖按 `(dt, …)` 升序排序（旧 `insert(0)` 反向遍历顺序错乱）。

### 修改
- `examples/s1_ai_report/s1_edges.py`、`examples/s1_ai_report_map/s1_edges.py`；
- `tests/fixtures/s1_thread.html` 离线 fixture。

### 测试
**测试方案**：真实页面离线解析正确（4 帖 21/81/13/5 条）。**测试方法**：
`pytest tests/test_s1_edges.py -q`（含 map 版）。**测试结果**：通过；实机跑出
`report.md` 137 行，与 opencode 直出版同级。

---

## 问题 4：标题由 LLM 复述浪费 token

### 问题
`SummarizeEdge` 让 LLM 复述标题，浪费输出 token 且可能失真。

### 方案
结构化标题：`title`/`url` 来自**抓取数据**（非 LLM）；LLM 只生成 `summary` 正文。
`ReportVertex.on_receive` 渲染为 `# [帖子标题](原帖链接)` + 小节正文。

### 修改
- `SummarizeEdge.post_process` 返回 `{"title","url","summary"}`；`ReportVertex` 渲染。
- 不再使用 `## Thread N` / `## Story N` 序号标题。

### 测试
**测试方案**：报告标题来自数据而非 LLM 复述。**测试方法**：`tests/test_s1_edges.py`
回归 + 实机报告人工核对。**测试结果**：报告首行 `# [标题](链接)`，无 `## Thread N`。

---

## 问题 5：中文 prompt（chatto-bot 风格）与「不限条数」

### 问题
旧 prompt 限定「最多 N 帖/最多 120 字」——机械截断被用户反对；且报告偏英文注水。

### 方案
统一中文 prompt（s1 多一个「按时间排列」小节），写「有多少选多少，不要限定数量」，
由模型自主判断。

### 修改
- `s1_edges.py` / `hn_edges.py` 的 filter/summarize prompt 中文化；filter 不限条数。

### 测试
**测试方案**：filter 输出数量不受硬编码限制。**测试方法**：实机跑，统计筛出帖数。
**测试结果**：s1 3 帖 / hn 5 帖（均非固定上限）。报告中文、含具名用户与楼层号。

---

## 问题 6：真实 token/耗时对比（map 直 vs 直出）

### 问题
需要量化「map（框架 pipeline）」与「opencode 直出」两种路线的成本差异，作为选择依据。

### 方案
同一批同源帖子跑两条路线，读真实 usage 与 opencode SQLite 数据对比。

### 修改
- `_http_base.py` 真实 token 捕获：`usage_log` + `get_usage_summary()`
  （`reasoning_tokens` / `visible_tokens` 拆分）。

### 测试（实测结果）
**测试方案**：S1/HN 各跑 map 与直出，记录耗时/token。**测试方法**：
`env -u HTTPS_PROXY -u HTTP_PROXY python examples/hn_ai_report/demo.py`
+ `opencode run --model opencode/hy3-free`。

| 指标 | s1 map | s1 直出 | hn map | hn 直出 |
|---|---|---|---|---|
| 耗时 | 321.5s （3 calls） | 418.2s（1 agent） | 101.6s（6 calls） | 304.0s（1 agent） |
| 总 tokens | 19,804 | 73,574 | 25,949 | 80,235 |
| input | 6,108 | 67,882 | 9,163 | 78,035 |
| completion | 13,696 | 5,692 | 16,786 | 2,200 |
| reasoning | 12,633 | 6,354 | 15,034 | 2,545 |
| visible | 1,063 | 5,692 | 1,752 | 2,200 |

**测试结果**：map 总消耗约直出的 **27%（s1）/ 32%（hn）**——直出 input 爆炸
（WebFetch 整页），map 只送 24h 回帖/前 15 条评论。

---

## 问题 7：proxy 与 base_url 的使用约定

### 问题
示例早期依赖环境变量代理（`HTTPS_PROXY`），不好复现；`base_url` 会被框架自动补
`/chat/completions`，与真实端不一致。

### 方案
- proxy 明确嵌入 config 的 `settings.https_proxy`（覆盖环境变量）；
- `base_url` 必须是完整 URL（含路径），`_endpoint_url(settings)` 原样使用，绝不自动拼。

### 修改
- `framework/agents/_http_base.py`：`_endpoint_url(settings)` 原样使用；`_client_for(settings)`
  按 proxy 缓存客户端。
- 各 config：`"https_proxy": "http://127.0.1.6:7890"`、`"base_url"` 完整。

### 测试
**测试方案**：config 内联 proxy 生效、不依赖环境变量。**测试方法**：
`env -u HTTPS_PROXY -u HTTP_PROXY python examples/hn_ai_report/demo.py`（无网络代理环境）
+ `grep https_proxy examples/hn_ai_report/config.json`。**测试结果**：正常联网出报告。

---

## 问题 8：fetch 步骤超时

### 问题
抓取步骤无超时控制，HN/ST 接口慢可能挂起整条管线。

### 方案
fetch 步骤从 settings 读取 `timeout`（默认 30s），可 per-step 声明。

### 修改
- `FetchCommentsEdge`/`FetchThreadsEdge`/`FetchEdge`：接受 `settings.timeout`。
- config pipeline 步骤可写 `"settings": {"timeout": 30}`。

### 测试
**测试方案**：超时值生效。**测试方法**：`tests/test_s1_edges.py` + 实机。**测试结果**：通过。

---

## 问题 9：文档残留 `agent` 字段与 per-edge agent 旧概念

### 问题
README/config 曾声称 edge 可从 settings 配 `agent`；实际 `Edge.__init__` 不再消费该字段，
agent 由脚本 Edge 子类 `__init__` 自持或走 Executor 级。

### 方案
文档与示例统一：**不要 `agent` 字段**；脚本 Edge 在 `__init__` 自持 agent。

### 修改
- 各 `report_hook.py` / `hn_edges.py` / `s1_edges.py`：`self.agent = ...` 在 `__init__` 内；
- README / ai_report_notes 删除 `settings.agent` 表述。

### 测试
**测试方案**：config 无 `agent` 字段且 edge 用自持 agent。**测试方法**：
`grep -rn '"agent"' examples/*/config.json`。**测试结果**：0 处；框架 0 处读取
`settings["agent"]`（`opencode_agent_runner.py` 的 `--agent` 是 CLI 参数，属保留）。

---

## 运行方式（四段式索引）

### 问题
跑法分散、proxy 依赖隐晦。

### 方案
统一一行命令；proxy 内嵌 config，无需环境变量。

### 修改
- `examples/hn_ai_report/demo.py`、`examples/s1_ai_report_map/demo.py`。

### 测试
**测试方法**：

```bash
env -u HTTPS_PROXY -u HTTP_PROXY python examples/hn_ai_report/demo.py
env -u HTTPS_PROXY -u HTTP_PROXY python examples/s1_ai_report_map/demo.py
HTTPS_PROXY=http://127.0.1.6:7890 opencode run --model opencode/hy3-free "$(cat /tmp/prompt.txt)"
```

**测试结果**：map 产出 `report.md`；直出产出 `opencode_direct.md`（数据见问题 6）。

---

## 相关提交（分支 vertex-edge-agent）

| 提交 | 内容 |
|---|---|
| `13b47a3` | hn 同步 s1 改进（中文 prompt、结构化标题、report_hook # [title](url)）；hn edges proxy/timeout；不限条数；新 map+直出报告 |
| `91b73a1` | settings 级显式 proxy；hn 实机跑 |
| `6683edf` | 显式 endpoint + fetch timeout；无回退填充 |
| `065ea5a` | 去掉 `## Thread N` + 时间线 |
| `90ea271` | chatto-bot 风格 prompt |
| `9aab7c8` | token 捕获 + demo 打印 |
| `75df373` | bs4 依赖 |
| `f4f7e17` | script_loader 显式类名 + 结构化标题 |
| `6a4d4e1` | 解析 bug + 测试 |
| `4cfedec` | 克隆 + MapEdge 修复 |

**当前测试**：**342 tests passed**。