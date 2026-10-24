# troll_tea_party：B 站 URL → 深度阅读笔记 DAG

一条只吃 URL 的 v8 DAG：B 站 URL → 6.8 用 bash 工具取元信息与 ASR 字幕 →
**并行两路事实/结构计划** → 初稿 → **并行两路事实/格式审校** → 定稿（fan-in）→
确定性收口 → 落盘 Markdown。**DAG 不读任何参考笔记**，输入只有 `seed_tag = "url"`。

## 跑法

```bash
cd /mnt/c/Users/lumin/Downloads/agent/v8
python3 -m dag8 run configs/troll_tea_party/dag.json --cache --workers 2
# 换一个视频：
python3 -m dag8 run configs/troll_tea_party/dag.json \
  --seed "https://www.bilibili.com/video/BVxxxxxxxxxx/" --workers 2
```

产物：

```text
configs/troll_tea_party/out/troll_tea_party_notes.md   # 定稿
configs/troll_tea_party/out/latest.md                  # 同内容，便于脚本取
```

`--workers 2` 是这张图有意义的前提：layer=1 的 `facts` / `outline` 真正并行，
layer=3 的 `fact_check` / `style_check` 也真正并行；默认 `--workers 1` 会退化成
逐条串行，但依赖关系不变。

## 图

```text
url
 └─ fetch  (pi, sensenova/6.8, tools=true)
      bash: cd configs/troll_tea_party && python3 fetch_bili.py "{url}" 2>/dev/null
      └─ source
           ├─ facts    (pi, laptop/6.8)  事实骨架
           └─ outline  (pi, laptop/6.8)  结构计划
                └─ draft  (pi, laptop/6.8)  source + facts + outline
                     ├─ fact_check   (pi, laptop/6.8)  对照 source 查事实
                     └─ style_check  (pi, laptop/6.8)  查结构与格式
                          └─ final  (pi, laptop/6.8)  source + draft + facts + 两份审校
                               └─ trim  (command, stdin)
                                    └─ save  (command, stdin)
```

- `fetch`：6.8 自己用 bash 工具跑 `fetch_bili.py`，把 stdout 原样回填。
- `facts` / `outline`：同一份 source 的两个独立视角，互不依赖，可以并行。
- `draft`：等两路都完成再写，避免“先写再补结构”。
- `fact_check` / `style_check`：对初稿的两路独立审校，互不依赖，可以并行。
- `final`：把 source、初稿、事实骨架、两份审校一起 fan-in，逐条修问题。
- `trim`：切掉代码块之后的内容、去 `<header>`、归一化标题/空行。
- `save`：兜底校验后写 `out/`。

## 为什么以前看起来是“一本道”

第一版确实是一条 `fetch → draft → final` 的直线：每一步的输入都只来自上一步，
v8 调度器按 tag 精确匹配，没有独立分支可并行。但 **DAG 只要求无环，不要求窄**；
把“事实提取”“结构计划”“事实审校”“格式审校”从主链里拆出来，声明成只依赖
source / draft 的独立 tag，调度器就会在同一波次并行执行，最后在 final 做 fan-in。
所以不是 DAG 只能一本道，是之前的依赖声明只写了一条链。

## 6.8 工具：之前为什么 400，怎么修的

不是模型不支持工具，是 provider 和 reasoning 字段两个问题叠在一起：

1. `laptop` 代理上的 6.8 带工具返回 400；
2. 换到 `sensenova` 直连 `https://token.sensenova.cn/v1` 仍 400；
3. 二分请求体发现：工具结果回放时，assistant 消息同时带 `reasoning` 与
   `reasoning_content`，该端点不接受这一对字段同时出现；删任意一个就通过。

`~/.pi/agent/models.json` 的关键配置：

```json
"sensenova": {
  "baseUrl": "https://token.sensenova.cn/v1",
  "api": "openai-completions",
  "apiKey": "!grep -m1 'SENSENOVA_API_KEY:' /home/lumin/.dsh/.credentials.yaml | sed 's/.*: //'",
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": true,
    "thinkingFormat": "deepseek",
    "requiresReasoningContentOnAssistantMessages": false
  },
  "models": [
    {"id": "sensenova-6.8-flash-lite", "name": "SenseNova 6.8 Flash Lite",
     "contextWindow": 262144, "input": ["text"], "reasoning": true, "maxTokens": 65536}
  ]
}
```

验证：

```bash
pi -p --provider sensenova --model sensenova-6.8-flash-lite \
  --tools bash <<< 'Run `echo TOOL_OK` and reply with its output only.'
```

## 双 provider 与 thinking=off

- `fetch` 走 `sensenova` 直连：短工具调用快（取字幕约 20s），且只有它支持工具。
- `draft` / `final` 走 `laptop`：同一 6.8，长文更稳；直连长文两次撞满 1200s 预算。
- `pi.thinking = "off"`：不关 thinking 时 final 会长时间空转；显式关掉后 final 约 85s。

两个 provider 都是 `sensenova-6.8-flash-lite`，没有换模型。

## 实测记录（2026-10-23/24）

| 节点 | 读数 |
|---|---|
| fetch | 首次 20.7s，2679 字符（6.8 + bash 工具）；之后命中缓存 |
| facts | 1623 字符 |
| outline | 30.7s，3120 字符 |
| draft | 185.5s，7600 字符 |
| fact_check | 123.1s，946 字符 |
| style_check | 21.5s，681 字符 |
| final | 211.0s，6485 字符；契约通过 |
| trim | 6480 字符；标题/空行归一化 |
| 目标笔记 | 5685 字符；本 DAG 6480 字符，结构与长度都贴近 |

layer=1 的两路和 layer=3 的两路由 `--workers 2` 并行；串行也能跑，只是慢。

## 三个容易复发的坑

1. **不要把“第二章 4 条”和“3.2 五阶段”写混**。两者都要求编号列表，模型会把
   第二章的四个摘要段落复制进 3.2。prompt 里必须显式写“3.2 只写 SOP 五阶段，
   内容必须与第二章不同，绝不要复制第二章”。
2. **抽象的结构约束会把开篇论证挤掉**。只写“第二章写历史与现实的镜像”时，
   模型会把引子、《过秦论》、三苏压成一句“古文里反复写过”。必须在 prompt 里
   点名“第 1 条先写引子，再写贾谊《过秦论》与三苏（苏洵、苏轼、苏辙）”。
3. **元信息时间以页面 JSON-LD 的 `dateCreated` 为准**。`view` 接口的 `pubdate`
   可能是后续编辑/定时发布的时间；本视频两者差一天多。`fetch_bili.py` 优先取
   页面 `dateCreated`，拿不到才退回 `view.pubdate`。

## 相关文件

- `configs/troll_tea_party/dag.json` — 分支 DAG 定义与各节点 prompt
- `configs/troll_tea_party/fetch_bili.py` — URL → 元信息 + 字幕（被 6.8 的 bash 工具调用）
- `configs/troll_tea_party/trim_md.py` — 切结语 + 标题/空行归一化
- `configs/troll_tea_party/save_md.py` — 定稿落盘与结构兜底
- `dag8/nodes.py` 的 `CommandNode` — `kind: "command"` 执行体
- `dag8/pi.py` / `dag8/config.py` — `thinking` 字段与 `--thinking` 透传
- `tests/test_dag8.py` — 离线测试
