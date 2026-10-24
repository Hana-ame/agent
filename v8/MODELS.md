# v8 可用模型清单

实测时间 2026-10-23。判据是**真的用 `pi` 发一次请求**，不是看 `/v1/models` 列表。

## 实际可用：`laptop`，5 个模型

| provider | model | 可用 | 上下文 | 输出上限 | 备注 |
|---|---|---|---|---|---|
| `laptop` | `deepseek-v4-flash` | ✅ | 1048576 | 384000 | 最快，v8 默认 |
| `laptop` | `glm-5.2` | ✅ | 1048576 | 131072 | |
| `laptop` | `kimi-k3` | ✅ | 1048576 | 1048576 | 输出上限最大 |
| `laptop` | `deepseek-flash` | ✅ | 1048576 | 393216 | |
| `laptop` | `sensenova-6.8-flash-lite` | ✅ | 262144 | 65536 | |

## 已移除

| provider | model | 移除原因 |
|---|---|---|
| `laptop` | `deepseek-v4.1-flash` | `403 model is not available in the current token plan` |

移除后行为已验证：`laptop/deepseek-v4.1-flash` 报 `Model "deepseek-v4.1-flash" not found`，
立刻失败，不会静默等 timeout。

## `sensenova` provider：直连可用，且 6.8 支持工具

`sensenova` 现在直连 `https://token.sensenova.cn/v1`，`apiKey` 从本机
`~/.dsh/.credentials.yaml` 取（命令插值，不把 key 写进文件）：

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
  "models": [ ...同 laptop 的 5 个... ]
}
```

关键：`requiresReasoningContentOnAssistantMessages` 必须是 `false`。pi 在工具结果
回放时会同时带 `reasoning` 与 `reasoning_content`，该端点不接受这一对字段同时出现，
表现为带工具就 400 `inference request is invalid`。改 `false` 后：

```bash
pi -p --provider sensenova --model sensenova-6.8-flash-lite --tools bash
# 能执行 echo 并把结果回给模型
```

注意：`laptop` 代理上的同一个 6.8 带工具仍会 400；工具要走 `sensenova` 直连。

⚠️ **不要把 `apiKey` 写成 `""`**。pi 的 schema 要求 ≥1 字符，空串会让**整个**
`models.json` 校验失败，连 `laptop` 的 5 个模型一起消失：

```
Invalid models.json schema:
- providers.sensenova.apiKey: must not have fewer than 1 characters
No models available.
```

「不设 key」= 省略字段；要占位就写 `"not-needed"` 这种非空串。

## 写进 v8 配置的样子

```json
"pi": {
  "provider": "laptop",
  "model": "deepseek-v4-flash",
  "tools": false,
  "timeout": 240,
  "retries": 3,
  "ssh_host": "172.29.89.192",
  "ssh_key": "~/.ssh/id_rsa"
}
```

## ⚠️ 模型名写错不会报错

`pi` 对未注册的模型会打印 `Warning: Model "x" not found for provider "laptop". Using custom model id.`
然后**照常透传给网关**。行为分两种：

- 网关恰好有这个名字 → 正常返回（`deepseek-flash` 早期就是这种情况）
- 网关没有 → 网关侧报 `404 model is not found`

危险的是第一种：拼写错了但碰巧命中，或者拼错了却被 `pi` 当 custom id 透传，
**`pi` 的退出码仍是 0**，v8 看不出异常。

因此跑之前先预检：

```bash
cd v8
python -m dag8 run configs/xxx.json --check-model   # 只验证 provider/model 可用，不跑边
```

## 上限是怎么测的

直接改请求 body 的 `max_tokens` 往上打，超限时服务端自己报范围：

```
field MaxTokens invalid, should be in [1, 384000]
```

括号里的数就是真上限。**不要**拿 `/v1/models` 的 `max_output_length` 当依据 ——
它把 `kimi-k3` 和 `deepseek-v4-flash` 都低估成 65536（实际分别是 1048576 和 384000）。

注意探测时要区分 `rpm`/`tpm` 限流错误，否则会把限流误当成上限。

## 速度实测（2026-10-23）

同一张简单图（1 个 pi 节点，写 150~250 字短文），各 3 次取中位数：

| 模型 | 时延 p50 | 质量（评审 3~15） |
|---|---:|---:|
| `deepseek-flash` | **6.20s** | 15 |
| `deepseek-v4-flash` | 8.27s | 15 |
| `glm-5.2` | 34.18s | 15 |

结论：简单任务上 `deepseek-flash` 比 `deepseek-v4-flash` 快约 25%，
「v4-flash 最快」的旧说法不成立。以 `scripts/search_schedule.py --models` 的实测为准。
