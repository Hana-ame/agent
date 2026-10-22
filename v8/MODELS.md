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

## `sensenova` provider 现状：已登记，但未配 key

`sensenova` 是 `laptop` 配置的整份复制，只换了 `baseUrl`，`apiKey` 字段**整个省略**：

```json
"sensenova": {
  "baseUrl": "https://sensenova.moonchan.xyz/v1",
  "api": "openai-completions",
  "compat": { ...同 laptop... },
  "models": [ ...同 laptop 的 5 个... ]
}
```

调用行为：`No API key found for sensenova.`，**约 1 秒内失败**。

配 auth 就能用，两种方式：
- `pi` 里 `/login` 选 sensenova
- 或命令行传 `--api-key`

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
