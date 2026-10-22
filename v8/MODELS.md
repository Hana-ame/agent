# v8 可用模型清单

实测时间 2026-10-23。判据是**真的用 `pi` 发一次请求**，不是看 `/v1/models` 列表。

## 结论：v8 只能用 `laptop` 这一家

| provider | model | v8 可用 | 上下文 | 输出上限 | 备注 |
|---|---|---|---|---|---|
| `laptop` | `deepseek-v4-flash` | ✅ | 1048576 | 384000 | 最快，v8 默认 |
| `laptop` | `glm-5.2` | ✅ | 1048576 | 131072 | |
| `laptop` | `kimi-k3` | ✅ | 1048576 | 1048576 | 输出上限最大 |
| `laptop` | `deepseek-flash` | ✅ | 1048576 | 393216 | |
| `laptop` | `sensenova-6.8-flash-lite` | ✅ | 262144 | 65536 | |
| `laptop` | `deepseek-v4.1-flash` | ❌ | — | — | `403 model is not available in the current token plan` |
| `sensenova` | `deepseek-flash` | ❌ | — | — | 挂死，45s 无任何输出 |
| `sensenova` | `sensenova-6.8-flash-lite` | ❌ | — | — | 挂死，45s 无任何输出 |

**`sensenova` 这个 provider 整个不可用**，两个模型都表现为静默挂死（不返回内容、不报错、不退出）。
别写进配置，会白等 `pi.timeout` 秒然后失败。

`deepseek-v4.1-flash` 在网关的 `/v1/models` 里列着，但实际调用被 token plan 拒掉。

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

- 网关恰好有这个名字 → 正常返回（`deepseek-flash` 就是这样，之前没注册也能跑）
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
