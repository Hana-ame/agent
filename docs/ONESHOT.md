# 一次 shot 自举：怎么跑，以及怎么判

## 一句话现状

框架侧全部闭合（36 测试绿、四代逐字节一致、固定点自校验含 sha+bytes 双核），
**但真正的 `/goal` 判据还差最后一步：一次 shot 用真模型连续生成三代整个 `vea_v6`。**
卡点只有一个 —— 没有一个可用的真实端点（8000 无服务、sensenova 无 key）。

## 跑之前：填端点

编辑 `models.json`（已被 gitignore，模板见 `models.json.example`）：

```json
{
  "laptop-6.8": {
    "api": "http://127.0.0.1:8000/v1",
    "key": "not-needed",
    "model": "<8000 上真实暴露的模型名>",
    "params": {"temperature": 0.2, "max_tokens": 8192}
  },
  "sensenova-6.8": {
    "api": "https://api.sensenova.cn/v1",
    "key": "<真实 key>",
    "model": "sensenova-6.8-flash-lite",
    "params": {"temperature": 0.2, "max_tokens": 8192}
  }
}
```

先用 `/v1/models` 确认模型名对得上：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

## 判定命令

```bash
# 正式判定：一次 shot，连续 3 代整个框架逐字节一致
python tools/oneshot_chain.py --generations 3 --model-key laptop-6.8

# 卡顿就换 model、不换 api
python tools/oneshot_chain.py --generations 3 --model-key sensenova-6.8

# 只验磁盘上的框架能否连续自举（不碰模型）
python tools/oneshot_chain.py --generations 3 --offline
```

通过的输出长这样（`gen0` 由模型生成，`gen1~gen3` 由上一代自己的
`generate_framework()` 重建，四代同一 `combined_sha256`）：

```
HTTP 7 次，端点 ['http://127.0.0.1:8000/v1']，耗时 41.3s
gen0  <sha16>  <bytes>B  自校验=OK
gen1  <sha16>  <bytes>B  == gen0  自校验=OK
gen2  <sha16>  <bytes>B  == gen1  自校验=OK
gen3  <sha16>  <bytes>B  == gen2  自校验=OK

通过：一次 shot 连续 3 代逐字节一致（模型 laptop-6.8）
```

`HTTP N 次` 是硬断言：0 次直接判失败（`DAG()` 在 `llm=None` 时会静默退化成
`SmartMockLLM`，那种 1 秒"跑通"是假通过）。

## 回档

每一代都落盘到 `bootstrap_runs/<run-id>/gen<N>/`，带 `HEAD.json`
（`parent` 指针 + 各文件 sha256/字节数 + 合并指纹），是一条可校验的链。

```bash
python tools/boot_rollback.py list                 # 所有 run 与各代
python tools/boot_rollback.py show <run> 2         # 某一代详情
python tools/boot_rollback.py diff <run> 0 3       # 哪一代改了哪个文件
python tools/boot_rollback.py verify <run>         # 校验整条链
python tools/boot_rollback.py checkout <run> 2     # 回档到工作区（先备份当前）
```

`checkout` 不删任何东西：先把当前 `vea_v6/` 整份复制到
`bootstrap_runs/_before_checkout_<时间戳>/`，再写入目标代。

## 没有真端点时怎么验链路

`tools/mock_openai_server.py` 是一个**只验链路**的假端点（不生成任何东西，
按 prompt 里的 `文件：<path>` 精确回原文）。它的意义是把"除模型以外的一切"
先验掉：

```bash
python tools/mock_openai_server.py --port 8000   # 终端 A
python tools/oneshot_chain.py --generations 3    # 终端 B
```

**注意**：假端点跑通只说明 DAG/HTTP/解析/assemble/自校验这条链没断，
不说明模型能写出能自举的框架。不要把它的结果当作 `/goal` 通过。

## 代码侧的四个闸门

```bash
python tools/fixpoint_v3.py --check   # 固定点自校验（清单 sha + bytes 逐项核对 + 二代复现）
python tools/gen3_probe.py            # 内存内连续四代自举并逐字节比对
python -m pytest tests/ -q            # 36 个行为测试
python tools/oneshot_chain.py --offline --generations 3   # 磁盘框架连续三代
```
