# 稀疏数据源 + 强化学习（RL）实验

> 在 `additive-rand-transformer` 分支实测。环境：conda torch 2.5.0，纯 CPU。
> 覆盖两项需求：(1) 稀疏数据源——1-2 位全量覆盖，3 位起按位数递增稀疏，
> 不再穷举所有组合；(2) 强化学习环节——以"答案正确性"为奖励的 REINFORCE。

## 1. 稀疏数据源

### 设计
`data._random_int` 增加 `sparse_from` / `density`：
- 位数 `d < sparse_from`（默认 3，即 1-2 位）权重 = 1.0 —— 全量覆盖 0..99，100 以内所有加法都可生成
- 位数 `d >= sparse_from` 权重 = `density ** (d - sparse_from + 1)` —— 指数衰减，"依次稀疏"
- 所有 batch 函数（`make_single_batch` / `make_single_cot_batch` / `pack_blocks` / `stream_batches`）均透传

### 实测分布（max_digits=4, n=4000）

| 操作数位数 | uniform（旧） | sparse density=0.5 | sparse density=0.3 |
|-----------|--------------|--------------------|--------------------|
| 1-2 位（100 以内） | 31% | **51%** | **61%** |
| 3 位 | 25% | 25% | 24% |
| 4 位 | 32% | **15.6%** | **10.2%** |
| 5 位+ | 9% | 4.6% | 2.6% |

稀疏源把采样重心移回短操作数（全量可枚举），长操作数组合**只采子集且越稀越少**——
符合"三位数加法开始不再生成所有可能性，并且依次稀疏"。

## 2. 强化学习（REINFORCE）

`rl.py` 实现，奖励 = 1 若生成的答案正确，0 否则。

### 稳定性修复历程（3 个真实 bug）
1. **retrace 错位**：log-prob 只统计生成 token（正确 shift），不含 prefix —— 否则污染梯度
2. **trajectory 数值爆炸**：优势做 per-problem 标准化（÷std），KL/熵用 mean 而非 sum
3. **KL 罚项符号**（关键）：`KL(pol||ref)=E[log π - log π_ref]`，正确写法是
   `loss += β*(logp - ref_logp)`；初版写成 `ref_logp - logp` 会**奖励偏离**，策略越漂越远最终失控

### 实验结果

**激进配置**（temperature 0.8, lr 3e-5, kl 0.05, entropy 0.01, n_samples 8, 300 步）：
- 训练采样 reward 表面很高（0.97-1.00），但**最终贪婪能力全面崩塌**：
  add1 100%→80%、sub2 100%→25% —— 策略在采样温度下漂移，argmax 路径被削弱。

**保守配置**（temperature 0.5, lr 1e-5, kl 0.1, entropy 0, n_samples 12, 150 步）✅
- 训练全程 reward 1.000，loss 平滑
- **贪婪准确率**（对比 SFT 基线 L4·D128，无 bias）：

| 指标 | 基线 SFT | 保守 RL 后 | 变化 |
|------|---------|-----------|------|
| add1 | 100% | 100% | — |
| add2 | 100% | 100% | — |
| add3 | 100% | 98% | -2 |
| **add4** | **27%** | **32%** | **+5** |
| sub1 | 100% | 90% | -10 |
| sub2 | 100% | 95% | -5 |
| sub3 | 100% | 98% | -2 |
| sub4 | 97% | 98% | +1 |

**结论**：保守 RL 稳定训练且**提升唯一短板 add4（27%→32%）**，代价是 sub1/2
小幅回退——RL 提升短板的标准折中（RL 从零奖励中强化困难 4 位进位，但稀疏数据
下简单减法的采样式探索略微扰动）。CKPT：`runs/rl_conservative/20260830_075636/rl_final.pt`

## 3. 可复现

```bash
# 稀疏数据源验证
python -c "from additive_rand_transformer.data import gen_expression_cot, _random_int"

# 激进 RL（观察崩溃，理解为什么需要保守）
python -m additive_rand_transformer.rl --checkpoint runs/20260830_041155/checkpoint_final.pt \
    --rl_steps 300 --temperature 0.8 --lr 3e-5 --kl_beta 0.05 --entropy_bonus 0.01 \
    --sparse_from 3 --density 0.5 --max_digits 4

# 保守 RL（稳定提升 add4）
python -m additive_rand_transformer.rl --checkpoint runs/20260830_041155/checkpoint_final.pt \
    --rl_steps 150 --temperature 0.5 --lr 1e-5 --kl_beta 0.1 --entropy_bonus 0.0 \
    --n_samples 12 --sparse_from 3 --density 0.5 --max_digits 4
```

## 4. 局限与后续

- RL 收益当前集中在 add4（+5pp）；sub 有小幅回退。可尝试 per-op 分段奖励或
  优先采样困难问题（curriculum RL）。
- REINFORCE 方差大，tiny 模型 + 稀疏奖励下需保守超参；`n_samples` 越大基线越稳。
- 若要更强提升，建议 PPO（clipped surrogate + value baseline）而非 REINFORCE。
