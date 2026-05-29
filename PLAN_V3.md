# V3 修复计划 — 多因素自适应 DP-FL

## 背景

V2 结果: Uniform 93.64% vs Adaptive 92.91%，Δ = -0.73%。

## 根因

| 问题 | 原因 |
|---|---|
| 梯度对齐永久为 0 | Non-IID 下稀缺客户端梯度方向天然不同，被 (1-KL) 门控误杀 |
| 损失下降无效 | 没有对过拟合做惩罚，学不动的客户端仍在消耗预算 |
| 分配差距太小 | 线性比例分配 → ε 范围仅 2.9~5.7 (2x) |
| 后期反超 | 高分客户端边际收益递减，低分客户端永远没机会 |

## 修复方案（4 项改动）

### 1. KL 门控对齐

```
旧:  align_score = raw_align                          ← KL 高也被惩罚
新:  align_score = raw_align × (1 - KL_norm)         ← KL 高 → 门窄 → 不对齐不扣分
```

### 2. 损失下降过拟合惩罚

```
旧:  loss_score = (loss_before - loss_after) / loss_before

新:  raw_drop = (loss_before - loss_after) / loss_before
     loss_score = raw_drop × min(1, raw_drop / 0.1)
     
     raw ≥ 0.1 → ×1.0 (正常)
     raw = 0.05 → ×0.5 (学得慢)
     raw = 0.02 → ×0.2 (几乎不动)
     raw = 0.01 → ×0.1 (学不动)
```

### 3. 指数分配（softmax + 温度）

```
旧:  ε_i = budget × Score_i / ΣScore_j     ← 线性，差距小

新:  ε_i = budget × exp(Score_i / T) / Σ exp(Score_j / T)
     T = 0.5
```

### 4. EMA 平滑 + 早停

```
每轮 EMA: smooth = 0.3×acc + 0.7×smooth_prev

收敛: 最近5轮best - 前5轮best < 0.002 → 停止
过拟合: 连续2轮 smooth < best_ever - 0.003 → 停止
```

## 改动范围

| 文件 | 改动 |
|---|---|
| `quality_estimator.py` | 修改 `compute_multi_factor_score`：align × (1-KL)，loss 加惩罚 |
| `allocator.py` | 新增 `allocate_exponential(scores, budget, T)` |
| `run_adaptive.py` | 主线用指数分配 + EMA早停 |

## 验证

```
Uniform DP vs Adaptive DP (相同数据、相同参数)
期望: Adaptive 准确率更高 + 收敛轮数更少
```
