# Sec_Lab 项目进度 — 2026-05-29

## 当前状态

Step 3 第二版（V2）已跑完，自适应未超过均匀，正在推进 V3 修复。

## 已完成

- [x] Step 1: Flower 环境验证
- [x] Step 2: 基线无DP FL → 96.05% (10轮 IID)
- [x] Step 3 V1: 熵+多样性 → 自适应 80.24% vs 均匀(未跑) → 方向反了
- [x] Step 3 V2: KL+损失下降+梯度对齐 → 自适应 92.91% vs 均匀 93.64% → 未超过

## V2 失败原因

| 问题 | 根因 |
|---|---|
| 自适应后期被反超 | 高分客户端边际收益递减 |
| 梯度对齐永久为 0（2/10客户端） | Non-IID 下稀缺客户端梯度天然不同 |
| 损失下降无效 | 没有过拟合惩罚 |
| ε 差距太小 | 线性分配 → 2.9~5.7 (仅2x) |

## V3 修复方案 → PLAN_V3.md

1. KL 门控对齐：`align_score = raw_align × (1 - KL_norm)`
2. 损失下降过拟合惩罚：`loss_score = raw_drop × min(1, raw_drop/0.1)`
3. 指数分配：`ε_i ∝ exp(Score_i / T)`, T=0.5
4. EMA 平滑 + 早停（收敛OR过拟合）

## V3 改动范围

| 文件 | 状态 |
|---|---|
| `quality_estimator.py` | 待改 — align × (1-KL), loss 加惩罚 |
| `allocator.py` | 待改 — 新增 allocate_exponential |
| `run_adaptive.py` | 待改 — 指数分配 + EMA早停 |

## 待做

- [ ] 完成 V3 修改
- [ ] 运行 V3 实验
- [ ] Step 4: exp1_comparison, exp2_ablation, exp3_mia, plot
