# Sec_Lab 项目进度 — 2026-05-30

## 一句话概述

DP-FL 时间自适应隐私预算调度。核心理念：每个客户端固定总预算 ε=6.0，在训练轮次间非均匀分配。对标 Kiani et al. ICLR 2025 的固定调度，提出**损失动量驱动的信号调度**——根据训练过程中的 loss 变化趋势自动决定每轮该花多少预算。

## 环境

- Windows 11, conda env `SecLab_env`
- Python 3.13, PyTorch 2.12.0 **CUDA 12.6**, Opacus 1.4.0
- GPU: RTX 3060 (6GB), Driver CUDA 13.1
- 项目路径: `C:\Users\zackwain\Desktop\Sec_Lab`
- `KMP_DUPLICATE_LIB_OK=TRUE`（必须设置）

## 当前状态

**固定 ε 主实验已完成**，结果正向。MIA 和可视化待做。

## 核心实验结果

| 策略 | 准确率 | Σε | 轮次 | 含义 |
|---|---|---|---|---|
| Uniform | 92.24% | 4.36 | 20 | 每轮均分 ε=0.3（基线） |
| KianiLinear | 92.04% | 4.33 | 20 | ε 从 0.15→0.45 线性递增（对标 ICLR 2025） |
| **Kiani+Momentum** | **92.85%** | **4.29** | 20 | Kiani 基调 × 损失动量指数调节（OURS）|

**核心结论**：我们的方法准确率最高 + ε 消耗最低 = 双向优势。

**KianiLinear 为什么不如 Uniform**：MNIST CNN 太容易，5-7 轮就基本收敛。Kiani 的"前期省钱后期花"策略在 1 epoch/轮的设置下，省钱期的预算削减反而拖累了学习，后期花钱时边际收益已归零。Kiani 论文用的是 L=30 epoch/轮，模型有持续进步的空间。这反而是我们方法的优势——**固定调度假设了特定的训练节奏，信号驱动方法自动适应**。

## 目录结构

```
Sec_Lab/
├── step1_verify/              # 历史参考 — Flower 环境验证
├── step2_baseline/            # model.py + data.py（正确，被复用）
├── step3_adaptive/            # 旧方案 — 跨客户端 ε 重新分配（废弃）
│
├── step4_time_adaptive/       # [当前活跃] 时间自适应 DP-FL
│   ├── config.py              # 所有超参数（BATCH_SIZE=600, GPU, ε=6.0, γ=2.0）
│   ├── scheduler.py           # 三种 ε 调度策略 + ε→σ 转换 + 损失动量计算
│   ├── trainer.py             # DP 训练核心 + 评估 + 完整训练循环
│   └── experiment.py          # 主实验脚本（跑这个）
│
├── results/
│   ├── logs/step4_v1_experiment.json   # 主实验结果
│   ├── logs/step4_v1_fixed_sigma.json  # 固定 σ 实验（补充，结果一般）
│   └── figures/                        # 图表（待生成）
│
├── docs/superpowers/plans/    # 实现计划
└── CLAUDE.md                  # 旧版项目文档（部分过时）
```

## 技术要点（新对话需要知道的）

### 1. 隐私会计方式：基本组合

每轮创建独立的 PrivacyEngine。总隐私 = (Σε_r, R×δ)-DP，直接加。保守但公平，三个策略在同一天平上比较。

### 2. 训练方式

每轮创建全新 `local_model` → 加载 master_params → DP-SGD 训练 → 返回新 params → 更新 master_params。避免 Opacus hook 重复添加错误。

### 3. ε→σ 校准

使用 Opacus 的 `get_noise_multiplier(target_epsilon, delta, sample_rate, epochs)` 反算 σ。当前 batch=600, q=0.01，近似误差约 30%（ε_allocated=0.30 → ε_spent≈0.22）。三个策略一致性高，对比公平。

### 4. 三种策略的调度逻辑

```
Uniform:      每轮 ε = 6.0/20 = 0.3
KianiLinear:  权重从 0.5→1.5 线性增长，归一化 Σ=6.0
Kiani+Momentum: kiani_base × exp(2.0 × momentum)
                momentum = 损失下降率 × exp(加速度×10)
                加速下降 → 加注；趋于平缓 → 省钱
```

### 5. 信号驱动策略的在线计算

KianiPlusMomentum 不是预分配 schedule，而是每轮根据累积的 loss_history 动态计算下轮 ε。第一轮退化为 KianiLinear 起点。

## 已知问题

1. **收敛太快**：MNIST CNN 1 epoch/轮 → 5-7 轮就接近收敛，Kiani 的省钱/花钱策略来不及发挥。可能需要增加 `EPOCHS_PER_ROUND` 或换 CIFAR-10。
2. **ε 未达预算上限**：每轮 ε_spent≈0.22 vs allocated=0.30，总 ε≈4.3 而非 6.0。`get_noise_multiplier` 在 q 中等时的近似误差。三策略误差一致，对比公平。
3. **KianiLinear 低于 Uniform**：原因见上"收敛太快"分析。

## 待做（优先级排序）

- [ ] **改进模型效果**：尝试 EPOCHS_PER_ROUND=3 或换数据集，让 Kiani 策略有时间发挥作用
- [ ] **MIA 隐私验证**：重写到新文件夹，避免与 step4 耦合
- [ ] **可视化**：新文件夹单独做，从 JSON 读取结果
- [ ] 多次重复实验（取均值±标准差）
- [ ] 消融实验（不同 ε_total、不同 γ）
- [ ] 答辩 PPT/报告

## 重要提醒

- 不要用旧方案（step3_adaptive）的逻辑，跨客户端 ε 池子在 DP 理论中不存在
- 不要用 `get_noise_multiplier` 做精确校准，它只是近似
- 当前是**固定 ε** 版本（控制 ε，测量模型效果）。**固定 σ** 版本已删除
- 新对话如果需要参考对比论文：`2502.18706v1.pdf`（Kiani et al. ICLR 2025）
