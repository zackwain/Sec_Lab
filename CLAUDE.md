# Sec_Lab — 时间自适应 DP-FL 隐私预算调度

## 一句话概述

> 差分隐私联邦学习中，每个客户端有固定总隐私预算 ε=6.0。如何跨训练轮次分配？对标 Kiani et al. (ICLR 2025) 的**固定调度**，提出**损失动量驱动的信号调度**——根据 loss 变化趋势自动决定每轮该省还是该花，在不消耗额外隐私的前提下，比均匀分配和固定调度获得更高准确率。

## 环境

- Windows 11, conda env `SecLab_env`
- Python 3.13, PyTorch 2.12.0 **CUDA 12.6**, Opacus 1.4.0
- GPU: RTX 3060 (6GB), Driver CUDA 13.1
- 项目路径: `C:\Users\zackwain\Desktop\Sec_Lab`
- **必须设置**: `KMP_DUPLICATE_LIB_OK=TRUE`

## 目录结构

```
Sec_Lab/
├── CLAUDE.md                      # 本文件 — 项目框架和规范
├── PROGRESS.md                    # 当前进度（对话交接用）
├── 2502.18706v1.pdf              # Kiani et al. ICLR 2025 参考论文
│
├── step1_verify/                  # [历史] Flower 环境验证
├── step2_baseline/                # [复用] CNN 模型 + MNIST 数据加载
│   ├── model.py                  # MNISTCNN 定义 + get/set_parameters
│   └── data.py                   # load_mnist, split_data, get_test_loader
│
├── step3_adaptive/                # [废弃] 旧方案 — 跨客户端 ε 重新分配（PRV 会计错误）
│
├── step4_time_adaptive/           # [当前] 时间自适应 DP-FL
│   ├── config.py                 # 所有超参数（改一处全局生效）
│   ├── scheduler.py              # 三种 ε 调度策略 + ε→σ 转换 + 损失动量
│   ├── trainer.py                # DP 训练核心 + 评估 + 完整训练循环
│   └── experiment.py             # 主实验脚本
│
└── results/
    ├── logs/step4_v1_experiment.json  # 主实验结果
    └── figures/                       # 图表（待生成）
```

## 核心框架

### 三种策略（同一总预算 ε=6.0，同一模型，同一数据）

```
Strategy A: Uniform（基线）
  每轮 ε = 6.0/20 = 0.3，不变

Strategy B: Kiani Linear（对标 ICLR 2025）
  ε 权重从 0.5→1.5 线性递增，归一化 Σ=6.0
  前期 ε≈0.15（省钱），后期 ε≈0.45（花钱）

Strategy C: Kiani + Loss Momentum（OURS）
  ε(r) = ε_kiani(r) × exp(γ × momentum(r)), γ=2.0
  momentum = 损失下降率 × exp(加速度×10)
  loss 加速下降 → 加注；趋于平缓 → 自动省钱
```

### 损失动量计算

```
给定最近 4 轮 loss 历史 [L_{r-3}, ..., L_r]:
  drops[i] = (L_i - L_{i+1}) / L_i         # 逐轮下降率
  current_drop = drops[-1]                  # 一阶：当前速度
  acceleration = drops[-1] - drops[-2]      # 二阶：加速度
  momentum = current_drop × exp(acceleration × 10)
  → 截断到 [-1, 1]
```

### 隐私会计

**基本组合定理**：每轮创建独立 PrivacyEngine，总隐私 = (Σε_r, R×δ)-DP。保守但正确，三策略公平可比。

### 训练方式

每轮创建全新 `MNISTCNN` → 加载 master_params → DP-SGD → 返回新参数。避免 Opacus hook 重复注册错误。

### ε→σ 校准

使用 Opacus `get_noise_multiplier(target_epsilon, delta, sample_rate, epochs)` 反算噪声乘数 σ。当前 batch=600, q=0.01，近似误差约 30%（ε_allocated=0.30 → ε_spent≈0.22），三策略一致性高，对比公平。

## 当前实验结果

| 策略 | 准确率 | Σε | 轮次 |
|---|---|---|---|
| Uniform | 92.24% | 4.36 | 20 |
| Kiani Linear | 92.04% | 4.33 | 20 |
| **Kiani + Momentum** | **92.85%** | **4.29** | 20 |

**结论**：信号驱动方法准确率最高 + ε 消耗最低。

**KianiLinear 为什么低于 Uniform**：MNIST CNN 收敛太快（5-7 轮），Kiani 的省钱期预算削减拖累了早期学习，后期多花的边际收益已归零。这反映了固定调度的弱点——预设节奏不适应实际学习动态。

## 已知问题

1. **收敛太快**：MNIST + 1 epoch/轮 → 模型快速收敛，调度策略差异窗口短
2. **ε 消耗不足**：每轮实际消耗 ≈0.22 vs 分配 0.30，总 ε≈4.3 未达 6.0
3. **KianiLinear < Uniform**：需增加 epochs/轮或换更难的数据集

## 重要规则（新对话必须遵守）

1. **不要**引用 step3_adaptive 的旧逻辑（跨客户端 ε 池子在 DP 理论中不存在）
2. **不要**用 `get_noise_multiplier` 做精确 ε 控制，它是近似值
3. **不要**跨客户端重新分配 ε——每个客户端独立消耗自己的预算
4. 当前是**固定 ε 版本**（分配 ε，推导 σ），固定 σ 版本已删除
5. 修改配置走 `config.py`，不要在各文件里硬编码参数
6. 每轮训练**必须创建新模型**，避免 Opacus hook 冲突
7. 对比论文是 `2502.18706v1.pdf`（Kiani et al. ICLR 2025）
